"""Nirnay decision head.

Frozen encoder outputs come in for three kinds of text, each encoded on its own:
  state     the thing being judged (one per call, shared by every question)
  question  "<type> question: <instructions>"         (cached per schema)
  option    "<label>: <description>" / level / yes-no  (cached per schema)

Fusion blocks then update only question and option tokens:
  question tokens attend to [state ; own question]
  option tokens   attend to [state ; their question ; own option]
Options never see each other, so the answer cannot depend on option order or on how many
other options share the question, and an option's encoding never competes for a token budget.
The score is read at each option's first token and softmaxed over the question's options.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def _bias(mask: torch.Tensor, dtype) -> torch.Tensor:
    """bool keep-mask -> additive bias (0 / -inf)."""
    return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, float("-inf"))


class FusionBlock(nn.Module):
    """Pre-LN block with zero-initialised residual branches (identity at init)."""

    def __init__(self, d: int, heads: int, ff: int, eps: float = 1e-6):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln_c = nn.LayerNorm(d, eps=eps)   # state context
        self.ln_t = nn.LayerNorm(d, eps=eps)   # question / option targets
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ln_f = nn.LayerNorm(d, eps=eps)
        self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)
        for lin in (self.o, self.f2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def heads(self, x):  # [..., L, d] -> [..., h, L, dh]
        return x.unflatten(-1, (self.h, self.dh)).transpose(-3, -2)

    def merge(self, x):  # [..., h, L, dh] -> [..., L, d]
        return x.transpose(-3, -2).flatten(-2)

    def ffn(self, x):
        return x + self.f2(F.gelu(self.f1(self.ln_f(x))))

    def forward(self, S, s_bias, Q, q_bias, O, o_bias):
        """S [Bs,n,d] (Bs = 1 or B), s_bias [Bs,n]; Q [B,Lq,d], q_bias [B,Lq]; O [B,K,Lo,d], o_bias [B,K,Lo]."""
        B, K, Lo, d = O.shape
        sc = 1.0 / math.sqrt(self.dh)
        s = self.ln_c(S)
        ks, vs = self.heads(self.k(s)), self.heads(self.v(s))            # [Bs,h,n,dh]
        if ks.shape[0] != B:
            ks, vs = ks.expand(B, -1, -1, -1), vs.expand(B, -1, -1, -1)
            s_bias = s_bias.expand(B, -1)
        qt = self.ln_t(Q)
        qq, kq, vq = self.heads(self.q(qt)), self.heads(self.k(qt)), self.heads(self.v(qt))   # [B,h,Lq,dh]
        kc, vc = torch.cat([ks, kq], 2), torch.cat([vs, vq], 2)           # [B,h,C,dh]  C = n + Lq
        c_bias = torch.cat([s_bias, q_bias], 1)                           # [B,C]

        # question tokens: attend to [state ; own question]
        att = torch.softmax(qq @ kc.transpose(-1, -2) * sc + c_bias[:, None, None, :], -1)
        Q2 = Q + self.o(self.merge(att @ vc))

        # option tokens: attend to [state ; question ; own option]
        ot = self.ln_t(O)
        qo, ko, vo = self.heads(self.q(ot)), self.heads(self.k(ot)), self.heads(self.v(ot))   # [B,K,h,Lo,dh]
        s_ctx = torch.einsum("bkhld,bhcd->bkhlc", qo, kc) * sc + c_bias[:, None, None, None, :]
        s_own = (qo @ ko.transpose(-1, -2)) * sc + o_bias[:, :, None, None, :]
        p = torch.softmax(torch.cat([s_ctx, s_own], -1), -1)
        C = kc.shape[2]
        out = torch.einsum("bkhlc,bhcd->bkhld", p[..., :C], vc) + p[..., C:] @ vo
        O2 = O + self.o(self.merge(out))
        return self.ffn(Q2), self.ffn(O2)


def _heads(x, h):  # [..., L, d] -> [..., h, L, dh]
    return x.unflatten(-1, (h, x.shape[-1] // h)).transpose(-3, -2)


def _merge(x):  # [..., h, L, dh] -> [..., L, d]
    return x.transpose(-3, -2).flatten(-2)


def state_update(layer, S, s_b):
    """State rows of one fused layer: plain self-attention over the state (identical to the encoder)."""
    h = layer.h
    sc = 1.0 / math.sqrt(S.shape[-1] // h)
    qs, ks, vs = _heads(layer.q(S), h), _heads(layer.k(S), h), _heads(layer.v(S), h)        # [Bs,h,n,dh]
    a = torch.softmax(qs @ ks.transpose(-1, -2) * sc + s_b[:, None, None, :], -1) @ vs
    S2 = layer.ln1(S + layer.o(_merge(a)))
    return layer.ln2(S2 + layer.f2(F.gelu(layer.f1(S2))))


def structured_layer(layer, S, s_b, Q, q_b, O, o_b):
    """One pretrained post-LN encoder layer applied to [state ; questions ; options] with the
    asymmetric mask: state -> state, question -> [state ; own question],
    option -> [state ; its question ; own option]. The state rows are exactly the original layer,
    so states stay shareable across questions (and cacheable across calls)."""
    Q2, O2 = cross_update(layer, S, s_b, Q, q_b, O, o_b)
    return state_update(layer, S, s_b), Q2, O2


def cross_update(layer, S, s_b, Q, q_b, O, o_b):
    """Question / option rows of one fused layer, reading keys and values from this layer's state input S."""
    h = layer.h
    B, K, Lo, d = O.shape
    sc = 1.0 / math.sqrt(d // h)
    ks, vs = _heads(layer.k(S), h), _heads(layer.v(S), h)                                    # [Bs,h,n,dh]
    if ks.shape[0] != B:
        ks, vs, s_b = ks.expand(B, -1, -1, -1), vs.expand(B, -1, -1, -1), s_b.expand(B, -1)
    qq, kq, vq = _heads(layer.q(Q), h), _heads(layer.k(Q), h), _heads(layer.v(Q), h)        # [B,h,Lq,dh]
    kc, vc = torch.cat([ks, kq], 2), torch.cat([vs, vq], 2)
    c_b = torch.cat([s_b, q_b], 1)
    a = torch.softmax(qq @ kc.transpose(-1, -2) * sc + c_b[:, None, None, :], -1) @ vc
    Q2 = layer.ln1(Q + layer.o(_merge(a)))
    Q2 = layer.ln2(Q2 + layer.f2(F.gelu(layer.f1(Q2))))
    qo, ko, vo = _heads(layer.q(O), h), _heads(layer.k(O), h), _heads(layer.v(O), h)        # [B,K,h,Lo,dh]
    s_ctx = torch.einsum("bkhld,bhcd->bkhlc", qo, kc) * sc + c_b[:, None, None, None, :]
    s_own = (qo @ ko.transpose(-1, -2)) * sc + o_b[:, :, None, None, :]
    p = torch.softmax(torch.cat([s_ctx, s_own], -1), -1)
    C = kc.shape[2]
    out = torch.einsum("bkhlc,bhcd->bkhld", p[..., :C], vc) + p[..., C:] @ vo
    O2 = layer.ln1(O + layer.o(_merge(out)))
    O2 = layer.ln2(O2 + layer.f2(F.gelu(layer.f1(O2))))
    return Q2, O2


class DeepFusionHead(nn.Module):
    """Top encoder layers (pretrained copies, trainable) run jointly over state, question and option
    tokens with the asymmetric mask (see structured_layer) -- a cross-encoder in the top layers that
    still encodes the state once for every question. Inputs are the encoder's `split_layer` outputs."""

    def __init__(self, enc_layers_top, d: int, pool: str = "cls", prior_init: float = 10.0):
        super().__init__()
        import copy
        self.layers = nn.ModuleList(copy.deepcopy(l) for l in enc_layers_top)
        self.pool = pool
        self.type_emb = nn.Embedding(3, d)
        nn.init.zeros_(self.type_emb.weight)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.prior = nn.Parameter(torch.tensor([prior_init, prior_init, 0.0]))
        self.register_buffer("temperature", torch.ones(3))

    def state_vec(self, S, s_mask):
        if self.pool == "cls":
            v = S[:, 0]
        else:
            m = s_mask.unsqueeze(-1).to(S.dtype)
            v = (S * m).sum(1) / m.sum(1).clamp_min(1.0)
        return F.normalize(v, dim=-1)

    def state_stream(self, S, s_mask):
        """Run the state through the fused layers on its own (it never sees questions):
        -> (per-layer state inputs, pooled final state vector). Computed once per call."""
        s_b = _bias(s_mask, S.dtype)
        ins = []
        for layer in self.layers:
            ins.append(S)
            S = state_update(layer, S, s_b)
        return ins, self.state_vec(S, s_mask)

    def forward(self, S, s_mask, Q, q_mask, O, o_mask, opt_valid, qtype, s_vec, o_vec, return_state_vec=False,
                stream=None):
        """Same signature as DecisionHead; S/Q/O are split-layer token states, `s_vec` is ignored
        (the state vector is pooled from the final fused state), `o_vec` are frozen option embeddings.
        `stream` = state_stream(S, s_mask) when already computed for this state."""
        dt = S.dtype
        s_b, q_b, o_b = _bias(s_mask, dt), _bias(q_mask, dt), _bias(o_mask, dt)
        ins, sv = stream if stream is not None else self.state_stream(S, s_mask)
        temb = self.type_emb(qtype)
        Q = Q + temb[:, None, :]
        O = O + temb[:, None, None, :]
        for layer, S_in in zip(self.layers, ins):
            Q, O = cross_update(layer, S_in, s_b, Q, q_b, O, o_b)
        logit = self.scorer(O[:, :, 0]).squeeze(-1)
        cos = torch.einsum("bkd,bd->bk", o_vec, sv.expand(o_vec.shape[0], -1))
        logit = logit + self.prior[qtype][:, None] * cos
        logit = logit.masked_fill(~opt_valid, float("-inf"))
        return (logit, sv) if return_state_vec else logit


class StateAdapter(nn.Module):
    """Trainable copies of the encoder's top layers, run on the state only (split at `split_layer`).
    Initialised from the pretrained layers, so at step 0 it reproduces the frozen encoder exactly."""

    def __init__(self, enc_layers, eps: float):
        super().__init__()
        import copy
        self.layers = nn.ModuleList(copy.deepcopy(l) for l in enc_layers)

    def forward(self, S, s_mask):
        bias = _bias(s_mask, S.dtype)[:, None, None, :]
        for l in self.layers:
            S = l(S, bias)
        return S


class DecisionHead(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, n_blocks: int, prior_init: float = 10.0):
        super().__init__()
        self.blocks = nn.ModuleList(FusionBlock(d, heads, ff) for _ in range(n_blocks))
        self.type_emb = nn.Embedding(3, d)
        nn.init.zeros_(self.type_emb.weight)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # per-type weight on the frozen-embedding cosine prior (yes/no options carry no meaning alone)
        self.prior = nn.Parameter(torch.tensor([prior_init, prior_init, 0.0]))
        self.register_buffer("temperature", torch.ones(3))

    def init_from_encoder(self, enc_layers):
        """Start attention/FFN-in projections from the top encoder layers (outputs stay zero-init)."""
        with torch.no_grad():
            for blk, lay in zip(self.blocks, list(enc_layers)[-len(self.blocks):]):
                for a, b in ((blk.q, lay.q), (blk.k, lay.k), (blk.v, lay.v), (blk.f1, lay.f1)):
                    a.weight.copy_(b.weight)
                    a.bias.copy_(b.bias)

    def forward(self, S, s_mask, Q, q_mask, O, o_mask, opt_valid, qtype, s_vec, o_vec):
        """
        S [Bs,n,d] s_mask [Bs,n]      frozen state token states (Bs = 1 shares one state across B questions)
        Q [B,Lq,d] q_mask [B,Lq]      frozen question token states
        O [B,K,Lo,d] o_mask [B,K,Lo]  frozen option token states
        opt_valid [B,K] bool, qtype [B] long
        s_vec [Bs,d], o_vec [B,K,d]   pooled, L2-normalised frozen embeddings (for the cosine prior)
        -> logits [B,K] (uncalibrated; -inf on padded options)
        """
        dt = S.dtype
        s_b, q_b, o_b = _bias(s_mask, dt), _bias(q_mask, dt), _bias(o_mask, dt)
        # padded option rows would have no live key in their own block; they still see the state, so no NaN
        temb = self.type_emb(qtype)
        Q = Q + temb[:, None, :]
        O = O + temb[:, None, None, :]
        for blk in self.blocks:
            Q, O = blk(S, s_b, Q, q_b, O, o_b)
        logit = self.scorer(O[:, :, 0]).squeeze(-1)
        cos = torch.einsum("bkd,bd->bk", o_vec, s_vec.expand(o_vec.shape[0], -1))
        logit = logit + self.prior[qtype][:, None] * cos
        return logit.masked_fill(~opt_valid, float("-inf"))
