"""Self-contained BERT / XLM-R style encoder (post-LayerNorm), loaded from Hugging Face weights.

Owning the forward pass (instead of calling transformers' modeling code) keeps the numerics
fixed across transformers releases, which the determinism guarantee depends on, and lets the
fusion head reuse the exact same attention kernel.
"""
import json
import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderConfig:
    def __init__(self, vocab_size: int, hidden: int, layers: int, heads: int, ff: int, max_pos: int,
                 type_vocab: int = 2, ln_eps: float = 1e-12, pos_offset: int = 0, pad_id: int = 0,
                 pooling: str = "mean", prefix: str = "", model_type: str = "bert"):
        self.vocab_size, self.hidden, self.layers, self.heads, self.ff = vocab_size, hidden, layers, heads, ff
        self.max_pos, self.type_vocab, self.ln_eps = max_pos, type_vocab, ln_eps
        self.pos_offset, self.pad_id, self.pooling, self.prefix = pos_offset, pad_id, pooling, prefix
        self.model_type = model_type

    @property
    def max_tokens(self) -> int:
        return self.max_pos - self.pos_offset

    def to_dict(self) -> Dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: Dict) -> "EncoderConfig":
        return cls(**d)

    @classmethod
    def from_hf(cls, hf_cfg: Dict, pooling: str, prefix: str) -> "EncoderConfig":
        mt = hf_cfg["model_type"]
        if mt not in ("bert", "xlm-roberta", "roberta"):
            raise ValueError("unsupported backbone type %r (bert / xlm-roberta only)" % mt)
        pad = int(hf_cfg.get("pad_token_id", 0) or 0)
        # RoBERTa/XLM-R number real tokens from padding_idx + 1
        offset = pad + 1 if mt in ("xlm-roberta", "roberta") else 0
        if hf_cfg.get("hidden_act", "gelu") not in ("gelu",):
            raise ValueError("unsupported activation %r" % hf_cfg.get("hidden_act"))
        return cls(vocab_size=hf_cfg["vocab_size"], hidden=hf_cfg["hidden_size"], layers=hf_cfg["num_hidden_layers"],
                   heads=hf_cfg["num_attention_heads"], ff=hf_cfg["intermediate_size"],
                   max_pos=hf_cfg["max_position_embeddings"], type_vocab=hf_cfg.get("type_vocab_size", 2),
                   ln_eps=hf_cfg.get("layer_norm_eps", 1e-12), pos_offset=offset, pad_id=pad,
                   pooling=pooling, prefix=prefix, model_type=mt)


def attention(q, k, v, bias: Optional[torch.Tensor]):
    """q [B,h,Lq,dh], k/v [B,h,Lk,dh], bias additive [B,1|h,Lq,Lk] or None."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


class PostLNLayer(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, eps: float):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ln1 = nn.LayerNorm(d, eps=eps)
        self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.ln2 = nn.LayerNorm(d, eps=eps)

    def split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.h, D // self.h).transpose(1, 2)

    def forward(self, x, bias):
        B, L, D = x.shape
        a = attention(self.split(self.q(x)), self.split(self.k(x)), self.split(self.v(x)), bias)
        x = self.ln1(x + self.o(a.transpose(1, 2).reshape(B, L, D)))
        return self.ln2(x + self.f2(F.gelu(self.f1(x))))


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.word = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.max_pos, d)
        self.typ = nn.Embedding(cfg.type_vocab, d)
        self.ln = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.layers = nn.ModuleList(PostLNLayer(d, cfg.heads, cfg.ff, cfg.ln_eps) for _ in range(cfg.layers))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, keep_layer: Optional[int] = None,
                stop_layer: Optional[int] = None):
        """ids/mask [B, L] (mask bool, right padded) -> hidden [B, L, d]
        (and the output of layer `keep_layer` as well, when given; `stop_layer` returns after that layer)."""
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0) + self.cfg.pos_offset
        x = self.ln(self.word(ids) + self.pos(pos) + self.typ.weight[0])
        bias = torch.zeros(B, 1, 1, L, dtype=x.dtype, device=x.device).masked_fill(~mask[:, None, None, :], float("-inf"))
        mid = None
        for i, layer in enumerate(self.layers):
            x = layer(x, bias)
            if keep_layer is not None and i + 1 == keep_layer:
                mid = x
            if stop_layer is not None and i + 1 == stop_layer:
                return x
        return x if keep_layer is None else (x, mid)

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "cls":
            v = h[:, 0]
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return F.normalize(v, dim=-1)


# HF parameter name -> ours (identical for BERT and XLM-R checkpoints)
def _hf_key_map(n_layers: int) -> Dict[str, str]:
    m = {"embeddings.word_embeddings.weight": "word.weight",
         "embeddings.position_embeddings.weight": "pos.weight",
         "embeddings.token_type_embeddings.weight": "typ.weight",
         "embeddings.LayerNorm.weight": "ln.weight", "embeddings.LayerNorm.bias": "ln.bias"}
    for i in range(n_layers):
        p, q = "encoder.layer.%d." % i, "layers.%d." % i
        for a, b in (("attention.self.query", "q"), ("attention.self.key", "k"), ("attention.self.value", "v"),
                     ("attention.output.dense", "o"), ("attention.output.LayerNorm", "ln1"),
                     ("intermediate.dense", "f1"), ("output.dense", "f2"), ("output.LayerNorm", "ln2")):
            for w in ("weight", "bias"):
                m[p + a + "." + w] = q + b + "." + w
    return m


def load_hf_encoder(repo_or_dir: str, pooling: str, prefix: str) -> Encoder:
    """Build an Encoder from a Hub repo id or local directory holding config.json + safetensors/bin."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = repo_or_dir if os.path.isdir(repo_or_dir) else snapshot_download(
        repo_or_dir, allow_patterns=["config.json", "*.safetensors", "tokenizer*", "sentencepiece*",
                                     "special_tokens_map.json", "*.model", "vocab.txt"])
    hf_cfg = json.load(open(os.path.join(path, "config.json")))
    cfg = EncoderConfig.from_hf(hf_cfg, pooling, prefix)
    enc = Encoder(cfg)
    files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    if files:
        sd = {}
        for f in files:
            sd.update(load_file(os.path.join(path, f)))
    else:
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    km = _hf_key_map(cfg.layers)
    new = {}
    for k, v in sd.items():
        k2 = k
        for pre in ("bert.", "roberta.", "model."):
            if k2.startswith(pre):
                k2 = k2[len(pre):]
        if k2 in km:
            new[km[k2]] = v.float()
    missing = [k for k in enc.state_dict() if k not in new]
    if missing:
        raise RuntimeError("encoder weights missing from %s: %s" % (repo_or_dir, missing[:5]))
    enc.load_state_dict(new, strict=True)
    enc.source_dir = path
    return enc.eval()
