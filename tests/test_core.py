"""Core invariants: encoder parity with transformers, head shapes, option-order invariance, determinism.

  python -m pytest tests -q          (or: python tests\\test_core.py)
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nirnay.model import DecisionHead  # noqa: E402
from nirnay.render import render_question, render_state  # noqa: E402


def test_encoder_matches_transformers():
    from transformers import AutoModel, AutoTokenizer
    from nirnay.encoder import load_hf_encoder
    name = os.environ.get("NIRNAY_BACKBONE", "intfloat/multilingual-e5-small")
    enc = load_hf_encoder(name, "mean", "query: ")
    tok = AutoTokenizer.from_pretrained(name)
    ref = AutoModel.from_pretrained(name).eval()
    b = tok(["query: hello world", "query: a much longer sentence about billing and refunds"], padding=True,
            return_tensors="pt")
    with torch.no_grad():
        h_ref = ref(**b).last_hidden_state
        h = enc(b["input_ids"], b["attention_mask"].bool())
    m = b["attention_mask"].bool()
    assert torch.allclose(h[m], h_ref[m], atol=2e-4), float((h[m] - h_ref[m]).abs().max())


def _rand_inputs(B=3, n=7, Lq=5, K=4, Lo=3, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(1, n, d, generator=g)
    sm = torch.ones(1, n, dtype=torch.bool)
    Q = torch.randn(B, Lq, d, generator=g)
    qm = torch.ones(B, Lq, dtype=torch.bool)
    O = torch.randn(B, K, Lo, d, generator=g)
    om = torch.ones(B, K, Lo, dtype=torch.bool)
    om[:, :, -1] = False
    ok = torch.ones(B, K, dtype=torch.bool)
    qt = torch.tensor([0, 1, 2][:B])
    sv = torch.nn.functional.normalize(torch.randn(1, d, generator=g), dim=-1)
    ov = torch.nn.functional.normalize(torch.randn(B, K, d, generator=g), dim=-1)
    return S, sm, Q, qm, O, om, ok, qt, sv, ov


def _head(d=32):
    torch.manual_seed(0)
    h = DecisionHead(d, 4, 64, 2)
    with torch.no_grad():  # make the zero-init branches live so the test is meaningful
        for p in h.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return h.eval()


def test_head_shapes_and_option_independence():
    h = _head()
    x = _rand_inputs()
    with torch.no_grad():
        z = h(*x)
    assert z.shape == (3, 4) and torch.isfinite(z).all()
    # permuting options permutes logits and nothing else
    perm = torch.tensor([2, 0, 3, 1])
    S, sm, Q, qm, O, om, ok, qt, sv, ov = x
    with torch.no_grad():
        zp = h(S, sm, Q, qm, O[:, perm], om[:, perm], ok[:, perm], qt, sv, ov[:, perm])
    assert torch.allclose(zp, z[:, perm], atol=1e-5)
    # adding an extra option leaves the existing options' logits unchanged
    O2 = torch.cat([O, torch.randn(3, 1, O.shape[2], O.shape[3])], 1)
    om2 = torch.cat([om, torch.ones(3, 1, O.shape[2], dtype=torch.bool)], 1)
    ok2 = torch.cat([ok, torch.ones(3, 1, dtype=torch.bool)], 1)
    ov2 = torch.cat([ov, torch.nn.functional.normalize(torch.randn(3, 1, ov.shape[2]), dim=-1)], 1)
    with torch.no_grad():
        z2 = h(S, sm, Q, qm, O2, om2, ok2, qt, sv, ov2)
    assert torch.allclose(z2[:, :4], z, atol=1e-5)


def test_render():
    assert render_state({"ticket": {"subject": "x", "messages": [{"text": "hi"}]}}) == \
        "ticket.subject: x\nticket.messages[0].text: hi"
    nq, qt, ot = render_question("t", {"type": "choice", "instructions": "Which team?",
                                       "criteria": {"billing": "payments", "sales_ops": None}})
    assert qt == "choice question: Which team?" and ot == ["billing: payments", "sales ops"]
    nq, qt, ot = render_question("u", {"type": "noul", "instructions": "Is it urgent?"})
    assert nq["keys"] == ["false", "true"] and ot[1].startswith("yes")
    nq, qt, ot = render_question("s", {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]})
    assert ot == ["level 0 of 2: low", "level 1 of 2: mid", "level 2 of 2: high"]


def test_router_language_detection():
    from nirnay.router import analyse
    assert analyse("My card was charged twice and I want a refund")["english"]
    assert analyse({"ticket": {"subject": "Refund", "body": "Please send the invoice for the order"}} and
                   "ticket.subject: Refund\nticket.body: Please send the invoice for the order")["english"]
    assert not analyse("എന്റെ കാർഡിൽ രണ്ട് തവണ പണം ഈടാക്കി")["english"]          # Malayalam
    assert not analyse("मेरा भुगतान तीन दिनों से विफल हो रहा है")["english"]        # Hindi
    assert not analyse("Meine Zahlung ist seit drei Tagen fehlgeschlagen, bitte helfen Sie")["english"]  # German


if __name__ == "__main__":
    test_router_language_detection()
    test_render()
    test_head_shapes_and_option_independence()
    test_encoder_matches_transformers()
    print("ok")
