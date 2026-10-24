"""GRO (Zhang et al., WSDM'24): the recommender is fine-tuned to keep its utility while
misleading a student (surrogate) trained on its top-k lists.

Per step, (1) a student SASRec is distilled on the defended model's top-k distribution, and
(2) the defended model minimizes next-item cross-entropy minus lambda times the student's
distillation loss on its own top-k outputs, which moves its lists away from what a list-trained
student currently predicts. Top-k lists are relaxed with a temperature softmax restricted to the
top-k. The defended model serves ranked lists without scores.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from . import sasrec

DEV = sasrec.DEV


def _topk_soft(logits, k=10, tau=1.0):
    top = logits.topk(k, dim=1)
    p = torch.zeros_like(logits)
    p.scatter_(1, top.indices, F.softmax(top.values / tau, 1))
    return p


def finetune(victim: sasrec.SASRec, seqs, lam: float, epochs: int = 5, bs: int = 256, lr: float = 5e-4,
             k: int = 10, seed: int = 0) -> sasrec.SASRec:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = copy.deepcopy(victim).train()
    student = sasrec.SASRec(copy.deepcopy(victim.cfg)).to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt_s = torch.optim.Adam(student.parameters(), lr=1e-3)
    seqs = [s for s in seqs if len(s) >= 2]
    for ep in range(epochs):
        for inp, tgt in sasrec._batches(seqs, model.cfg.maxlen, bs, rng):
            inp, tgt = inp.to(DEV), tgt.to(DEV)
            # (1) student distils the defended model's top-k lists (last position = served query)
            with torch.no_grad():
                p_t = _topk_soft(model.logits_all(inp), k)
            ls = -(p_t * F.log_softmax(student.logits_all(inp), 1)).sum(1).mean()
            opt_s.zero_grad()
            ls.backward()
            opt_s.step()
            # (2) defended model: utility minus lambda * student's loss on its outputs
            h = model.hidden(inp)
            m = tgt > 0
            util = F.cross_entropy(h[m] @ model.item.weight[1:].T, tgt[m] - 1)
            p = _topk_soft(h[:, -1] @ model.item.weight[1:].T, k)
            with torch.no_grad():
                lq = F.log_softmax(student.logits_all(inp), 1)
            adv = -(p * lq).sum(1).mean()
            loss = util - lam * adv
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()
