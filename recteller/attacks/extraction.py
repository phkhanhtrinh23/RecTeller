"""Model extraction (App. C.3 "Model extraction"), following Yue et al. (2021) and LLM4MEA.

The attacker registers fresh accounts (no stored history), grows ``n_queries`` synthetic
sequences autoregressively to ``length`` items: at each step it issues a typed SEARCH ("more
like this" on the last item, which stores C(q) in the session) and a PERSONALISE with the
current sequence as session context, then appends an item sampled from the returned list.
Requests are spread evenly over ``days`` windows and over ``n_accounts`` accounts. The
gatekeeper's counters decide what each request gets (refused beyond P, denied beyond B,
public slots beyond n_max). A surrogate SASRec is trained on the collected (sequence, list)
pairs with a listwise ranking loss (RecTeller's levels as weights, scores where released) and
scored by Agr@10 against the victim on held-out real sessions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .. import sasrec
from ..gatekeeper import Request
from ..intents import Personalise, Search
from ..modes import build

DEV = sasrec.DEV


@dataclass
class Pair:
    context: list[int]
    items: list[int]
    q: np.ndarray            # target weights over items (sum 1)
    personalised: bool


READING = "levels"   # "levels": RecTeller levels as weights, "list": every response read as a ranked list


def targets(resp, kind: str) -> np.ndarray:
    """How the attacker reads a response as soft ranking targets."""
    n = len(resp.items)
    if READING == "list" and resp.scores is None:
        w = 1.0 / np.log2(np.arange(n) + 2.0)
        return (w / w.sum()).astype(np.float32)
    if resp.scores is not None:                       # Open API (exact) / JSON top-K (quantised)
        s = np.asarray(resp.scores, dtype=np.float64)
        w = np.exp(s - s.max())
    elif resp.n_pers and resp.kind in ("noisy", "exact_kp"):
        lv = np.asarray(resp.levels, dtype=np.float64)
        w = np.where(lv > 0, 1.0 + lv, 0.25)          # levels as weights, public fill weakly
    else:                                             # ranked list without scores / public reply
        w = 1.0 / np.log2(np.arange(n) + 2.0)
    return (w / w.sum()).astype(np.float32)


def pick_next(resp, rng) -> int:
    if READING != "list" and resp.n_pers and resp.kind in ("noisy", "exact_kp"):
        return int(resp.items[rng.integers(resp.n_pers)])
    top = resp.items[:10]
    return int(top[rng.integers(len(top))])


def collect(ctx, mode: str, n_accounts: int, days: int, seed: int, n_queries: int = 5000, length: int = 20,
            proto=None, cap30=None, record_contexts: bool = False):
    """Run the attack through the access mode and return (pairs, stats)."""
    rng = np.random.default_rng(seed)
    gk = build(mode, ctx, seed=seed + 1000, proto=proto, cap30=cap30)
    cat = ctx.cat
    seqs = [[int(rng.integers(1, cat.n + 1))] for _ in range(n_queries)]
    toks = [gk.auth.issue(f"att{a}", covers=[]) for a in range(n_accounts)]
    total = n_queries * length
    horizon = days * 24.0
    pairs: list[Pair] = []
    r_idx = 0
    for step in range(length):
        sreq, preq = [], []
        for j, s in enumerate(seqs):
            now = horizon * (r_idx / total)
            r_idx += 1
            tok = toks[j % n_accounts]
            sess = f"q{j}"
            sreq.append(Request(tok, sess, now, intent=Search(text=cat.text(s[-1]))))
            preq.append(Request(tok, sess, now, intent=Personalise((), tuple(s[-50:]))))
        # each sequence's Search precedes its Personalise, interleaved to respect counter order
        reqs = [r for pair in zip(sreq, preq) for r in pair]
        out = gk.serve_batch(reqs)
        for j, s in enumerate(seqs):
            r = out[2 * j + 1]
            if r.status != "OK" or not r.items:
                s.append(int(rng.integers(1, cat.n + 1)))      # denied: no information
                continue
            pairs.append(Pair(list(s[-50:]), list(r.items), targets(r, r.kind), bool(r.n_pers and r.personalised)))
            s.append(pick_next(r, rng))
    st = dict(gk.stats)
    st["pairs"] = len(pairs)
    st["personalised_pairs"] = sum(p.personalised for p in pairs)
    return pairs, st


def train_surrogate(pairs: list[Pair], n_items: int, seed: int = 0, visits: int = 400_000, bs: int = 256,
                    n_neg: int = 100, lr: float = 1e-3, maxlen: int = 50) -> sasrec.SASRec:
    """Listwise distillation: CE between the response targets q and the surrogate's softmax over
    the returned items plus random negatives."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = sasrec.SASRec(sasrec.SASRecConfig(n_items, maxlen=maxlen, dropout=0.2)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if not pairs:
        return model.eval()
    m = max(len(p.items) for p in pairs)
    ctx = sasrec.pad_seqs([p.context for p in pairs], maxlen)
    items = np.zeros((len(pairs), m), dtype=np.int64)
    q = np.zeros((len(pairs), m), dtype=np.float32)
    for k, p in enumerate(pairs):
        items[k, :len(p.items)] = p.items
        q[k, :len(p.items)] = p.q
    items, q = torch.from_numpy(items), torch.from_numpy(q)
    steps = max(200, visits // bs)
    model.train()
    for st in range(steps):
        idx = torch.from_numpy(rng.integers(0, len(pairs), size=bs))
        H = ctx[idx].to(DEV)
        it = items[idx].to(DEV)
        neg = torch.randint(1, n_items + 1, (bs, n_neg), device=DEV)
        allc = torch.cat([it, neg], 1)
        tgt = torch.cat([q[idx].to(DEV), torch.zeros(bs, n_neg, device=DEV)], 1)
        lg = model.logits_cands(H, allc)
        lg = lg.masked_fill(allc == 0, -1e4)
        loss = -(tgt * F.log_softmax(lg, 1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def agreement(victim: sasrec.Scorer, surrogate: sasrec.SASRec, sessions: list[list[int]], k: int = 10) -> np.ndarray:
    """Agr@k per held-out session: top-k overlap of surrogate and victim over the catalogue."""
    a = victim.topk_all(sessions, k)
    b = sasrec.Scorer(surrogate).topk_all(sessions, k)
    return np.array([len(set(x) & set(y)) / k for x, y in zip(a, b)])


def run(ctx, mode, victim, sessions, n_accounts=50, days=7, seed=0, proto=None, cap30=None, n_queries=5000,
        length=20, visits=400_000):
    pairs, st = collect(ctx, mode, n_accounts, days, seed, n_queries, length, proto, cap30)
    sur = train_surrogate(pairs, ctx.cat.n, seed=seed, visits=visits)
    agr = agreement(victim, sur, sessions)
    st["agr10"] = float(agr.mean())
    return st, agr, pairs
