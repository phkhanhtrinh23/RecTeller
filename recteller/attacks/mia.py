"""Membership inference (App. C.3), after Zhang et al. (2021).

The attacker knows each target's history, submits it once as session context (after a SEARCH
that stores a candidate set: the history's last items plus their public neighbours), and
classifies membership from the distance between the history and the returned items in the
public embedding space. The classifier is trained on a shadow SASRec over disjoint users,
queried through a simulation of the same access mode (the mechanism is public).
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..gatekeeper import Request
from ..intents import Personalize, Search
from ..modes import Context, build


def _cands(ctx: Context, hist: list[int], size: int) -> tuple[int, ...]:
    """Candidate set of the attacker's SEARCH: recent history items plus public neighbours."""
    own = list(dict.fromkeys(hist[-20:]))
    nb = ctx.retrieval.neighbours()[hist[-1]]
    rest = [int(i) for i in nb if i not in set(own)]
    return tuple((own + rest)[:size])


def query(ctx: Context, mode: str, histories: list[list[int]], seed: int, n_accounts: int = 50, days: int = 7):
    """One session-context query per target, spread over accounts and windows (below the cap)."""
    gk = build(mode, ctx, seed=seed)
    toks = [gk.auth.issue(f"mia{a}", covers=[]) for a in range(n_accounts)]
    reqs = []
    n = len(histories)
    for j, h in enumerate(histories):
        now = days * 24.0 * j / n
        tok = toks[j % n_accounts]
        reqs.append(Request(tok, f"t{j}", now, intent=Search(text="", items=_cands(ctx, h, ctx.proto.search_size))))
        reqs.append(Request(tok, f"t{j}", now, intent=Personalize((), tuple(h[-50:]))))
    out = gk.serve_batch(reqs)
    return [out[2 * j + 1] for j in range(n)]


NF = 5


def features(ctx: Context, hist: list[int], resp) -> np.ndarray:
    """Distance between the history and the full returned item list in the public embedding space.

    The released levels (including the level-0 fill marker) and scores are ignored. Items are
    weighted by their position in the returned list only."""
    E = ctx.pe.item
    hs = set(hist)
    hc = E[hist[-50:]].mean(0)
    if resp.status != "OK" or not resp.items:
        return np.zeros(NF + E.shape[1], dtype=np.float32)
    items = resp.items[: ctx.proto.K]
    n = len(items)
    w = 1.0 / np.log2(np.arange(n) + 2.0)
    w = w / w.sum()
    inh = np.array([i in hs for i in items], dtype=np.float64)
    rc = (E[items] * w[:, None]).sum(0)
    cos = float(rc @ hc / (np.linalg.norm(rc) * np.linalg.norm(hc) + 1e-9))
    sims = E[items] @ hc
    f = [cos, float((w * sims).sum()), float(sims.max()), float((w * inh).sum()), float(inh.mean())]
    return np.concatenate([np.array(f, dtype=np.float32), (hc - rc).astype(np.float32)])


def attack(victim_ctx: Context, shadow_ctx: Context, mode: str, targets_in, targets_out, shadow_in, shadow_out,
           seed: int = 0) -> dict:
    """Train on shadow (in/out) responses, evaluate AUC on victim targets."""
    def feats(ctx, hs, s):
        rs = query(ctx, mode, hs, seed=s)
        return np.stack([features(ctx, h, r) for h, r in zip(hs, rs)])

    Xs = feats(shadow_ctx, shadow_in + shadow_out, seed + 7)
    ys = np.r_[np.ones(len(shadow_in)), np.zeros(len(shadow_out))]
    Xt = feats(victim_ctx, targets_in + targets_out, seed + 11)
    yt = np.r_[np.ones(len(targets_in)), np.zeros(len(targets_out))]
    clf = make_pipeline(StandardScaler(), PCA(n_components=24, random_state=seed),
                        LogisticRegression(C=1.0, max_iter=2000))
    # distance features + PCA of Zhang et al.'s difference vector
    clf.fit(Xs, ys)
    p = clf.predict_proba(Xt)[:, 1]
    auc = roc_auc_score(yt, p)
    # Hanley-McNeil standard error
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    n1, n0 = yt.sum(), len(yt) - yt.sum()
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    return {"auc": float(auc), "ci95": float(1.96 * se), "shadow_train_auc": float(roc_auc_score(ys, clf.predict_proba(Xs)[:, 1]))}
