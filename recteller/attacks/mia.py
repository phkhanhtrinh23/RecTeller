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
from ..intents import Personalise, Search
from ..modes import Context, build


def _cands(ctx: Context, hist: list[int], size: int) -> tuple[int, ...]:
    """The held-back last item, the other recent history items and public neighbours."""
    own = list(dict.fromkeys([hist[-1]] + hist[-20:-1]))
    nb = ctx.retrieval.neighbours()[hist[-2] if len(hist) > 1 else hist[-1]]
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
        # context = history without its last item, which the attacker holds back as a probe
        reqs.append(Request(tok, f"t{j}", now, intent=Personalise((), tuple(h[:-1][-50:]))))
    out = gk.serve_batch(reqs)
    return [out[2 * j + 1] for j in range(n)]


NF = 12


def features(ctx: Context, hist: list[int], resp) -> np.ndarray:
    E = ctx.pe.item
    probe = hist[-1]
    hist = hist[:-1] if len(hist) > 1 else hist
    hs = set(hist)
    hc = E[hist[-50:]].mean(0)
    if resp.status != "OK" or not resp.items:
        return np.zeros(NF + E.shape[1], dtype=np.float32)
    items = resp.items[:10] if resp.scores is None or resp.kind != "all_scores" else resp.items[:10]
    n = len(items)
    if resp.kind in ("noisy", "exact_kp"):
        w = np.where(np.asarray(resp.levels[:n]) > 0, 1.0 + np.asarray(resp.levels[:n]), 0.0)
        if w.sum() == 0:
            w = np.ones(n)
    else:
        w = 1.0 / np.log2(np.arange(n) + 2.0)
    w = w / w.sum()
    inh = np.array([i in hs for i in items], dtype=np.float64)
    rc = (E[items] * w[:, None]).sum(0)
    cos = float(rc @ hc / (np.linalg.norm(rc) * np.linalg.norm(hc) + 1e-9))
    f = [float((w * inh).sum()), float(inh[:3].mean()), cos, float((E[items] @ hc).mean()), float(inh.mean())]
    if resp.scores is not None:                          # released scores (Open API / JSON top-K)
        s = np.asarray(resp.scores)
        allit = resp.items
        m = np.array([i in hs for i in allit])
        f += [float(s[m].mean() - s[~m].mean()) if m.any() and (~m).any() else 0.0, float(s.max()), float(s.std())]
    else:
        f += [0.0, 0.0, 0.0]
    # probe (held-back next item): returned? position, level, released score relative to the others
    if probe in resp.items:
        k = resp.items.index(probe)
        lvl = resp.levels[k] if resp.levels else 0
        rel = 0.0
        if resp.scores is not None:
            s = np.asarray(resp.scores)
            rel = float((s[k] - s.mean()) / (s.std() + 1e-6))
        f += [1.0, 1.0 / (k + 1), float(lvl), rel]
    else:
        f += [0.0, 0.0, 0.0, 0.0]
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
    # hand features + PCA of Zhang et al.'s difference vector
    Xs2 = Xs
    clf.fit(Xs2, ys)
    p = clf.predict_proba(Xt)[:, 1]
    auc = roc_auc_score(yt, p)
    # Hanley-McNeil standard error
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    n1, n0 = yt.sum(), len(yt) - yt.sum()
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    return {"auc": float(auc), "ci95": float(1.96 * se), "shadow_train_auc": float(roc_auc_score(ys, clf.predict_proba(Xs2)[:, 1]))}
