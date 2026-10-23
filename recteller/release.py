"""Budgeted release (Sec. 4.2, Eqs. select/weights, Prop. 1).

Given the clipped score vector s on a fixed candidate set C(q) (public retrieval, H1):

* ``noisy``      : one-shot Gumbel top-k' with scale b_sel = 2 k_p Delta / eps_sel, then Laplace
                   noise of scale b_w = k_p Delta / eps_w on the selected scores, quantised to
                   levels ceil(L * softmax(w_hat)). Default RecTeller release.
* ``exact_kp``   : the eps_resp = infinity point: exact top-k_p, levels from exact scores, same
                   public fill and slots. Never aliased with the control below.
* ``topK_json``  : Capped / JSON top-K control: exact top-K with quantised scores.
* ``random_replace(p)``: Capped top-K with a share p of the list replaced by random candidates.
* ``topK_list``  : exact ranked top-K without scores (MCP tools, GRO, Firewall).
* ``all_scores`` : exact scores of every candidate (Open API).

Public fill F: items of C(q) \\ S by public relevance r_i(q), level 0, no privacy cost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Release:
    items: list[int]                 # ordered: S then F (indices into the catalogue)
    levels: list[int]                # weight levels (0 for public fill)
    n_pers: int                      # |S|
    scores: list[float] | None = None  # only for topK_json (quantised) / all_scores (exact)
    kind: str = "noisy"


def clip(s: np.ndarray, c: float) -> np.ndarray:
    return np.clip(s, -c, c)


def levels_from(w: np.ndarray, L: int) -> list[int]:
    w = np.asarray(w, dtype=np.float64)
    p = np.exp(w - w.max())
    p /= p.sum()
    return [int(max(1, math.ceil(L * pk - 1e-12))) for pk in p]


def gumbel_topk(s: np.ndarray, k: int, b: float, rng, gumbel: np.ndarray | None = None) -> np.ndarray:
    """Eq. (select): ordered top-k of s_i + G_i, G_i ~ Gumbel(0, b). Only indices leave."""
    g = rng.gumbel(0.0, b, size=len(s)) if gumbel is None else np.asarray(gumbel)
    return np.argsort(-(s + g), kind="stable")[:k]


def public_fill(cands: list[int], public_order: list[int], chosen: list[int], n: int) -> list[int]:
    ch = set(chosen)
    return [i for i in public_order if i not in ch][:n]


def release(scores_raw: np.ndarray, cands: list[int], public_order: list[int], proto, rng, kind: str = "noisy",
            kp: int | None = None, p_replace: float = 0.0, use_levels: bool = True,
            gumbel: np.ndarray | None = None, laplace: np.ndarray | None = None) -> Release:
    """Release for one personalised request. ``scores_raw`` are f_theta logits aligned with ``cands``.
    ``public_order`` is C(q) sorted by public relevance."""
    kp = proto.kp if kp is None else kp
    K = proto.K
    Kp = min(K, len(cands))
    if kind == "all_scores":
        order = np.argsort(-scores_raw, kind="stable")
        return Release([cands[j] for j in order], [0] * len(cands), len(cands),
                       [float(scores_raw[j]) for j in order], kind)
    s = clip(np.asarray(scores_raw, dtype=np.float64), proto.clip)
    if kind in ("topK_json", "topK_list", "random_replace"):
        order = list(np.argsort(-s, kind="stable")[:Kp])
        if kind == "random_replace" and p_replace > 0:
            n_rep = int(round(p_replace * Kp))
            pos = rng.choice(Kp, size=n_rep, replace=False)
            rest = [j for j in range(len(cands)) if j not in order]
            new = list(rng.choice(rest, size=min(n_rep, len(rest)), replace=False))
            for p_, j in zip(pos, new):
                order[p_] = j
        items = [cands[j] for j in order]
        sc = [round(float(s[j]), 1) for j in order] if kind != "topK_list" else None
        return Release(items, [0] * len(items), len(items), sc, kind)
    k1 = min(kp, len(cands))
    if k1 == 0:
        return Release(public_fill(cands, public_order, [], Kp), [0] * Kp, 0, None, kind)
    if kind == "exact_kp" or not math.isfinite(proto.eps_sel):
        sel = np.argsort(-s, kind="stable")[:k1]
        w = s[sel]
    elif kind == "noisy":
        b_sel = 2 * kp * proto.Delta / proto.eps_sel     # uses k_p even when k' < k_p
        sel = gumbel_topk(s, k1, b_sel, rng, gumbel)
        if use_levels and math.isfinite(proto.eps_w) and proto.eps_w > 0:
            b_w = kp * proto.Delta / proto.eps_w
            eta = rng.laplace(0.0, b_w, size=k1) if laplace is None else np.asarray(laplace)
            w = s[sel] + eta                              # noise precedes quantisation
        elif use_levels:
            w = s[sel]                                    # eps_w = inf
        else:
            w = None
    else:
        raise ValueError(kind)
    S = [cands[j] for j in sel]
    lv = levels_from(w, proto.L) if (w is not None and use_levels) else [1] * k1
    F = public_fill(cands, public_order, S, Kp - k1)
    return Release(S + F, lv + [0] * len(F), k1, None, kind)
