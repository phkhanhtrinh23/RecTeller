"""Floods (App. C.3): one account sends ``n`` requests within one window over ``sessions``
parallel sessions, and a share ``repeat`` repeats an earlier typed request. We count the platform
calls (retrieval, catalog or scorer) that the access mode forwards."""
from __future__ import annotations

import numpy as np

from ..gatekeeper import Request
from ..intents import Personalize, Search
from ..modes import build


def run(ctx, mode: str, n: int = 1000, sessions: int = 5, repeat: float = 0.05, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    gk = build(mode, ctx, seed=seed)
    tok = gk.auth.issue("flooder", covers=[])
    reqs, issued = [], []
    for j in range(n):
        sess = f"s{j % sessions}"
        now = 23.0 * j / n
        if issued and rng.random() < repeat:
            s_, it = issued[rng.integers(len(issued))]
            reqs.append(Request(tok, s_, now, intent=it))
            continue
        if j % 2 == 0:
            it = Search(text=ctx.cat.text(int(rng.integers(1, ctx.cat.n + 1))))
        else:
            it = Personalize((), tuple(int(x) for x in rng.integers(1, ctx.cat.n + 1, size=5)))
        issued.append((sess, it))
        reqs.append(Request(tok, sess, now, intent=it))
    for r in reqs:                       # sequential: each request sees the counters of the earlier ones
        gk.serve(r)
    return {"platform_calls": gk.stats["platform_calls"], "cache_hits": gk.stats["cache_hits"],
            "admitted": gk.stats["platform_calls"] + gk.stats["cache_hits"] + gk.stats["public_after_cap"],
            "denied_B": gk.stats["deny_B"], "denied_admission": gk.stats["deny_admission"],
            "parses_charged": n - gk.stats["deny_admission"]}
