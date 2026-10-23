"""Access modes as services over the same session-context interface (Sec. 5.1).

Gatekeeper modes (code authority, typed intents, policy, counters):
  ours      RecTeller: per-account cap, noisy release, slots         (release 'noisy')
  einf      RecTeller at eps_resp = inf (Capped top-k_p)             (release 'exact_kp')
  pub       RecTeller with public slots only (k_p = 0)
  jtopk     JSON top-K: per-session limits, exact top-K + quantised scores
  capped    Capped top-K: per-account cap, exact top-K + quantised scores
  stitch    Capped top-K + GRO (stitched gateway)
  rr<p>     Capped top-K with a share p replaced at random
  nocap     RecTeller with counters/cache keyed per session (ablation "- cap")
  neither   exact top-k_p keyed per session (ablation "- noise - cap")
Open modes (no per-account counters, every request reaches the recommender):
  openapi   exact scores of all candidates (+ item embeddings)
  mcp       recommender top-10 (ranked ids, no scores) + native catalogue tools
  firewall  LLM converter + deterministic verifier in front of the MCP tools
  iagent    user-side reranking of the MCP results
  gro       GRO-defended model behind the MCP interface (ranked lists, no scores)
  web, block  no recommender access
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .gatekeeper import Gatekeeper, Request, Response
from .intents import Personalise, Search
from .tokens import TokenAuthority

GK_MODES = {
    "ours": dict(release_kind="noisy", keying="account"),
    "einf": dict(release_kind="exact_kp", keying="account"),
    "pub": dict(release_kind="noisy", keying="account", kp=0),
    "jtopk": dict(release_kind="topK_json", keying="session"),
    "capped": dict(release_kind="topK_json", keying="account"),
    "stitch": dict(release_kind="topK_json", keying="account", gro=True),
    "nocap": dict(release_kind="noisy", keying="session"),
    "neither": dict(release_kind="exact_kp", keying="session"),
    "nolev": dict(release_kind="noisy", keying="account", use_levels=False),
    "kp1": dict(release_kind="noisy", keying="account", kp=1),
    "kp5": dict(release_kind="noisy", keying="account", kp=5),
}
OPEN_MODES = {
    "openapi": dict(release_kind="all_scores"),
    "mcp": dict(release_kind="topK_list"),
    "firewall": dict(release_kind="topK_list"),
    "iagent": dict(release_kind="topK_list"),
    "gro": dict(release_kind="topK_list", gro=True),
}
NO_RECSYS = ("web", "block")


@dataclass
class Context:
    cat: object
    pe: object
    retrieval: object
    scorer: object
    proto: object
    gro_scorer: object | None = None


def build(mode: str, ctx: Context, seed: int = 0, proto=None, auth: TokenAuthority | None = None,
          histories=None, cap30=None, parser=None) -> Gatekeeper:
    proto = proto or ctx.proto
    auth = auth or TokenAuthority(b"platform-key-0123456789abcdef!!!")
    if mode.startswith("rr"):
        cfg = dict(release_kind="random_replace", keying="account", p_replace=float(mode[2:]))
    elif mode in GK_MODES:
        cfg = dict(GK_MODES[mode])
    elif mode in OPEN_MODES:
        cfg = dict(OPEN_MODES[mode], keying="account", budgets=False)
    else:
        raise ValueError(mode)
    gro = cfg.pop("gro", False)
    scorer = ctx.gro_scorer if gro else ctx.scorer
    if gro and scorer is None:
        raise ValueError("GRO model not available")
    return Gatekeeper(ctx.cat, ctx.retrieval, scorer, proto, auth, histories=histories or {}, seed=seed,
                      cap30=cap30, parser=parser, **cfg)


def serve_tasks(mode: str, tasks, ctx: Context, seed: int = 0, proto=None, use_session: bool = False) -> list[Response]:
    """One fresh account per task: Search over the task's candidates, then Personalise."""
    if mode in NO_RECSYS:
        return [Response("OK", kind="none") for _ in tasks]
    hist = {f"u{k}": list(t.history) for k, t in enumerate(tasks)}
    gk = build(mode, ctx, seed, proto, histories=hist)
    ctx.retrieval.prime([t.need for t in tasks])
    reqs = []
    for k, t in enumerate(tasks):
        tok = gk.auth.issue(f"u{k}")
        reqs.append(Request(tok, "s0", 0.0, intent=Search(text=t.need, items=tuple(t.cands))))
    gk.serve_batch(reqs)
    reqs = []
    for k, t in enumerate(tasks):
        tok = gk.auth.issue(f"u{k}")
        it = Personalise((), tuple(t.history)) if use_session else Personalise((f"u{k}",), ())
        reqs.append(Request(tok, "s0", 0.0, intent=it))
    return gk.serve_batch(reqs)


def serve_account_days(mode: str, tasks, ctx: Context, n: int = 12, seed: int = 0) -> list[Response]:
    """Fig. budget (a): groups of n tasks served one after another to one fresh account in one window."""
    hist = {}
    gk = build(mode, ctx, seed, histories=hist)
    ctx.retrieval.prime([t.need for t in tasks])
    out = []
    for g in range(0, len(tasks), n):
        acct = f"day{g // n}"
        tok = gk.auth.issue(acct)
        for j, t in enumerate(tasks[g:g + n]):
            now = 1.0 + j * 0.5
            gk.serve(Request(tok, f"s{j}", now, intent=Search(text=t.need, items=tuple(t.cands))))
            out.append(gk.serve(Request(tok, f"s{j}", now, intent=Personalise((), tuple(t.history)))))
    return out
