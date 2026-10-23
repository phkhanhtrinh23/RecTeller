"""Gatekeeper sessions: Algorithm 1 ("serving one request in a session") end to end.

Lines 1-6 are C1 (admission, quarantined parse, code policy, request budget, log, cache),
lines 7-11 C2 (atomic reservation, scoring, release, public fill), lines 12-13 C3 (slots,
done by the reader in ``clients.py`` from the released tuple and public data only).

The same class also serves the controls and baselines of Sec. 5 by configuration:
``release_kind`` in {noisy, exact_kp, topK_json, random_replace, topK_list, all_scores},
``keying`` in {account, session} and ``budgets`` on/off. Requests are processed in order
for all counter and cache decisions, and scoring of the admitted personalised requests is
batched (``serve_batch``) so long attacks run on the GPU.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import policy
from .accounts import AccountStore
from .intents import Compare, Details, Personalise, Search
from .release import Release, release
from .tokens import Token, TokenAuthority


@dataclass
class Request:
    token: Token | None
    session: str
    now: float = 0.0
    text: str | None = None          # free text (goes through the quarantined parser)
    intent: object | None = None     # typed intent (A2 path: no parser)


@dataclass
class Response:
    status: str                       # OK | DENY
    reason: str = ""
    intent: object | None = None
    items: list[int] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)
    n_pers: int = 0
    scores: list[float] | None = None
    kind: str = ""
    attrs: tuple = ()                 # attributes for the slots' attribute blocks
    query: str = ""
    cached: bool = False
    platform_call: bool = False
    personalised: bool = False
    details: dict | None = None


@dataclass
class SessionState:
    q: Search | None = None
    cands: list[int] = field(default_factory=list)   # C(q) ordered by public relevance


class Gatekeeper:
    def __init__(self, cat, retrieval, scorer, proto, authority: TokenAuthority, release_kind: str = "noisy",
                 keying: str = "account", budgets: bool = True, cache: bool | None = None, parser=None,
                 histories: dict | None = None, seed: int = 0, kp: int | None = None, p_replace: float = 0.0,
                 use_levels: bool = True, cap30: int | None = None):
        self.cat, self.retrieval, self.scorer, self.proto = cat, retrieval, scorer, proto
        self.auth = authority
        self.kind = release_kind
        self.store = AccountStore(proto.P, proto.B, proto.n_max, proto.window_hours, cap30 or proto.cap30,
                                  keying=keying, enabled=budgets)
        self.cache_on = budgets if cache is None else cache
        self.cache: dict = {}
        self.parser = parser
        self.histories = histories or {}
        self.rng = np.random.default_rng(seed)
        self.kp = proto.kp if kp is None else kp
        self.p_replace = p_replace
        self.use_levels = use_levels
        self.sessions: dict = defaultdict(SessionState)
        self.public_attrs = set(cat.default_attrs) | {k for a in cat.attrs[1:200] for k in a}
        self.log: list[tuple] = []        # (account, intent name, agent-traffic label)
        self.stats = defaultdict(int)

    # -------------------------------------------------------------- helpers
    def _owner(self, tok: Token, session: str) -> str:
        return self.store.key(tok.account, session)

    def _skey(self, tok: Token, session: str):
        return (tok.account, session)

    def _public_message(self, st: SessionState, n: int | None = None) -> tuple[list[int], list[int]]:
        n = min(self.proto.K, len(st.cands)) if n is None else n
        return st.cands[:n], [0] * n

    def _attrs(self, st: SessionState) -> tuple:
        if st.q is not None and st.q.constraints:
            a = tuple(dict.fromkeys(c[0] for c in st.q.constraints if c[0] in self.public_attrs))
            if a:
                return a
        return tuple(self.cat.default_attrs)

    # -------------------------------------------------------------- Algorithm 1
    def serve(self, req: Request) -> Response:
        return self.serve_batch([req])[0]

    def serve_batch(self, reqs: list[Request]) -> list[Response]:
        out: list[Response | None] = [None] * len(reqs)
        jobs = []                                    # (index, history, st, cache key, attrs)
        for k, r in enumerate(reqs):
            tok = r.token
            # line 1: token valid and Reserve(p_u, P), before any LLM runs
            if not self.auth.verify(tok, r.now) or not self.store.reserve("p", tok.account, r.session, r.now):
                out[k] = Response("DENY", "admission")
                self.stats["deny_admission"] += 1
                continue
            # line 2: quarantined parser (or typed intent on the A2 path)
            it = r.intent
            if it is None and r.text is not None and self.parser is not None:
                it = self.parser.parse(r.text)
                self.stats["parser_calls"] += 1
            # line 3: unparsed or denied by pi -> DENY, before any platform call
            dec, it = policy.evaluate(tok, it, self.public_attrs, self.proto.K,
                                      has_history=lambda a: bool(self.histories.get(a)))
            if dec == policy.DENY:
                out[k] = Response("DENY", "policy")
                self.stats["deny_policy"] += 1
                continue
            # line 4: narrow, then Reserve(b_u, B)
            if not self.store.reserve("b", tok.account, r.session, r.now):
                out[k] = Response("DENY", "budget_B", it)
                self.stats["deny_B"] += 1
                continue
            # line 5: log (tau, iota), then cache lookup keyed by owner, narrowed intent, C, epoch
            self.log.append((tok.account, it.name, "agent"))
            st = self.sessions[self._skey(tok, r.session)]
            epoch = self.store.epoch(r.now)
            owner = self._owner(tok, r.session)
            ckey = (owner, it.key(), tuple(st.cands) if isinstance(it, Personalise) else (), epoch)
            if self.cache_on and ckey in self.cache:
                resp, qc = self.cache[ckey]
                if isinstance(it, Search):
                    st.q, st.cands = qc
                out[k] = Response(**{**resp.__dict__, "cached": True, "platform_call": False})
                self.stats["cache_hits"] += 1
                continue
            # line 6: candidates of iota
            if isinstance(it, Search):
                st.q, st.cands = it, self.retrieval.search(it)
                items, lv = self._public_message(st)
                resp = Response("OK", "", it, items, lv, 0, kind="public", attrs=self._attrs(st),
                                query=it.text, platform_call=True)
                self._finish(out, k, resp, ckey, st)
                continue
            if isinstance(it, (Compare, Details)):
                items = list(it.items) if isinstance(it, Compare) else [it.item]
                attrs = (it.attributes if isinstance(it, Details) and it.attributes else tuple(self.cat.default_attrs))
                det = {i: {a: self.cat.attrs[i].get(a) for a in attrs} for i in items}
                resp = Response("OK", "", it, items, [0] * len(items), 0, kind="public", attrs=attrs,
                                platform_call=True, details=det)
                self._finish(out, k, resp, ckey, st)
                continue
            # PERSONALISE
            if not st.cands:                          # no stored candidate set: empty, nothing reserved
                out[k] = Response("OK", "no_candidates", it, [], [], 0, kind="empty")
                continue
            # line 7: Reserve(n_u, n_max) in one atomic step, before any score is computed
            if self.store.reserve("n", tok.account, r.session, r.now):
                hist = []
                for a in it.scope:
                    hist += list(self.histories.get(a, []))
                hist += list(it.session)
                jobs.append((k, hist, SessionState(st.q, list(st.cands)), ckey, it))
            else:
                items, lv = self._public_message(st)
                resp = Response("OK", "cap", it, items, lv, 0, kind="public", attrs=self._attrs(st))
                self.stats["public_after_cap"] += 1
                self._finish(out, k, resp, ckey, st)
        # lines 8-10: scores read only after the reservation, batched over admitted jobs
        if jobs:
            S = self.scorer.scores([j[1] for j in jobs], [j[2].cands for j in jobs])
            for (k, hist, st, ckey, it), s in zip(jobs, S):
                rel: Release = release(s, st.cands, st.cands, self.proto, self.rng, kind=self.kind, kp=self.kp,
                                       p_replace=self.p_replace, use_levels=self.use_levels)
                resp = Response("OK", "", it, rel.items, rel.levels, rel.n_pers, rel.scores, rel.kind,
                                self._attrs(st), st.q.text if st.q else "", platform_call=True, personalised=True)
                self.stats["personalised"] += 1
                self._finish(out, k, resp, ckey, st)
        return out  # type: ignore[return-value]

    def _finish(self, out, k, resp: Response, ckey, st: SessionState):
        # line 12: cache (message, q, C) for the owner
        if self.cache_on:
            self.cache[ckey] = (resp, (st.q, list(st.cands)))
        if resp.platform_call:
            self.stats["platform_calls"] += 1
        out[k] = resp
