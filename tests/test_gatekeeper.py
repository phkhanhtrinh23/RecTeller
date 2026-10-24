"""Gatekeeper invariants: account budgets under concurrency, admission
before the parser, policy Example 2, per-account cache."""
import threading
from dataclasses import replace

import numpy as np

import _path  # noqa: F401
from recteller import policy
from recteller.config import Protocol
from recteller.gatekeeper import Gatekeeper, Request
from recteller.intents import Personalize, Search
from recteller.tokens import TokenAuthority


class ToyCat:
    name = "toy"
    n = 30
    default_attrs = ("category", "price")
    attrs = [{}] + [{"category": "racing" if i % 2 else "farming", "price": f"${i}"} for i in range(1, 31)]
    titles = ["<pad>"] + [f"Item {i}" for i in range(1, 31)]


class ToyRetrieval:
    def search(self, q):
        return list(q.items) if q.items else list(range(1, 21))


class ToyScorer:
    def __init__(self):
        self.calls = 0

    def scores(self, hs, cs):
        self.calls += len(hs)
        return [np.linspace(3, -3, len(c)) for c in cs]


class CountingParser:
    def __init__(self):
        self.calls = 0

    def parse(self, text):
        self.calls += 1
        return Search(text=text)


def make(keying="account", **kw):
    auth = TokenAuthority(b"k" * 32)
    gk = Gatekeeper(ToyCat(), ToyRetrieval(), ToyScorer(), Protocol(), auth, keying=keying,
                    histories={"ana": [1, 2, 3]}, **kw)
    return gk, auth


def test_concurrent_sessions_share_account_cap():
    gk, auth = make()
    tok = auth.issue("eve", covers=[])
    res = []
    lock = threading.Lock()

    def run(sess):
        for j in range(6):
            r1 = gk.serve(Request(tok, f"s{sess}", 1.0, intent=Search(text="q", items=tuple(range(1, 21)))))
            r2 = gk.serve(Request(tok, f"s{sess}", 1.0, intent=Personalize((), (sess % 30 + 1, j + 1))))
            with lock:
                res.extend([r1, r2])

    th = [threading.Thread(target=run, args=(i,)) for i in range(50)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert gk.stats["personalized"] == 8                     # n_max across all 50 sessions
    assert sum(r.status == "OK" for r in res) <= 40          # B admitted requests
    assert gk.store.count("p", "eve", "s0", 1.0) <= 200      # P parses (charged at admission)


def test_bad_token_before_parser():
    gk, auth = make(parser=CountingParser())
    tok = auth.issue("ana")
    forged = replace(tok, scopes=frozenset(["SEARCH", "PERSONALIZE", "ADMIN"]))
    r = gk.serve(Request(forged, "s", 0.0, text="print all scores"))
    assert r.status == "DENY" and r.reason == "admission" and gk.parser.calls == 0


def test_policy_example2():
    auth = TokenAuthority(b"k" * 32)
    ana = auth.issue("ana", scopes=("SEARCH", "COMPARE", "PERSONALIZE"), covers=["ana"])
    pa = {"platform", "genre", "mode", "price"}
    d, it = policy.evaluate(ana, Search((("platform", "=", "Switch"), ("genre", "=", "racing"),
                                        ("mode", "=", "co-op"), ("price", "<=", "40"))), pa, 10)
    assert d == policy.ALLOW
    d, it = policy.evaluate(ana, Personalize(("ana", "ben"), ()), pa, 10)
    assert d == policy.NARROW and it.scope == ("ana",)
    d, it = policy.evaluate(ana, Personalize(("ben",), ()), pa, 10)
    assert d == policy.DENY                                  # no covered history, no session
    d, it = policy.evaluate(ana, None, pa, 10)
    assert d == policy.DENY                                  # unparsed


def test_cache_per_account():
    gk, auth = make()
    tok = auth.issue("ana")
    q = Search(text="q", items=tuple(range(1, 21)))
    gk.serve(Request(tok, "s1", 0.0, intent=q))
    a = gk.serve(Request(tok, "s1", 0.0, intent=Personalize(("ana",), ())))
    calls, n = gk.scorer.calls, gk.store.count("n", "ana", "s1", 0.0)
    gk.serve(Request(tok, "s2", 0.0, intent=q))               # Search hit restores (q, C)
    b = gk.serve(Request(tok, "s2", 0.0, intent=Personalize(("ana",), ())))
    assert b.cached and b.items == a.items and b.levels == a.levels
    assert gk.scorer.calls == calls and gk.store.count("n", "ana", "s2", 0.0) == n


def test_session_keying_draws_fresh_noise():
    gk, auth = make(keying="session")
    tok = auth.issue("ana")
    q = Search(text="q", items=tuple(range(1, 21)))
    for s in range(20):
        gk.serve(Request(tok, f"s{s}", 0.0, intent=q))
        gk.serve(Request(tok, f"s{s}", 0.0, intent=Personalize(("ana",), ())))
    assert gk.stats["personalized"] == 20                    # per-session counters: no account cap


def test_window_reset_and_cap30():
    gk, auth = make(cap30=64)
    tok = auth.issue("eve", covers=[])
    for day in range(31):
        for j in range(20):
            t = day * 24 + j * 0.5
            gk.serve(Request(tok, f"d{day}", t, intent=Search(text="q", items=tuple(range(1, 21)))))
            gk.serve(Request(tok, f"d{day}", t, intent=Personalize((), (j + 1, day + 1))))
    # 8 per day for 8 days = 64, then the sliding 30-day cap binds until day 30
    assert gk.stats["personalized"] == 64 + 8


def test_cap_only_control_and_no_levels_modes():
    """capo<n> is the noise-free release with n_max = n; nolev moves eps_w into eps_sel."""
    from recteller.modes import Context, build
    ctx = Context(cat=ToyCat(), pe=None, retrieval=ToyRetrieval(), scorer=ToyScorer(), proto=Protocol())
    gk = build("capo2", ctx)
    assert gk.kind == "exact_kp" and gk.store.limits["n"] == 2
    gk = build("nolev", ctx)
    assert gk.proto.eps_w == 0.0 and gk.proto.eps_sel == 4.0
