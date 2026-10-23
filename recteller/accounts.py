"""Per-account counters p_u <= P, b_u <= B, n_u <= n_max per reservation window (Sec. 4.1,
Alg. 1, App. D "Counters").

``Reserve(x, X)`` atomically increments x if x < X (one lock = linearisable compare-and-swap,
H5). Counters are keyed by the *account* (default) or, for the ablation "- cap" and the
JSON top-K control, by the *session*. An optional sliding 30-day cap N_30 is reserved in the
same atomic step as n_u (App. "Per-account accounting", remark).
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque


class AccountStore:
    def __init__(self, P: int, B: int, n_max: int, window_hours: float = 24.0, cap30: int | None = None,
                 keying: str = "account", enabled: bool = True):
        assert keying in ("account", "session")
        self.limits = {"p": P, "b": B, "n": n_max}
        self.window = window_hours
        self.cap30 = cap30
        self.keying = keying
        self.enabled = enabled           # False = no budgets at all (open modes)
        self._lock = threading.Lock()
        self._c: dict[tuple, int] = defaultdict(int)
        self._hist30: dict[str, deque] = defaultdict(deque)   # reservation times of n_u

    def epoch(self, now: float) -> int:
        return int(now // self.window)

    def key(self, account: str, session: str) -> str:
        return account if self.keying == "account" else f"{account}/{session}"

    def reserve(self, name: str, account: str, session: str, now: float) -> bool:
        """Atomic Reserve(x_u, X) for x in {p, b, n}."""
        if not self.enabled:
            return True
        k = self.key(account, session)
        ck = (name, k, self.epoch(now))
        with self._lock:
            if self._c[ck] >= self.limits[name]:
                return False
            if name == "n" and self.cap30 is not None:
                dq = self._hist30[k]
                while dq and dq[0] <= now - 30 * 24:
                    dq.popleft()
                if len(dq) >= self.cap30:
                    return False
                dq.append(now)
            self._c[ck] += 1
            return True

    def refund(self, name: str, account: str, session: str, now: float) -> None:
        """Refund n_u only before scorer invocation (App. D)."""
        if not self.enabled:
            return
        k = self.key(account, session)
        with self._lock:
            self._c[(name, k, self.epoch(now))] -= 1
            if name == "n" and self.cap30 is not None and self._hist30[k]:
                self._hist30[k].pop()

    def count(self, name: str, account: str, session: str, now: float) -> int:
        return self._c[(name, self.key(account, session), self.epoch(now))]
