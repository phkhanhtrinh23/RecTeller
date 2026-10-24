"""Delegation tokens tau = (u, Sigma_tau, exp) (Sec. 3, "Personal agents, delegation and
sessions"), signed with HMAC-SHA256. Budgets are keyed by the account u, never the token."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field

INTENTS = ("SEARCH", "COMPARE", "DETAILS", "PERSONALIZE")


@dataclass(frozen=True)
class Token:
    account: str
    scopes: frozenset[str]                 # intent names this token may use
    covers: frozenset[str] = frozenset()   # accounts whose histories PERSONALIZE may read
    exp: float = float("inf")              # expiry (simulated hours)
    sig: str = ""


class TokenAuthority:
    """Issues and verifies tokens for the platform (the account owner signs the delegation)."""

    def __init__(self, key: bytes | None = None):
        self.key = key or os.urandom(32)

    def _mac(self, t: Token) -> str:
        msg = json.dumps([t.account, sorted(t.scopes), sorted(t.covers), t.exp], default=str).encode()
        return base64.urlsafe_b64encode(hmac.new(self.key, msg, hashlib.sha256).digest()).decode()

    def issue(self, account: str, scopes=INTENTS, covers=None, exp: float = float("inf")) -> Token:
        covers = frozenset([account] if covers is None else covers)
        t = Token(account, frozenset(scopes), covers, exp)
        return Token(t.account, t.scopes, t.covers, t.exp, self._mac(t))

    def verify(self, t: Token | None, now: float) -> bool:
        if t is None or not isinstance(t, Token):
            return False
        return hmac.compare_digest(t.sig, self._mac(t)) and now < t.exp
