"""Access-control policy pi(tau, iota) in {ALLOW, NARROW, DENY}, pure code (Sec. 4.1).

Intents within the token's scopes are allowed, excess scope is narrowed, the rest denied.
Example 2 of the paper is a unit test (tests/test_gatekeeper.py).
"""
from __future__ import annotations

from dataclasses import replace

from .intents import Compare, Details, Personalize, Search
from .tokens import Token

ALLOW, NARROW, DENY = "ALLOW", "NARROW", "DENY"
NONPUBLIC = {"score", "scores", "logit", "logits", "rank", "history", "histories", "interactions", "users",
             "embedding", "embeddings", "rating", "ratings"}


def evaluate(tok: Token, it, public_attrs: set[str], K: int, has_history=lambda acc: True):
    """Return (decision, narrowed intent or None)."""
    if it is None or it.name not in tok.scopes:
        return DENY, None
    if isinstance(it, Search):
        cons = tuple(c for c in it.constraints if c[0] in public_attrs)
        new = replace(it, constraints=cons)
        return (ALLOW if cons == it.constraints else NARROW), new
    if isinstance(it, Compare):
        if not it.items:
            return DENY, None
        items = it.items[:K]                         # never more than K records
        return (ALLOW if items == it.items else NARROW), replace(it, items=items)
    if isinstance(it, Details):
        attrs = tuple(a for a in it.attributes if a in public_attrs and a not in NONPUBLIC)
        if it.attributes and not attrs:
            return DENY, None                        # only non-public fields requested
        return (ALLOW if attrs == it.attributes else NARROW), replace(it, attributes=attrs)
    if isinstance(it, Personalize):
        # uncovered names are dropped identically whether or not they exist
        scope = tuple(s for s in it.scope if s in tok.covers)
        hist_ok = any(has_history(s) for s in scope)
        if not hist_ok and not it.session:
            return DENY, None
        return (ALLOW if scope == it.scope else NARROW), replace(it, scope=scope)
    return DENY, None
