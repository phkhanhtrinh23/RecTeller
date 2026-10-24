"""Typed-intent grammar (App. D "Typed-intent grammar").

The quarantined parser emits exactly one intent as JSON, and ``from_json`` validates it against
this closed grammar. Anything else is ``None`` (UNPARSED).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

OPS = ("=", "<=", ">=")


@dataclass(frozen=True)
class Search:
    constraints: tuple = ()      # ((attribute, op, value), ...)
    text: str = ""               # free-text need (public relevance only)
    items: tuple = ()            # optional restriction to listed public item ids
    name: str = "SEARCH"

    def key(self):
        return ("SEARCH", self.constraints, self.text, self.items)


@dataclass(frozen=True)
class Compare:
    items: tuple = ()
    name: str = "COMPARE"

    def key(self):
        return ("COMPARE", self.items)


@dataclass(frozen=True)
class Details:
    item: int = 0
    attributes: tuple = ()
    name: str = "DETAILS"

    def key(self):
        return ("DETAILS", self.item, self.attributes)


@dataclass(frozen=True)
class Personalize:
    scope: tuple = ()            # account names whose histories to use
    session: tuple = ()          # client-supplied session items
    name: str = "PERSONALIZE"

    def key(self):
        return ("PERSONALIZE", self.scope, self.session)


Intent = Search | Compare | Details | Personalize

SCHEMA = {
    "oneOf": [
        {"intent": "SEARCH", "constraints": [["attribute", "= | <= | >=", "value"]], "text": "str",
         "items": ["item id"]},
        {"intent": "COMPARE", "items": ["item id"]},
        {"intent": "DETAILS", "item": "item id", "attributes": ["attribute"]},
        {"intent": "PERSONALIZE", "scope": ["account name"], "session": ["item id"]},
    ]
}


def _ints(xs, n_items: int, limit: int = 200):
    if not isinstance(xs, list) or len(xs) > limit:
        raise ValueError
    out = []
    for x in xs:
        x = int(x)
        if not (1 <= x <= n_items):
            raise ValueError
        out.append(x)
    return tuple(out)


def from_json(obj, n_items: int) -> Intent | None:
    """Validate a parser output against the closed grammar (None = UNPARSED)."""
    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
        if not isinstance(obj, dict):
            return None
        kind = obj.get("intent")
        allowed = {"SEARCH": {"intent", "constraints", "text", "items"}, "COMPARE": {"intent", "items"},
                   "DETAILS": {"intent", "item", "attributes"}, "PERSONALIZE": {"intent", "scope", "session"}}
        if kind not in allowed or set(obj) - allowed[kind]:
            return None
        if kind == "SEARCH":
            cons = []
            for c in obj.get("constraints", []):
                a, op, v = c
                if op not in OPS:
                    return None
                cons.append((str(a)[:40], op, str(v)[:60]))
            return Search(tuple(cons), str(obj.get("text", ""))[:300], _ints(obj.get("items", []), n_items, 100))
        if kind == "COMPARE":
            return Compare(_ints(obj.get("items", []), n_items, 50))
        if kind == "DETAILS":
            return Details(int(_ints([obj["item"]], n_items)[0]), tuple(str(a)[:40] for a in obj.get("attributes", [])))
        if kind == "PERSONALIZE":
            scope = obj.get("scope", [])
            if not isinstance(scope, list):
                return None
            return Personalize(tuple(str(s)[:40] for s in scope), _ints(obj.get("session", []), n_items, 200))
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return None
