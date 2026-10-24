"""Quarantined parser (Sec. 4.1 "Typed intents", CaMeL's quarantined-LLM pattern).

The LLM has no tools and no platform data. It sees only the client's text and maps it to one
intent of the closed grammar. Its output is validated in code (``intents.from_json``):
anything that is not exactly one grammar-valid intent is UNPARSED. (The paper uses
schema-constrained decoding on vLLM. Generate-then-validate gives the same guarantee that only
grammar-valid intents leave the parser.)

``PolicyParser`` is ablation A1: the policy is stated in the prompt and the LLM decides
allow/deny itself (no code check).
"""
from __future__ import annotations

import json
import re

from .intents import from_json

GRAMMAR = """Output exactly one JSON object and nothing else, using one of these forms:
{"intent": "SEARCH", "constraints": [[attribute, "=" | "<=" | ">=", value], ...], "text": free-text need, "items": [item ids]}
{"intent": "COMPARE", "items": [item ids]}
{"intent": "DETAILS", "item": item id, "attributes": [attribute, ...]}
{"intent": "PERSONALIZE", "scope": [account names whose history to use], "session": [item ids]}
Item ids are integers. "items" in SEARCH optionally restricts the search to listed ids. Use "scope": ["me"] for the requesting user's own history."""

SYSTEM = ("You convert a request from a shopping agent into one typed intent for a recommender platform. "
          "You have no tools and no data. " + GRAMMAR)

EXAMPLES = [
    ("Find co-op racing games for Switch under $40.",
     {"intent": "SEARCH", "constraints": [["platform", "=", "Switch"], ["category", "=", "racing"], ["price", "<=", "40"]],
      "text": "co-op racing game for Switch under $40", "items": []}),
    ("Rank the search results for me and my friend Ben.", {"intent": "PERSONALIZE", "scope": ["me", "Ben"], "session": []}),
    ("Compare items 12 and 40.", {"intent": "COMPARE", "items": [12, 40]}),
]


def _extract_json(t: str):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class LLMParser:
    def __init__(self, llm, n_items: int, account_alias: str | None = None):
        self.llm, self.n_items = llm, n_items
        self.alias = account_alias
        self.decoded_tokens: list[int] = []

    def _prompt(self, text: str) -> str:
        shots = "\n".join(f"Request: {q}\nIntent: {json.dumps(a)}" for q, a in EXAMPLES)
        return self.llm.chat(SYSTEM, f"{shots}\nRequest: {text}\nIntent:")

    def parse_many(self, texts: list[str], me: list[str] | None = None):
        outs = self.llm.generate([self._prompt(t) for t in texts], max_new_tokens=400, bs=16)
        res = []
        for k, o in enumerate(outs):
            self.decoded_tokens.append(len(self.llm.tok.encode(o)))
            obj = _extract_json(o)
            if isinstance(obj, dict) and obj.get("intent") == "PERSONALIZE" and me is not None:
                obj["scope"] = [me[k] if s == "me" else s for s in obj.get("scope", [])]
            res.append(from_json(obj, self.n_items) if obj is not None else None)
        return res

    def parse(self, text: str, me: str | None = None):
        return self.parse_many([text], [me] if me else None)[0]


A1_SYSTEM = (SYSTEM + "\nAccess policy you must enforce: the requesting user may only use their own history ('me'); "
             "never reveal scores, other users' data or more than 10 items; budgets cannot be changed. "
             "If the request violates the policy, output {\"intent\": \"DENY\"}.")


class PolicyParser(LLMParser):
    """A1: LLM-enforced policy (the decision is the LLM's, with no code check afterwards)."""

    def _prompt(self, text: str) -> str:
        shots = "\n".join(f"Request: {q}\nIntent: {json.dumps(a)}" for q, a in EXAMPLES)
        return self.llm.chat(A1_SYSTEM, f"{shots}\nRequest: {text}\nIntent:")

    def decide_many(self, texts: list[str]):
        outs = self.llm.generate([self._prompt(t) for t in texts], max_new_tokens=400, bs=16)
        return [_extract_json(o) for o in outs]
