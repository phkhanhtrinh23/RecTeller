"""Client-agent prompts for every access mode (Sec. 5.1 "Access modes") and the two readers
of RecTeller's slotted message (Sec. 4.3): soft tokens through the projector (open-weight
client) and the compact text rendering (any client).

Every client receives the need text and the 20 candidate identifiers and titles, and the mode's
answer adds to that. Identifiers are task-local letters A..T.
"""
from __future__ import annotations

import json

from . import config as C
from .llm import LLM

SYSTEM = ("You are a personal AI agent shopping on behalf of one user. You must pick the single candidate "
          "item your user is most likely to choose next, using the user's request and any information "
          "the platform returned. Answer with the candidate's letter only.")

HEADER = "id | title | attribute values | level"


def _schema_block() -> str:
    """Ablation A2 (no parser): the client itself writes typed intents, so it reads the schema."""
    from .intents import SCHEMA
    return ("You query the platform with typed intents. Send exactly one JSON object per request in one "
            "of these forms (malformed requests are denied):\n" + json.dumps(SCHEMA))


SCHEMA_BLOCK = _schema_block()


def _title(cat, i, n=60):
    t = cat.titles[i]
    return t if len(t) <= n else t[: n - 1] + "…"


def task_block(cat, task) -> str:
    lines = [f"{C.LETTERS[k]}. {_title(cat, i, 90)}" for k, i in enumerate(task.cands)]
    return f"User's request: {task.need}\n\nCandidate items:\n" + "\n".join(lines)


def letter(task, i) -> str:
    return C.LETTERS[task.cands.index(i)]


def attr_values(cat, i, attrs) -> str:
    return ", ".join(str(cat.attrs[i].get(a)) for a in attrs if cat.attrs[i].get(a) is not None)


# ------------------------------------------------------------------ mode sections
def sec_block(cat, task, resp=None) -> str:
    return "The platform did not answer your agent (agents are blocked)."


def sec_web(cat, task, resp=None) -> str:
    pages = []
    for k, i in enumerate(task.cands):
        a = cat.attrs[i]
        feats = " ".join(f"<li>{f}</li>" for f in cat.features[i][:4])
        attrs = "".join(f"<tr><td>{kk}</td><td>{v}</td></tr>" for kk, v in a.items())
        pages.append(f"<div class='product' id='{C.LETTERS[k]}'><h1>{cat.titles[i]}</h1><table>{attrs}</table>"
                     f"<ul>{feats}</ul><button>Add to cart</button></div>")
    return "Rendered product pages you browsed:\n" + "\n".join(pages)


def _records(cat, task, items, extra=None, features=True):
    out = []
    for k, i in enumerate(items):
        r = {"id": letter(task, i), "title": _title(cat, i, 80), **cat.attrs[i]}
        if features and cat.features[i]:
            r["features"] = cat.features[i][:2]
        if extra is not None:
            r.update(extra[k])
        out.append(r)
    return out


def sec_mcp(cat, task, resp) -> str:
    top = resp.items[:10]
    s = (f"Tool call: recommend_for_session(user) -> {json.dumps([letter(task, i) for i in top])}\n"
         f"Tool call: view_product_information({json.dumps([letter(task, i) for i in top])}) -> "
         f"{json.dumps(_records(cat, task, top), ensure_ascii=False)}")
    return s


def sec_iagent(cat, task, resp, profile_items) -> str:
    prof = "; ".join(_title(cat, i, 60) for i in profile_items[-10:])
    return (sec_mcp(cat, task, resp) + "\n\nYour user-side profile (items your user interacted with, held by you, "
            f"most recent last): {prof}\nRerank the platform's recommendations with this profile before choosing.")


def sec_ranked(cat, task, resp) -> str:
    """An exact ranked list returned behind the per-account cap (random-replacement control)."""
    recs = _records(cat, task, resp.items, None, features=False)
    return "Platform gatekeeper response (ranked list, JSON):\n" + json.dumps(recs, ensure_ascii=False)


def render_compact(cat, task, resp) -> str:
    """Reader (b): deterministic compact text, one line per slot under a fixed header."""
    lines = [HEADER] + [f"{letter(task, i)} | {_title(cat, i, 50)} | {attr_values(cat, i, resp.attrs)} | {lv}"
                        for i, lv in zip(resp.items, resp.levels)]
    return "\n".join(lines)


def sec_ours_text(cat, task, resp) -> str:
    return ("Platform gatekeeper message (one line per slot, in the platform's order; level 1-4 = personalized "
            "for your user, higher is stronger; level 0 = public result):\n" + render_compact(cat, task, resp))


def sec_ours_soft(cat, task, resp) -> str:
    """Reader (a): each slot as its identifier in text followed by one soft token."""
    slots = "\n".join(f"[{letter(task, i)}] {LLM.SOFT_MARK}" for i in resp.items)
    return ("Platform gatekeeper message (one slot per item, in the platform's order; each slot is the item's "
            "letter followed by an embedding of the item, its attributes and its personalization level):\n" + slots)


SECTIONS = {"block": sec_block, "web": sec_web, "mcp": sec_mcp, "firewall": sec_mcp, "gro": sec_mcp,
            "ranked": sec_ranked,
            "ours_text": sec_ours_text, "ours_soft": sec_ours_soft}


def user_text(cat, task, section: str) -> str:
    return f"{task_block(cat, task)}\n\n{section}\n\nWhich candidate should your user choose? Answer with one letter (A-T)."


# ------------------------------------------------------------------ closed client
class ClosedClient:
    """GPT-4o-mini through the OpenAI API, temperature 0, one seed (reads text only)."""

    def __init__(self, model: str = C.CLOSED_MODEL):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        import tiktoken
        self.enc = tiktoken.get_encoding("o200k_base")

    def n_tokens(self, system, user) -> int:
        return len(self.enc.encode(system)) + len(self.enc.encode(user)) + 7

    def choose(self, system: str, users: list[str], workers: int = 16) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        def one(u):
            for _ in range(5):
                try:
                    r = self.client.chat.completions.create(
                        model=self.model, temperature=0, max_tokens=2,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": u}])
                    return (r.choices[0].message.content or "").strip()[:1].upper()
                except Exception as e:  # noqa: BLE001  (rate limits, transient errors)
                    import time
                    time.sleep(2)
            return ""

        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, users))
