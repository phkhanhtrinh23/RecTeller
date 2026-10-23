"""Need texts written by a fixed LLM prompt from the held-out item's public attributes
(App. C.1): category, price band and two features (Amazon) or genres and release decade
(ML-1M), never the title or the brand."""
from __future__ import annotations

import re

SYSTEM = ("You write short shopping requests that a user gives to their personal AI agent. "
          "Write one or two sentences in the first person. Never mention product names, titles, "
          "brands or franchises. Do not invent details beyond the attributes given.")


def need_prompt(cat_name: str, a: dict) -> str:
    if cat_name == "ml":
        return (f"Attributes of the movie the user wants: genres = {a.get('genres')}; release decade = "
                f"{a.get('decade')}.\nWrite the user's request for a movie to watch.")
    parts = [f"category = {a.get('category')}"]
    if "platform" in a:
        parts.append(f"platform = {a.get('platform')}")
    parts.append(f"price band = {a.get('price')}")
    # Only coarse attributes are given so that the need does not single out one candidate.
    return (f"Attributes of the product the user wants: {'; '.join(parts)}.\n"
            "Write the user's request in at most 25 words. Describe only the general kind of product and what it "
            "is for, in plain everyday words, so that many similar products in this category would match. Do not "
            "copy feature wording; no numbers, model names, characters, franchises or other distinctive details.")


def template_need(cat_name: str, a: dict) -> str:
    """Deterministic fallback (used only when no LLM is available)."""
    if cat_name == "ml":
        return f"I'd like to watch a {a.get('genres', '').lower()} movie from the {a.get('decade')}."
    f = (a.get("features") or [""])[0][:80]
    plat = f" for {a['platform']}" if "platform" in a else ""
    return f"I'm looking for {a.get('category', 'an item').lower()}{plat}, {a.get('price')}. {f}".strip()


def clean(t: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip().strip('"').strip()
    return t.split("\n")[0][:400]


def write_needs(llm, cat, attr_list: list[dict], bs: int = 64) -> list[str]:
    prompts = [llm.chat(SYSTEM, need_prompt(cat.name, a)) for a in attr_list]
    return [clean(t) for t in llm.generate(prompts, max_new_tokens=70, bs=bs)]
