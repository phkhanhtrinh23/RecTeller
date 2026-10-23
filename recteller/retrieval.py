"""Public retrieval C(q) with public relevance r_i(q) (Sec. 3, H1: reads no f_theta and no
interaction data, only public item text/attributes and the public encoder)."""
from __future__ import annotations

import numpy as np

from .encoder import PublicEmbeddings, embed
from .intents import Search


def _satisfies(val: str, op: str, want: str) -> bool:
    if op == "=":
        return want.lower() in str(val).lower()
    try:  # numeric comparison on prices "$15-$30" / "under $15"
        nums = [float(x) for x in "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in str(val)).split()]
        w = float("".join(ch for ch in want if ch.isdigit() or ch == "."))
        if not nums:
            return False
        return (min(nums) <= w) if op == "<=" else (max(nums) >= w)
    except ValueError:
        return False


class Retrieval:
    def __init__(self, pe: PublicEmbeddings, size: int = 100):
        self.pe = pe
        self.cat = pe.cat
        self.size = size
        self._qcache: dict[str, np.ndarray] = {}
        self._item_text = None

    def qvec(self, text: str) -> np.ndarray:
        if self._item_text is None:  # a query equal to an item's public text reuses e(i)
            self._item_text = {self.cat.text(i): i for i in range(1, self.cat.n + 1)}
        if text in self._item_text:
            return self.pe.item[self._item_text[text]]
        if text not in self._qcache:
            self._qcache[text] = embed([text])[0]
        return self._qcache[text]

    def prime(self, texts: list[str]) -> None:
        """Batch-encode request texts (speed only, same vectors as qvec)."""
        new = [t for t in dict.fromkeys(texts) if t and t not in self._qcache]
        if new:
            for t, v in zip(new, embed(new)):
                self._qcache[t] = v

    def relevance(self, q: Search, items: np.ndarray) -> np.ndarray:
        """r_i(q): encoder similarity to the request text plus satisfied constraints. Ranks
        rather than filters, so near-misses still enter C(q)."""
        txt = q.text or " ".join(f"{a} {o} {v}" for a, o, v in q.constraints)
        r = self.pe.item[items] @ self.qvec(txt) if txt else np.zeros(len(items), np.float32)
        for a, op, v in q.constraints:
            r = r + 0.2 * np.array([_satisfies(self.cat.attrs[i].get(a, ""), op, v) for i in items])
        return r

    def neighbours(self) -> np.ndarray:
        """Top-|C| public neighbours of every item (cached, identical to search(text=x_i))."""
        if getattr(self, "_nb", None) is None:
            import torch
            E = torch.as_tensor(self.pe.item, device="cuda" if torch.cuda.is_available() else "cpu")
            out = []
            for b in range(1, len(E), 4096):
                out.append(torch.topk(E[b:b + 4096] @ E[1:].T, self.size, dim=1).indices.cpu().numpy() + 1)
            self._nb = np.concatenate([np.zeros((1, self.size), dtype=np.int64)] + out)
        return self._nb

    def search(self, q: Search) -> list[int]:
        """C(q) ordered by public relevance (the order of the public fill)."""
        if not q.items and not q.constraints and q.text:
            if self._item_text is None:
                self.qvec(q.text)
            if q.text in self._item_text:          # "more like this" fast path
                return [int(i) for i in self.neighbours()[self._item_text[q.text]]]
        items = np.array(q.items if q.items else range(1, self.cat.n + 1))
        r = self.relevance(q, items)
        order = np.argsort(-r, kind="stable")[: self.size]
        return [int(items[j]) for j in order]

    def search_by_vector(self, v: np.ndarray, exclude=()) -> list[int]:
        """'More like this' search used by scripted attackers (public embedding of an item)."""
        r = self.pe.item[1:] @ v
        if exclude:
            r[np.array(list(exclude)) - 1] = -9
        return [int(j + 1) for j in np.argsort(-r)[: self.size]]
