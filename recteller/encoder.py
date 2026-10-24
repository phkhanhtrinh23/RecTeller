"""Public encoder E and slotted messages (Sec. 4.3, Eq. slot).

z_k = [ e(o_k) || sum_v alpha_{k,v} e(v) || phi(level_k) ]

e(i) = E(x_i) and e(v) = E(v) come from a fixed, published sentence encoder over public
catalog text only (H2). phi is a learned level table owned by the projector, so the
transported array here holds the first two blocks plus the integer level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import config as C


_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(C.ENCODER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    return _MODEL


def embed(texts: list[str], bs: int = 256) -> np.ndarray:
    return model().encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False).astype(np.float32)


class PublicEmbeddings:
    """e(i) for every catalog item and e(v) for every attribute value, cached on disk."""

    def __init__(self, cat):
        self.cat = cat
        path = C.CACHE / f"emb_{cat.name}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            self.item = z["item"]
            self.vals = list(z["vals"])
            self.val_emb = z["val_emb"]
        else:
            C.ensure_dirs()
            texts = [""] + [cat.text(i) for i in range(1, cat.n + 1)]
            self.item = embed(texts)
            self.item[0] = 0
            vals = sorted({f"{k}: {v}" for a in cat.attrs[1:] for k, v in a.items()})
            # multi-valued genres embed per genre
            if cat.name == "ml":
                vals = sorted(set(vals) | {f"genre: {g}" for a in cat.attrs[1:] for g in a["genres"].split(", ")})
            self.vals = vals
            self.val_emb = embed(vals)
            np.savez(path, item=self.item, vals=np.array(vals, dtype=object), val_emb=self.val_emb)
        self.vidx = {v: k for k, v in enumerate(self.vals)}
        self.d = self.item.shape[1]

    def values_of(self, i: int, attrs) -> list[str]:
        out = []
        for a in attrs:
            v = self.cat.attrs[i].get(a)
            if v is None:
                continue
            if a == "genres":
                out += [f"genre: {g}" for g in v.split(", ")]
            else:
                out.append(f"{a}: {v}")
        return out

    def attr_block(self, i: int, attrs) -> np.ndarray:
        vs = [self.vidx[v] for v in self.values_of(i, attrs) if v in self.vidx]
        if not vs:
            return np.zeros(self.d, dtype=np.float32)          # item without values -> zero vector
        return self.val_emb[vs].mean(0)                         # alpha spreads unit weight


@dataclass
class Slots:
    """A slotted message: K' slots in the order S then F."""
    items: list[int]
    item_emb: np.ndarray     # (K', d)
    attr_emb: np.ndarray     # (K', d)
    levels: np.ndarray       # (K',) ints in {0..L}

    def array(self, d_phi_table: np.ndarray | None = None) -> np.ndarray:
        parts = [self.item_emb, self.attr_emb]
        if d_phi_table is not None:
            parts.append(d_phi_table[self.levels])
        return np.concatenate(parts, 1)


def make_slots(pe: PublicEmbeddings, items: list[int], levels: list[int], attrs=None,
               shuffle_attrs_rng=None) -> Slots:
    attrs = attrs or pe.cat.default_attrs
    ie = pe.item[items]
    ae = np.stack([pe.attr_block(i, attrs) for i in items]) if items else np.zeros((0, pe.d), np.float32)
    if shuffle_attrs_rng is not None and len(items) > 1:      # ablation A3
        ae = ae[shuffle_attrs_rng.permutation(len(items))]
    return Slots(list(items), ie, ae, np.asarray(levels, dtype=np.int64))
