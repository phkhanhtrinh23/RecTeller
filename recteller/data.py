"""Datasets, public item attributes, leave-last-out splits and pick-one tasks.

Datasets (Sec. 5.1): Amazon Reviews 2023 5-core ``Video_Games`` (games) and
``Industrial_and_Scientific`` (ind), and MovieLens-1M (ml), all split leave-last-out.

Every item gets public text ``x_i`` and public attribute values ``V_i`` (H1: nothing here
reads interaction data except the popularity-free 5-core filter that defines the catalog).
"""
from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

AMAZON = {"games": "Video_Games", "ind": "Industrial_and_Scientific"}


@dataclass
class Catalog:
    """Public catalog. Item index 0 is padding, real items are 1..n."""
    name: str
    ids: list[str]                      # raw id per index (index 0 = "<pad>")
    titles: list[str]
    attrs: list[dict[str, str]]         # public attribute values V_i per item
    features: list[list[str]]           # public feature strings (need-text source)
    leaf: list[str]                     # leaf category (negative sampling / retrieval)
    default_attrs: tuple[str, ...]      # attributes used in slots when no Search constrains

    @property
    def n(self) -> int:
        return len(self.ids) - 1

    def text(self, i: int) -> str:
        """Public item text x_i."""
        a = "; ".join(f"{k}: {v}" for k, v in self.attrs[i].items())
        f = " ".join(self.features[i][:2])
        return f"{self.titles[i]}. {a}. {f}".strip()


Catalogue = Catalog  # alias so datasets pickled under the former class name still load


@dataclass
class Split:
    """Leave-last-out split. ``seqs[u]`` is user u's full chronological item list."""
    seqs: dict[int, list[int]]
    members: list[int]                 # users whose prefixes train the victim
    nonmembers: list[int]              # MIA non-members, excluded from victim training
    test_users: list[int]              # pick-one test tasks (members)
    val_users: list[int]               # disjoint validation users (members)
    shadow_members: list[int]          # attacker's shadow model training users
    shadow_nonmembers: list[int]
    mia_members: list[int]             # victim members targeted by MIA
    session_users: list[int]           # held-out real sessions for Agr@10


@dataclass
class Task:
    uid: int
    history: list[int]                 # last proto.hist items before the target
    target: int
    cands: list[int]                   # 20 candidates (shuffled), C(q) of the task
    need_attrs: dict = field(default_factory=dict)
    need: str = ""                     # need text (filled by needs.py)

    @property
    def label(self) -> str:
        return C.LETTERS[self.cands.index(self.target)]


# ----------------------------------------------------------------------------- helpers
def price_bands(prices: list[float | None]) -> tuple[list[str], list[float]]:
    known = np.array([p for p in prices if p is not None and p > 0])
    qs = np.quantile(known, [0.25, 0.5, 0.75]) if len(known) else np.array([10, 25, 50])
    qs = [float(np.round(q)) for q in qs]
    out = []
    for p in prices:
        if p is None or p <= 0:
            out.append("price unknown")
        elif p < qs[0]:
            out.append(f"under ${qs[0]:.0f}")
        elif p < qs[1]:
            out.append(f"${qs[0]:.0f}-${qs[1]:.0f}")
        elif p < qs[2]:
            out.append(f"${qs[1]:.0f}-${qs[2]:.0f}")
        else:
            out.append(f"over ${qs[2]:.0f}")
    return out, qs


def _clean(s: str, n: int = 120) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:n]


def _parse_price(p) -> float | None:
    if p is None:
        return None
    try:
        return float(str(p).replace("$", "").replace(",", ""))
    except ValueError:
        return None


# ----------------------------------------------------------------------------- amazon
def _load_amazon(ds: str):
    cat = AMAZON[ds]
    base = C.RAW / "amazon" / "benchmark" / "5core" / "last_out_w_his"
    test = pd.read_csv(base / f"{cat}.test.csv")
    # full sequence = history (all earlier interactions, chronological) + held-out item
    seq_raw = {r.user_id: (str(r.history).split() if isinstance(r.history, str) else []) + [r.parent_asin]
               for r in test.itertuples()}
    asins = sorted({a for s in seq_raw.values() for a in s})
    meta = {}
    with open(C.RAW / "amazon" / "raw" / "meta_categories" / f"meta_{cat}.jsonl") as f:
        want = set(asins)
        for line in f:
            d = json.loads(line)
            if d["parent_asin"] in want:
                meta[d["parent_asin"]] = d
    prices = [_parse_price(meta.get(a, {}).get("price")) for a in asins]
    bands, _ = price_bands(prices)
    ids, titles, attrs, feats, leaf = ["<pad>"], ["<pad>"], [{}], [[]], [""]
    for a, band in zip(asins, bands):
        d = meta.get(a, {})
        cats = [c for c in (d.get("categories") or []) if c]
        brand = _clean(d.get("store") or (d.get("details") or {}).get("Manufacturer") or "unknown brand", 40)
        if ds == "games":
            platform = cats[1] if len(cats) > 1 else "Video Games"
            kind = cats[-1] if len(cats) > 2 else (cats[1] if len(cats) > 1 else "Games")
            at = {"platform": platform, "category": kind, "price": band, "brand": brand}
        else:
            kind = cats[-1] if cats else "Industrial & Scientific"
            at = {"category": kind, "price": band, "brand": brand}
        ids.append(a)
        titles.append(_clean(d.get("title") or a, 100))
        attrs.append(at)
        feats.append([_clean(x, 100) for x in (d.get("features") or [])[:4]])
        leaf.append(" > ".join(cats[1:]) if len(cats) > 1 else (cats[0] if cats else "misc"))
    idx = {a: i for i, a in enumerate(ids)}
    seqs = {u: [idx[a] for a in s] for u, s in seq_raw.items()}
    default = ("platform", "category", "price", "brand") if ds == "games" else ("category", "price", "brand")
    cat_obj = Catalog(ds, ids, titles, attrs, feats, leaf, default)
    return cat_obj, seqs


# ----------------------------------------------------------------------------- movielens
def _load_ml():
    base = C.RAW / "ml-1m"
    r = pd.read_csv(base / "ratings.dat", sep="::", engine="python", names=["u", "i", "r", "t"])
    m = pd.read_csv(base / "movies.dat", sep="::", engine="python", names=["i", "title", "genres"],
                    encoding="latin-1")
    # Yue et al. / BERT4Rec preprocessing: all ratings as implicit feedback, items with >= 5
    # ratings, users with >= 5 ratings.
    ic = r.i.value_counts()
    r = r[r.i.isin(ic[ic >= 5].index)]
    uc = r.u.value_counts()
    r = r[r.u.isin(uc[uc >= 5].index)]
    r = r.sort_values(["u", "t"], kind="mergesort")
    movies = sorted(r.i.unique())
    mm = m.set_index("i")
    ids, titles, attrs, feats, leaf = ["<pad>"], ["<pad>"], [{}], [[]], [""]
    for i in movies:
        t = mm.loc[i, "title"]
        g = mm.loc[i, "genres"].split("|")
        yr = re.search(r"\((\d{4})\)\s*$", t)
        dec = f"{yr.group(1)[:3]}0s" if yr else "unknown decade"
        ids.append(str(i))
        titles.append(t)
        attrs.append({"genres": ", ".join(g), "decade": dec})
        feats.append([])
        leaf.append(g[0])
    idx = {s: k for k, s in enumerate(ids)}
    seqs = {int(u): [idx[str(i)] for i in grp.i.tolist()] for u, grp in r.groupby("u")}
    return Catalog("ml", ids, titles, attrs, feats, leaf, ("genres", "decade")), seqs


# ----------------------------------------------------------------------------- splits/tasks
def _split(seqs: dict, ds: str, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    users = np.array(sorted(seqs, key=str), dtype=object)
    users = [u for u in users if len(seqs[u]) >= 5]
    rng.shuffle(users)
    n = len(users)
    nm = C.MIA_MEMBERS
    nonmembers = users[:nm]
    rest = users[nm:]
    shadow_n = min(2000, n // 3)
    shadow = rest[:shadow_n]
    pool = rest[shadow_n:]
    # The attacker's shadow users are disjoint from every MIA target. They may also be
    # victim members (the attacker only needs same-distribution data).
    members = pool + shadow
    test_users = pool[: C.TEST_USERS]
    val_users = pool[C.TEST_USERS: C.TEST_USERS + C.VAL_USERS]
    mia_members = pool[C.TEST_USERS + C.VAL_USERS: C.TEST_USERS + C.VAL_USERS + nm]
    session_users = pool[C.TEST_USERS + C.VAL_USERS + nm: C.TEST_USERS + C.VAL_USERS + nm + 1000]
    if len(session_users) < 500:  # ml: reuse validation users' prefixes as held-out sessions
        session_users = val_users[:1000]
    half = len(shadow) // 2
    # int-cast ids for ml, keep strings for amazon
    return Split({u: seqs[u] for u in users}, members, nonmembers, test_users, val_users,
                 shadow[:half], shadow[half:], mia_members, session_users)


def _band(a: dict) -> str:
    """Negative-sampling band: the price band (Amazon). ML-1M negatives share the first genre,
    and optionally also the release decade (RECTELLER_ML_NEG_DECADE=1)."""
    if "price" in a:
        return a["price"]
    return a.get("decade", "") if C.ML_NEG_DECADE else ""


def make_tasks(cat: Catalog, split: Split, users: list, seed: int, hist: int = C.DEFAULT.hist) -> list[Task]:
    """LLaRA-style pick-one tasks: target + 19 negatives of its leaf category and price band."""
    rng = np.random.default_rng(seed)
    by_leaf_band, by_leaf, by_top = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in range(1, cat.n + 1):
        band = _band(cat.attrs[i])
        by_leaf_band[(cat.leaf[i], band)].append(i)
        by_leaf[cat.leaf[i]].append(i)
        by_top[cat.leaf[i].split(" > ")[0]].append(i)
    tasks = []
    for u in users:
        s = split.seqs[u]
        target, h = s[-1], s[:-1][-hist:]
        excl = set(s)
        band = _band(cat.attrs[target])
        negs: list[int] = []
        for pool in (by_leaf_band[(cat.leaf[target], band)], by_leaf[cat.leaf[target]],
                     by_top[cat.leaf[target].split(" > ")[0]], range(1, cat.n + 1)):
            cand = [i for i in pool if i not in excl and i not in negs]
            need = C.DEFAULT.ncand - 1 - len(negs)
            if need <= 0:
                break
            if len(cand) > need:
                cand = list(rng.choice(cand, size=need, replace=False))
            negs += [int(i) for i in cand]
        cands = negs[: C.DEFAULT.ncand - 1] + [target]
        rng.shuffle(cands)
        a = cat.attrs[target]
        need_attrs = dict(a)
        if cat.features[target]:
            need_attrs["features"] = cat.features[target][:2]
        tasks.append(Task(u, h, int(target), [int(c) for c in cands], need_attrs))
    return tasks


@dataclass
class Dataset:
    cat: Catalog
    split: Split
    test: list[Task]
    val: list[Task]

    def train_seqs(self, users=None) -> list[list[int]]:
        """Victim training sequences: members' items without the validation and test items
        (leave-last-out, the second-last item selects the epoch by validation NDCG@10)."""
        users = self.split.members if users is None else users
        return [self.split.seqs[u][:-2] for u in users]

    def val_pairs(self, users=None, n: int = 3000, seed: int = 0):
        users = self.split.members if users is None else users
        rng = np.random.default_rng(seed)
        us = rng.choice(len(users), size=min(n, len(users)), replace=False)
        return [(self.split.seqs[users[k]][:-2], self.split.seqs[users[k]][-2]) for k in us]


def load(ds: str, rebuild: bool = False) -> Dataset:
    C.ensure_dirs()
    path = C.PROC / f"{ds}.pkl"
    if path.exists() and not rebuild:
        with open(path, "rb") as f:
            return pickle.load(f)
    cat, seqs = _load_ml() if ds == "ml" else _load_amazon(ds)
    split = _split(seqs, ds)
    test = make_tasks(cat, split, split.test_users, seed=1)
    val = make_tasks(cat, split, split.val_users, seed=2)
    d = Dataset(cat, split, test, val)
    with open(path, "wb") as f:
        pickle.dump(d, f)
    return d


def save(d: Dataset) -> None:
    with open(C.PROC / f"{d.cat.name}.pkl", "wb") as f:
        pickle.dump(d, f)


def stats(d: Dataset) -> dict:
    n_int = sum(len(s) for s in d.split.seqs.values())
    return {"users": len(d.split.seqs), "items": d.cat.n, "interactions": n_int,
            "avg_len": n_int / len(d.split.seqs), "members": len(d.split.members),
            "test_tasks": len(d.test), "val_tasks": len(d.val)}


def make_synth_tasks(cat: Catalog, n: int, exclude: set[int], seed: int = 3) -> list[Task]:
    """Public synthetic tasks for projector training (App. D "Projector training"): a public
    catalog target (never a test/validation target), candidates as in ``make_tasks``."""
    rng = np.random.default_rng(seed)
    pool = [i for i in range(1, cat.n + 1) if i not in exclude]
    targets = rng.choice(pool, size=n, replace=len(pool) < n)
    fake = Split({k: [int(t)] for k, t in enumerate(targets)}, [], [], [], [], [], [], [], [])
    tasks = make_tasks(cat, fake, list(range(n)), seed=seed, hist=0)
    for t in tasks:
        t.uid, t.history = -1, []
    return tasks
