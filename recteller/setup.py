"""Shared construction of the platform context per dataset."""
from __future__ import annotations

from . import config as C, data, sasrec
from .encoder import PublicEmbeddings
from .modes import Context
from .retrieval import Retrieval


def context(ds: str, proto=None, gro: bool = False, victim: str = "victim") -> tuple[Context, data.Dataset]:
    d = data.load(ds)
    pe = PublicEmbeddings(d.cat)
    ret = Retrieval(pe, size=(proto or C.DEFAULT).search_size)
    scorer = sasrec.Scorer(sasrec.load(C.CKPT / f"sasrec_{ds}_{victim}.pt"))
    gro_scorer = None
    gp = C.CKPT / f"sasrec_{ds}_gro.pt"
    if gro and gp.exists():
        gro_scorer = sasrec.Scorer(sasrec.load(gp))
    return Context(d.cat, pe, ret, scorer, proto or C.DEFAULT, gro_scorer), d


def sessions(d: data.Dataset, n: int = 1000) -> list[list[int]]:
    """Held-out real sessions for Agr@10 (users never targeted by the attacks)."""
    return [d.split.seqs[u][:-1][-50:] for u in d.split.session_users[:n]]
