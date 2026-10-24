"""Protocol constants and paths.

Defaults follow the experimental setup of the paper (Sec. 5.1). Change them through
``Protocol(...)`` overrides.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Large artefacts (raw data, checkpoints, caches) live outside the repository by default.
WORK = Path(os.environ.get("RECTELLER_WORK", Path.home() / "recteller_work"))
RAW = WORK / "raw"
PROC = WORK / "processed"
CKPT = WORK / "checkpoints"
CACHE = WORK / "cache"
RESULTS = Path(os.environ.get("RECTELLER_RESULTS", ROOT / "results"))

HF_HOME = os.environ.get("HF_HOME", None)

# Models (Hugging Face names). Override with the environment variables below.
CLIENT_MODEL = os.environ.get("RECTELLER_CLIENT", "Qwen/Qwen3-8B")
PARSER_MODEL = os.environ.get("RECTELLER_PARSER", "Qwen/Qwen3-8B")
ENCODER_MODEL = os.environ.get("RECTELLER_ENCODER", "sentence-transformers/all-mpnet-base-v2")  # d = 768
CLOSED_MODEL = os.environ.get("RECTELLER_CLOSED", "gpt-4o-mini")

DATASETS = ("games", "ind", "ml")

# Task and simulator protocol (App. C.1, App. D "Projector training"). Defaults follow the paper.
NEED_FEATURES = os.environ.get("RECTELLER_NEED_FEATURES", "1") == "1"   # need text uses two item features
ML_NEG_DECADE = os.environ.get("RECTELLER_ML_NEG_DECADE", "0") == "1"   # ML-1M negatives also share the decade
PROJ_ANCHOR = os.environ.get("RECTELLER_PROJ_ANCHOR", "need")           # proxy scores: similarity to the need or target


@dataclass(frozen=True)
class Protocol:
    # release (Sec. 4.2)
    K: int = 10                 # K, slots per response
    kp: int = 3                 # k_p, personalized slots
    L: int = 4                  # L, weight levels
    clip: float = 5.0           # c, score clipping bound
    Delta: float = 1.0          # Delta, score sensitivity
    eps_sel: float = 3.0        # eps_sel, selection budget
    eps_w: float = 1.0          # eps_w, weight-level budget
    # per-account budgets per window (Sec. 4.1)
    n_max: int = 8              # n_max, personalized responses per window
    B: int = 40                 # B, admitted requests per window
    P: int = 200                # P, parser calls per window
    window_hours: int = 24      # window length (hours)
    cap30: int | None = None    # optional sliding 30-day cap (64 in the paper), off by default
    # tasks
    ncand: int = 20             # candidates per pick-one task
    hist: int = 50              # SASRec maximum history length
    search_size: int = 100      # |C(q)| of a free Search (public retrieval depth)

    @property
    def eps_resp(self) -> float:
        return self.eps_sel + self.eps_w

    @property
    def eps_W(self) -> float:
        return self.n_max * self.eps_resp

    def with_eps(self, eps_resp: float, w_share: float = 0.25) -> "Protocol":
        """Scale (eps_sel, eps_w) keeping the default 3:1 split (eps_w = eps_resp/4)."""
        if eps_resp == float("inf"):
            return replace(self, eps_sel=float("inf"), eps_w=float("inf"))
        return replace(self, eps_sel=eps_resp * (1 - w_share), eps_w=eps_resp * w_share)


DEFAULT = Protocol()

# Experiment protocol constants
TEST_USERS = 2000          # test users per dataset
VAL_USERS = 2000           # validation users (Games)
MIA_MEMBERS = 500          # MIA members (and non-members)
ATTACK_QUERIES = 5000      # synthetic extraction sequences
ATTACK_LEN = 20            # steps per extraction sequence
ATTACK_DAYS = 7            # attack horizon (windows)
SYBIL = 50                 # colluding accounts
ATTACK_SEEDS = 5           # attack seeds
FLOOD_N = 1000             # flood requests
FLOOD_SESSIONS = 5         # parallel flood sessions
FLOOD_REPEAT = 0.05        # share of repeated flood requests
REP_R = 8                  # repeats per context (repetition attack)
SEQ_N = 12                 # tasks per account-day
LETTERS = [chr(ord("A") + i) for i in range(20)]  # task-local candidate identifiers


def ensure_dirs() -> None:
    for p in (RAW, PROC, CKPT, CACHE, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
