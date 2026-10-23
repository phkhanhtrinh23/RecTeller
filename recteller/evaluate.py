"""Utility evaluation (RQ1/RQ3/RQ4): serve each pick-one task through an access mode, build the
client prompt for the mode's reader, and score HR@1 and client input tokens."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch

from . import clients, config as C
from .encoder import make_slots
from .llm import LLM
from .modes import serve_tasks

# name -> (service mode, reader, options)
SPECS = {
    "block": ("block", "block", {}),
    "web": ("web", "web", {}),
    "mcp": ("mcp", "mcp", {}),
    "firewall": ("firewall", "firewall", {}),
    "iagent": ("iagent", "iagent", {}),
    "openapi": ("openapi", "openapi", {}),
    "gro": ("gro", "mcp", {}),
    "jtopk": ("jtopk", "jtopk", {}),
    "capped": ("capped", "jtopk", {}),
    "stitch": ("stitch", "jtopk", {}),
    "ours": ("ours", "soft", {}),
    "ours_text": ("ours", "text", {}),
    "einf": ("einf", "soft", {}),
    "einf_text": ("einf", "text", {}),
    "pub": ("pub", "soft", {}),
    "nolev": ("nolev", "soft", {}),
    "kp1": ("kp1", "soft", {}),
    "kp5": ("kp5", "soft", {}),
    "ours_shuf": ("ours", "soft", {"shuffle": True}),        # A3
    "ours_xproj": ("ours", "soft", {"proj": "other"}),       # A5
}
for e in (0.5, 1.0, 2.0, 8.0, 16.0):
    SPECS[f"ours@{e:g}"] = ("ours", "soft", {"eps": e})
    SPECS[f"ours_text@{e:g}"] = ("ours", "text", {"eps": e})
for p in (0.1, 0.3, 0.5):
    SPECS[f"rr{p:g}"] = (f"rr{p:g}", "jtopk", {})

DETERMINISTIC = {"block", "web", "mcp", "firewall", "iagent", "openapi", "gro", "jtopk", "capped", "stitch",
                 "einf", "einf_text", "pub"} | {f"rr{p:g}" for p in (0.1, 0.3, 0.5)}


def section(reader: str, cat, task, resp) -> str:
    if reader == "soft":
        return clients.sec_ours_soft(cat, task, resp)
    if reader == "text":
        return clients.sec_ours_text(cat, task, resp)
    if reader == "iagent":
        return clients.sec_iagent(cat, task, resp, task.history)
    if reader == "firewall":
        return clients.sec_mcp(cat, task, resp) if resp.items else clients.sec_block(cat, task)
    return clients.SECTIONS[reader](cat, task, resp)


def build_prompts(name, tasks, ctx, seed=0, projector=None, llm: LLM | None = None, firewall_ok=None):
    svc, reader, opt = SPECS[name]
    proto = ctx.proto.with_eps(opt["eps"]) if "eps" in opt else None
    resps = serve_tasks(svc, tasks, ctx, seed=seed, proto=proto)
    if firewall_ok is not None:      # converter drops -> no recommender output
        for r, ok in zip(resps, firewall_ok):
            if not ok:
                r.items = []
    users, softs = [], []
    rng = np.random.default_rng(seed + 99)
    for t, r in zip(tasks, resps):
        users.append(clients.user_text(ctx.cat, t, section(reader, ctx.cat, t, r)))
        if reader == "soft":
            sl = make_slots(ctx.pe, r.items, r.levels, attrs=r.attrs,
                            shuffle_attrs_rng=rng if opt.get("shuffle") else None)
            with torch.no_grad():
                softs.append(projector.soft(sl).to(torch.bfloat16))
        else:
            softs.append(None)
    return users, softs, resps


def run_open(name, tasks, ctx, llm: LLM, seeds=(0,), projector=None, firewall_ok=None) -> dict:
    """Open-weight client (Qwen3-8B): greedy pick over candidate letters, seeds redraw the release."""
    seeds = (seeds[0],) if name in DETERMINISTIC else seeds
    correct, toks = [], None
    for s in seeds:
        users, softs, _ = build_prompts(name, tasks, ctx, s, projector, llm, firewall_ok)
        prompts = [llm.chat(clients.SYSTEM, u) for u in users]
        if toks is None:
            toks = [llm.n_tokens(p) for p in prompts]
        lp = llm.choice_logprobs(prompts, softs)
        pick = lp.argmax(1)
        correct.append([C.LETTERS[p] == t.label for p, t in zip(pick, tasks)])
    correct = np.array(correct, dtype=np.float32)            # (seeds, tasks)
    return {"mode": name, "hr1": float(correct.mean()), "hr1_per_seed": correct.mean(1).tolist(),
            "per_task": correct.mean(0).tolist(), "tokens": float(np.mean(toks)), "seeds": len(seeds)}


def run_closed(name, tasks, ctx, client, firewall_ok=None) -> dict:
    svc, reader, opt = SPECS[name]
    if reader == "soft":
        raise ValueError("closed clients read text")
    users, _, _ = build_prompts(name, tasks, ctx, 0, None, None, firewall_ok)
    ans = client.choose(clients.SYSTEM, users)
    correct = np.array([a == t.label for a, t in zip(ans, tasks)], dtype=np.float32)
    toks = [client.n_tokens(clients.SYSTEM, u) for u in users]
    return {"mode": name, "hr1": float(correct.mean()), "per_task": correct.tolist(), "tokens": float(np.mean(toks)),
            "invalid": int(sum(a not in C.LETTERS for a in ans))}


def bootstrap_diff(a, b, n=2000, seed=0) -> tuple[float, float]:
    """95% paired bootstrap over tasks of mean(a) - mean(b): (diff, half-width)."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(a.mean() - b.mean()), float((hi - lo) / 2)


def bootstrap_ci(a, n=2000, seed=0) -> float:
    a = np.asarray(a)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), size=(n, len(a)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float((hi - lo) / 2)
