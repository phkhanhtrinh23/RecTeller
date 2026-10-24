"""Per-receiver projector g_psi (Sec. 4.3, Eq. proj) and its public training simulator.

g_psi is a two-layer MLP from a slot z_k = [e(o_k) || attribute block || phi(level)] to one
soft token in the frozen receiver's input-embedding space. phi is a learned level table
(d_phi = 64). Training uses public synthetic tasks only (H2): proxy scores are encoder
similarity of need and item text plus Gaussian noise, then the deployed release (Gumbel
top-k_p, Laplace levels, public fill) at eps_resp = 4. The loss is -log p_LLM(y | prompt)
with y the target identifier, copied from the input.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .encoder import PublicEmbeddings, embed, make_slots
from .release import release


class Projector(nn.Module):
    def __init__(self, d: int = 768, d_phi: int = 64, hidden: int = 4096, out: int = 4096, L: int = 4,
                 target_norm: float = 1.0):
        super().__init__()
        self.phi = nn.Embedding(L + 1, d_phi)
        self.mlp = nn.Sequential(nn.Linear(2 * d + d_phi, hidden), nn.GELU(), nn.Linear(hidden, out))
        self.register_buffer("target_norm", torch.tensor(float(target_norm)))
        self.cfg = dict(d=d, d_phi=d_phi, hidden=hidden, out=out, L=L, target_norm=float(target_norm))

    def forward(self, item_emb: torch.Tensor, attr_emb: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        z = torch.cat([item_emb, attr_emb, self.phi(levels)], -1)
        y = self.mlp(z)
        # match the scale of the receiver's token embeddings
        return y / (y.norm(dim=-1, keepdim=True) + 1e-6) * self.target_norm

    def soft(self, slots, device="cuda") -> torch.Tensor:
        p = next(self.parameters())
        ie = torch.as_tensor(slots.item_emb, device=device, dtype=p.dtype)
        ae = torch.as_tensor(slots.attr_emb, device=device, dtype=p.dtype)
        lv = torch.as_tensor(slots.levels, device=device, dtype=torch.long)
        return self(ie, ae, lv)


def save(proj: Projector, path) -> None:
    torch.save({"cfg": proj.cfg, "state": proj.state_dict()}, path)


def load(path, device="cuda") -> Projector:
    ck = torch.load(path, map_location=device, weights_only=False)
    p = Projector(**ck["cfg"]).to(device)   # level_bias is restored from the state dict
    p.load_state_dict(ck["state"])
    return p.eval()


# ------------------------------------------------------------------ simulator (fixed a priori)
# Simulator configuration, fixed before any evaluation run.
SIM = {"scale": 10.0, "sigma": 2.0, "eps_resp": 4.0, "anchor": C.PROJ_ANCHOR}


def simulate_release(pe: PublicEmbeddings, task, need_vec: np.ndarray, proto, rng):
    """Proxy scores (public only) -> deployed release. Returns (Release, public order).

    Proxy scores are the public encoder similarity between the need text and each candidate
    plus Gaussian noise (App. D). With anchor "target", the similarity to the synthetic target
    item (a public catalog item) is used instead. The public fill is ordered by similarity to
    the need text."""
    cands = task.cands
    need_sim = pe.item[cands] @ need_vec
    public_order = [cands[j] for j in np.argsort(-need_sim, kind="stable")]
    sim = pe.item[cands] @ pe.item[task.target] if SIM["anchor"] == "target" else need_sim
    s = SIM["scale"] * sim + rng.normal(0, SIM["sigma"], size=len(cands))
    s = s - s.mean()
    rel = release(s, cands, public_order, proto.with_eps(SIM["eps_resp"]), rng, kind="noisy")
    return rel, public_order


def train(llm, pe: PublicEmbeddings, tasks, prompt_fn, proto, epochs: int = 1, bs: int = 8, lr: float = 1e-4,
          seed: int = 0, log_every: int = 50, max_steps: int | None = None) -> Projector:
    """prompt_fn(task, release) -> templated prompt with soft markers (same as evaluation)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        tn = llm.emb.weight.float().norm(dim=-1).mean().item()
    proj = Projector(pe.d, 64, 4096, llm.hidden, proto.L, tn).to(llm.device)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=0.0)
    need_vecs = embed([t.need for t in tasks])
    llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.model.train()
    steps_total = epochs * math.ceil(len(tasks) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_total, pct_start=0.05)
    step, t0, run = 0, time.time(), []
    for ep in range(epochs):
        order = rng.permutation(len(tasks))
        for b in range(0, len(order), bs):
            items, ys = [], []
            for i in order[b:b + bs]:
                t = tasks[i]
                rel, _ = simulate_release(pe, t, need_vecs[i], proto, rng)
                slots = make_slots(pe, rel.items, rel.levels)
                ids, pos = llm.encode_segments(prompt_fn(t, rel))
                soft = proj.soft(slots, llm.device)
                items.append((ids, pos, soft))
                ys.append(llm.letter_ids[t.cands.index(t.target)])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = llm.last_logits(items, grad=True)
            loss = nn.functional.cross_entropy(lg, torch.tensor(ys, device=llm.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run.append(loss.item())
            if step % log_every == 0:
                print(f"  proj step {step}/{steps_total} loss {np.mean(run[-log_every:]):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if max_steps and step >= max_steps:
                break
    llm.model.gradient_checkpointing_disable()
    llm.model.eval()
    return proj.eval()
