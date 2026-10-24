"""SASRec (Kang & McAuley, 2018): the platform's private scorer f_theta, the attacker's
surrogate and the MIA shadow model.

``f_theta(h, i)`` is the logit of item i after context h (dot product of the last hidden
state with the item embedding).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SASRecConfig:
    n_items: int
    maxlen: int = 50
    d: int = 64
    blocks: int = 2
    heads: int = 1
    dropout: float = 0.3


class SASRec(nn.Module):
    def __init__(self, cfg: SASRecConfig):
        super().__init__()
        self.cfg = cfg
        self.item = nn.Embedding(cfg.n_items + 1, cfg.d, padding_idx=0)
        self.pos = nn.Embedding(cfg.maxlen, cfg.d)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(cfg.d, cfg.heads, cfg.d, cfg.dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, cfg.blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d)
        nn.init.normal_(self.item.weight, std=0.05)

    def hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T) left-padded item ids -> (B, T, d)."""
        T = seq.size(1)
        x = self.item(seq) * math.sqrt(self.cfg.d) + self.pos(torch.arange(T, device=seq.device))
        x = self.drop(x)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=seq.device), 1)
        pad = seq == 0
        # keep at least one attendable key per row (all-pad rows produce NaNs otherwise)
        pad = pad & ~(pad.all(1, keepdim=True) & (torch.arange(T, device=seq.device) == T - 1))
        x = self.enc(x, mask=causal, src_key_padding_mask=pad)
        return self.norm(x)

    def last(self, seq: torch.Tensor) -> torch.Tensor:
        return self.hidden(seq)[:, -1]

    def logits_all(self, seq: torch.Tensor) -> torch.Tensor:
        return self.last(seq) @ self.item.weight[1:].T  # (B, n_items) for items 1..n

    def logits_cands(self, seq: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,bkd->bk", self.last(seq), self.item(cands))


def pad_seqs(seqs: list[list[int]], maxlen: int) -> torch.Tensor:
    out = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for k, s in enumerate(seqs):
        s = s[-maxlen:]
        if s:
            out[k, -len(s):] = s
    return torch.from_numpy(out)


class Scorer:
    """Batched wrapper around a frozen SASRec: the platform's f_theta."""

    def __init__(self, model: SASRec):
        self.model = model.eval().to(DEV)
        self.maxlen = model.cfg.maxlen
        self.n_items = model.cfg.n_items

    @torch.no_grad()
    def scores(self, histories: list[list[int]], cands: list[list[int]], bs: int = 2048) -> list[np.ndarray]:
        """Raw logits f_theta(h, i) for each (history, candidate list)."""
        out: list[np.ndarray] = []
        for b in range(0, len(histories), bs):
            H = pad_seqs(histories[b:b + bs], self.maxlen).to(DEV)
            cs = cands[b:b + bs]
            m = max(len(c) for c in cs)
            Cm = torch.zeros(len(cs), m, dtype=torch.long)
            for k, c in enumerate(cs):
                Cm[k, :len(c)] = torch.tensor(c)
            lg = self.model.logits_cands(H, Cm.to(DEV)).float().cpu().numpy()
            out += [lg[k, :len(c)] for k, c in enumerate(cs)]
        return out

    @torch.no_grad()
    def topk_all(self, histories: list[list[int]], k: int = 10, bs: int = 2048) -> np.ndarray:
        """Top-k item ids over the whole catalog (for Agr@10)."""
        out = []
        for b in range(0, len(histories), bs):
            H = pad_seqs(histories[b:b + bs], self.maxlen).to(DEV)
            out.append((self.model.logits_all(H).topk(k, dim=1).indices + 1).cpu().numpy())
        return np.concatenate(out)


# ----------------------------------------------------------------------------- training
def _batches(seqs, maxlen, bs, rng):
    idx = rng.permutation(len(seqs))
    for b in range(0, len(idx), bs):
        chunk = []
        for i in idx[b:b + bs]:
            s = seqs[i]
            # random crops of long sequences (ML-1M) augment the last-window view
            end = len(s) if len(s) <= maxlen + 1 or rng.random() < 0.5 else int(rng.integers(maxlen + 1, len(s) + 1))
            chunk.append(s[max(0, end - maxlen - 1):end])
        inp = pad_seqs([s[:-1] for s in chunk], maxlen)
        tgt = pad_seqs([s[1:] for s in chunk], maxlen)
        yield inp, tgt


@torch.no_grad()
def evaluate(model: SASRec, pairs, k: int = 10, bs: int = 2048) -> dict:
    """Full-catalog NDCG@k / HR@k for (context, next item) pairs."""
    model.eval()
    hits, ndcg = [], []
    for b in range(0, len(pairs), bs):
        ch = pairs[b:b + bs]
        H = pad_seqs([c for c, _ in ch], model.cfg.maxlen).to(DEV)
        tgt = torch.tensor([t for _, t in ch], device=DEV)
        lg = model.logits_all(H)
        tl = lg.gather(1, (tgt - 1)[:, None])
        rank = (lg > tl).sum(1)
        hits.append((rank < k).float().cpu())
        ndcg.append(torch.where(rank < k, 1 / torch.log2(rank.float() + 2), torch.zeros_like(rank.float())).cpu())
    return {"hr@10": float(torch.cat(hits).mean()), "ndcg@10": float(torch.cat(ndcg).mean())}


def train(seqs: list[list[int]], n_items: int, val_pairs=None, epochs: int = 200, bs: int = 256, lr: float = 1e-3,
          cfg: SASRecConfig | None = None, patience: int = 10, seed: int = 0, verbose: bool = True,
          eval_every: int = 2) -> SASRec:
    """Next-item training with full softmax cross-entropy, best epoch by validation NDCG@10."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = cfg or SASRecConfig(n_items)
    model = SASRec(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    seqs = [s for s in seqs if len(s) >= 2]
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for inp, tgt in _batches(seqs, cfg.maxlen, bs, rng):
            inp, tgt = inp.to(DEV), tgt.to(DEV)
            h = model.hidden(inp)
            m = tgt > 0
            lg = h[m] @ model.item.weight[1:].T
            loss = F.cross_entropy(lg, tgt[m] - 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if val_pairs is not None and (ep + 1) % eval_every == 0:
            r = evaluate(model, val_pairs)
            if verbose:
                print(f"  ep {ep + 1:3d} loss {tot:.1f} val {r}", flush=True)
            if r["ndcg@10"] > best:
                best, best_state, bad = r["ndcg@10"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def save(model: SASRec, path) -> None:
    torch.save({"cfg": model.cfg.__dict__, "state": model.state_dict()}, path)


def load(path) -> SASRec:
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = SASRec(SASRecConfig(**ck["cfg"])).to(DEV)
    m.load_state_dict(ck["state"])
    return m.eval()
