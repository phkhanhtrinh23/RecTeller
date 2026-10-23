"""Frozen open-weight LLM (Qwen3-8B by default) through Hugging Face transformers.

Prompts are lists of segments: ``str`` (text) or ``SOFT`` markers, which are filled with
soft-token vectors (projector outputs, one per slot) in the receiver's input-embedding space.
The client answers a pick-one task with one candidate letter. We read the next-token
distribution restricted to the candidate letters (one forward pass per task, "seeds" sample
from this distribution at temperature T, greedy = argmax).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C

SOFT = object()   # placeholder segment for one soft token


class LLM:
    def __init__(self, name: str = C.CLIENT_MODEL, dtype=torch.bfloat16, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device,
                                                          attn_implementation="sdpa")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.emb = self.model.get_input_embeddings()
        self.hidden = self.emb.weight.shape[1]
        self.letter_ids = [self._single_id(l) for l in C.LETTERS]

    def _single_id(self, s: str) -> int:
        ids = self.tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, (s, ids)
        return ids[0]

    # ------------------------------------------------------------------ prompt building
    def chat(self, system: str, user: str, thinking: bool = False) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {"enable_thinking": thinking} if "qwen3" in self.name.lower() else {}
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    SOFT_MARK = "⁣<soft>⁣"   # marker text inside the user message

    def encode_segments(self, templated: str) -> tuple[list[int], list[int]]:
        """Split a templated prompt at soft markers -> token ids and soft positions."""
        parts = templated.split(self.SOFT_MARK)
        ids, pos = [], []
        for k, p in enumerate(parts):
            ids += self.tok.encode(p, add_special_tokens=False)
            if k < len(parts) - 1:
                pos.append(len(ids))
                ids.append(self.tok.pad_token_id)       # placeholder id, replaced by the soft vector
        return ids, pos

    def n_tokens(self, templated: str) -> int:
        ids, pos = self.encode_segments(templated)
        return len(ids)

    def _batch_embeds(self, items: list[tuple[list[int], list[int], torch.Tensor | None]]):
        T = max(len(i[0]) for i in items)
        B = len(items)
        ids = torch.full((B, T), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((B, T), dtype=torch.long)
        for b, (x, _, _) in enumerate(items):
            ids[b, T - len(x):] = torch.tensor(x)
            att[b, T - len(x):] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        emb = self.emb(ids)
        if any(i[2] is not None and len(i[1]) for i in items):
            emb = emb.clone()
            for b, (x, pos, soft) in enumerate(items):
                if soft is None or not pos:
                    continue
                off = T - len(x)
                p = torch.tensor(pos, device=self.device) + off
                emb[b, p] = soft.to(emb.dtype)
        return emb, att

    def last_logits(self, items, grad: bool = False) -> torch.Tensor:
        emb, att = self._batch_embeds(items)
        pos_ids = (att.cumsum(1) - 1).clamp(min=0)
        with torch.set_grad_enabled(grad):
            out = self.model(inputs_embeds=emb, attention_mask=att, position_ids=pos_ids, logits_to_keep=1,
                             use_cache=False)
        return out.logits[:, -1].float()

    # ------------------------------------------------------------------ pick-one choice
    @torch.no_grad()
    def choice_logprobs(self, prompts: list[str], softs: list[torch.Tensor | None] | None = None,
                        n_letters: int = 20, max_tokens: int = 24000) -> np.ndarray:
        """Log-probabilities over the first ``n_letters`` candidate letters, (N, n_letters)."""
        enc = [self.encode_segments(p) for p in prompts]
        softs = softs or [None] * len(prompts)
        order = np.argsort([-len(e[0]) for e in enc])
        out = np.zeros((len(prompts), n_letters), dtype=np.float32)
        lid = torch.tensor(self.letter_ids[:n_letters], device=self.device)
        b = 0
        while b < len(order):
            T = len(enc[order[b]][0])
            bs = max(1, min(64, max_tokens // max(T, 1)))
            idx = order[b:b + bs]
            items = [(enc[i][0], enc[i][1], softs[i]) for i in idx]
            lg = self.last_logits(items)
            lp = torch.log_softmax(lg[:, lid], dim=1).cpu().numpy()
            out[idx] = lp
            b += bs
        return out

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 64, bs: int = 48, temperature: float = 0.0) -> list[str]:
        outs = [""] * len(prompts)
        order = np.argsort([-len(p) for p in prompts])
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            kw = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature > 0:
                kw.update(temperature=temperature, top_p=1.0)
            else:
                kw.update(temperature=None, top_p=None, top_k=None)
            g = self.model.generate(**enc, **kw)
            txt = self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, txt):
                outs[i] = t.strip()
        return outs


def sample_choices(logp: np.ndarray, seeds: int, temperature: float = 0.7, seed: int = 0) -> np.ndarray:
    """(N, seeds) sampled choices, seed 0 is greedy when temperature == 0."""
    rng = np.random.default_rng(seed)
    if temperature == 0:
        return np.repeat(logp.argmax(1)[:, None], seeds, 1)
    p = np.exp((logp - logp.max(1, keepdims=True)) / temperature)
    p /= p.sum(1, keepdims=True)
    return np.stack([[rng.choice(p.shape[1], p=pi) for _ in range(seeds)] for pi in p])
