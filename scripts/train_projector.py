"""Train the projector g_psi for one frozen receiver on public synthetic tasks (Eq. proj),
then check it on held-out synthetic tasks against the compact-text reader.

    python scripts/train_projector.py --dataset games [--receiver Qwen/Qwen3-8B]
"""
import argparse
import json
import pickle
from types import SimpleNamespace

import numpy as np
import torch

import _path  # noqa: F401
from recteller import clients, config as C, data, projector
from recteller.encoder import PublicEmbeddings, embed, make_slots
from recteller.llm import LLM


def tag(name):
    return name.split("/")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--receiver", default=C.CLIENT_MODEL)
    ap.add_argument("--n", type=int, default=7500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()
    if (C.CKPT / f"projector_{args.dataset}_{tag(args.receiver)}.pt").exists() and \
            (C.RESULTS / "projector" / f"{args.dataset}_{tag(args.receiver)}.json").exists():
        print("projector exists, skip")
        return
    d = data.load(args.dataset)
    syn = pickle.load(open(C.PROC / f"{args.dataset}_synth.pkl", "rb"))
    train, held = syn[: args.n], syn[-500:]
    pe = PublicEmbeddings(d.cat)
    llm = LLM(args.receiver)
    cat = d.cat
    default_attrs = tuple(cat.default_attrs)

    def resp_of(rel):
        return SimpleNamespace(items=rel.items, levels=rel.levels, attrs=default_attrs)

    def prompt_fn(task, rel):
        return llm.chat(clients.SYSTEM, clients.user_text(cat, task, clients.sec_ours_soft(cat, task, resp_of(rel))))

    proj = projector.train(llm, pe, train, prompt_fn, C.DEFAULT, epochs=args.epochs, bs=args.bs, lr=args.lr,
                           max_steps=args.max_steps)
    out = C.CKPT / f"projector_{args.dataset}_{tag(args.receiver)}{args.out_tag}.pt"
    projector.save(proj, out)

    # held-out synthetic check: soft reader vs compact text reader on the same releases
    rng = np.random.default_rng(123)
    nv = embed([t.need for t in held])
    sp, tp, softs = [], [], []
    for t, v in zip(held, nv):
        rel, _ = projector.simulate_release(pe, t, v, C.DEFAULT, rng)
        r = resp_of(rel)
        sp.append(llm.chat(clients.SYSTEM, clients.user_text(cat, t, clients.sec_ours_soft(cat, t, r))))
        tp.append(llm.chat(clients.SYSTEM, clients.user_text(cat, t, clients.sec_ours_text(cat, t, r))))
        with torch.no_grad():
            softs.append(proj.soft(make_slots(pe, rel.items, rel.levels)).to(torch.bfloat16))
    labels = [t.label for t in held]
    acc_s = float(np.mean([C.LETTERS[i] == y for i, y in zip(llm.choice_logprobs(sp, softs).argmax(1), labels)]))
    acc_t = float(np.mean([C.LETTERS[i] == y for i, y in zip(llm.choice_logprobs(tp).argmax(1), labels)]))
    res = {"held_soft_acc": acc_s, "held_text_acc": acc_t, "n_train": len(train), "sim": projector.SIM,
           "receiver": args.receiver}
    print(res)
    (C.RESULTS / "projector").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(C.RESULTS / "projector" / f"{args.dataset}_{tag(args.receiver)}{args.out_tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
