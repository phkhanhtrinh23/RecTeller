"""Fig. budget (a): utility under the per-account cap. Test tasks are grouped at random into
account-days of 12 tasks, served one after another to one fresh account within one window.
The day mean at n averages the first n tasks. MCP tools has no per-account bound, so its curve
is flat at its per-task HR@1 (read from the utility runs). capo<n> is the cap-only k_p = 3 control.

    python scripts/run_budget.py --dataset games
"""
import argparse
import json

import numpy as np
import torch

import _path  # noqa: F401
from recteller import clients, config as C, projector, setup
from recteller.encoder import make_slots
from recteller.evaluate import section
from recteller.llm import LLM
from recteller.modes import serve_account_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="games")
    ap.add_argument("--n", type=int, default=C.TEST_USERS)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--modes", nargs="+", default=["ours", "capo2", "capo4"],
                    help="capo<n>: cap-only k_p = 3 control with n_max = n")
    args = ap.parse_args()
    ctx, d = setup.context(args.dataset)
    rng = np.random.default_rng(0)
    tasks = [d.test[i] for i in rng.permutation(min(args.n, len(d.test)))]
    tasks = tasks[: len(tasks) // C.SEQ_N * C.SEQ_N]
    llm = LLM()
    proj = projector.load(C.CKPT / f"projector_{args.dataset}_Qwen3-8B.pt")
    res = {}
    readers = {}
    for mode in args.modes:
        reader = readers.get(mode, "soft")
        corr = []
        for s in range(args.seeds):
            resps = serve_account_days(mode, tasks, ctx, C.SEQ_N, seed=s)
            prompts, softs = [], []
            for t, r in zip(tasks, resps):
                prompts.append(llm.chat(clients.SYSTEM, clients.user_text(ctx.cat, t, section(reader, ctx.cat, t, r))))
                if reader == "soft":
                    with torch.no_grad():
                        softs.append(proj.soft(make_slots(ctx.pe, r.items, r.levels, attrs=r.attrs)).to(torch.bfloat16))
                else:
                    softs.append(None)
            pick = llm.choice_logprobs(prompts, softs).argmax(1)
            corr.append([C.LETTERS[p] == t.label for p, t in zip(pick, tasks)])
            pers = [r.n_pers > 0 for r in resps]
        c = np.array(corr, dtype=np.float32).mean(0).reshape(-1, C.SEQ_N)       # (days, position)
        res[mode] = {"by_position": c.mean(0).tolist(),
                     "day_mean": [float(c[:, :n].mean()) for n in range(1, C.SEQ_N + 1)],
                     "personalized_by_position": np.array(pers).reshape(-1, C.SEQ_N).mean(0).tolist()}
        print(mode, res[mode], flush=True)
    out = C.RESULTS / "budget"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{args.dataset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
