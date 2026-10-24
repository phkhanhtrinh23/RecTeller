"""RQ1 / RQ3 / RQ4 utility runs: HR@1 and client input tokens per access mode.

    python scripts/run_utility.py --dataset games --client qwen --modes block web mcp ... --seeds 3
    python scripts/run_utility.py --dataset games --client gpt --modes block mcp gro ours_text
"""
import argparse
import json

import _path  # noqa: F401
from recteller import config as C, evaluate, projector, setup
from recteller.intents import Personalize, Search


def firewall_converter(ctx, tasks, llm, path):
    """Firewall's LLM converter maps the client's two requests onto the closed protocol. The
    deterministic verifier accepts only grammar-valid SEARCH then PERSONALIZE calls."""
    if path.exists():
        ok = json.load(open(path))
        if len(ok) >= len(tasks):
            return ok[: len(tasks)]
    from recteller.parser import LLMParser
    p = LLMParser(llm, ctx.cat.n)
    a = p.parse_many([f"Search among items {list(t.cands)} for: {t.need}" for t in tasks])
    b = p.parse_many(["Rank the search results for me using my history." for _ in tasks],
                     me=["me"] * len(tasks))
    ok = [isinstance(x, Search) and isinstance(y, Personalize) for x, y in zip(a, b)]
    json.dump(ok, open(path, "w"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--client", default="qwen", choices=["qwen", "gpt"])
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=C.TEST_USERS)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--receiver", default=C.CLIENT_MODEL, help="open-weight client (A5 control: Llama-3.1-8B)")
    args = ap.parse_args()
    ctx, d = setup.context(args.dataset, gro=True)
    tasks = (d.test if args.split == "test" else d.val)[: args.n]
    out = C.RESULTS / "utility" / args.split / f"{args.dataset}_{args.client}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    llm = client = proj = xproj = None
    if args.client == "qwen":
        from recteller.llm import LLM
        llm = LLM(args.receiver)
        pp = C.CKPT / f"projector_{args.dataset}_{args.receiver.rstrip('/').split('/')[-1]}.pt"
        proj = projector.load(pp) if pp.exists() else None
        xp = C.CKPT / f"projector_{args.dataset}_other.pt"
        xproj = projector.load(xp) if xp.exists() else None
    else:
        from recteller.clients import ClosedClient
        client = ClosedClient()
    fw_ok = None
    for m in args.modes:
        f = out / f"{m}.json"
        if f.exists() and not args.force:
            print("skip", m)
            continue
        if m == "firewall":
            conv = out.parent / f"{args.dataset}_firewall_converter.json"
            if llm is None and not conv.exists():
                from recteller.llm import LLM
                llm = LLM()
            fw_ok = firewall_converter(ctx, tasks, llm, conv)
        if args.client == "qwen":
            pj = xproj if evaluate.SPECS[m][2].get("proj") == "other" else proj
            r = evaluate.run_open(m, tasks, ctx, llm, seeds=tuple(range(args.seeds)), projector=pj,
                                  firewall_ok=fw_ok if m == "firewall" else None)
        else:
            r = evaluate.run_closed(m, tasks, ctx, client, firewall_ok=fw_ok if m == "firewall" else None)
        r["ci95"] = evaluate.bootstrap_ci(r["per_task"])
        json.dump(r, open(f, "w"))
        print(f"{args.dataset} {args.client} {m:12s} HR@1 {r['hr1']:.3f} ±{r['ci95']:.3f} tok {r['tokens']:.0f}",
              flush=True)


if __name__ == "__main__":
    main()
