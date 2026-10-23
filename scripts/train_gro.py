"""Fine-tune the GRO-defended victim. Lambda is chosen on validation users from GRO's grid
{0.001, 0.01, 0.1, 1}: the largest lambda whose validation NDCG@10 stays within 10% of the
undefended model's.

    python scripts/train_gro.py --dataset games
"""
import argparse
import json

import _path  # noqa: F401
from recteller import config as C, data, gro, sasrec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    d = data.load(args.dataset)
    victim = sasrec.load(C.CKPT / f"sasrec_{args.dataset}_victim.pt")
    vp = d.val_pairs(d.split.val_users if d.split.val_users else None)
    base = sasrec.evaluate(victim, vp)
    seqs = d.train_seqs()
    res = {"base": base, "grid": {}}
    best = None
    for lam in [0.001, 0.01, 0.1, 1.0]:
        m = gro.finetune(victim, seqs, lam, epochs=args.epochs)
        r = sasrec.evaluate(m, vp)
        res["grid"][lam] = r
        print(lam, r, flush=True)
        if r["ndcg@10"] >= 0.9 * base["ndcg@10"]:
            best = (lam, m)
    lam, m = best if best else (0.001, gro.finetune(victim, seqs, 0.001, epochs=args.epochs))
    res["chosen_lambda"] = lam
    sasrec.save(m, C.CKPT / f"sasrec_{args.dataset}_gro.pt")
    json.dump(res, open(C.RESULTS / "sasrec" / f"{args.dataset}_gro.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
