"""Train the victim SASRec per dataset (and the MIA shadow model).

    python scripts/train_sasrec.py --dataset games
"""
import argparse
import json

import _path  # noqa: F401
from recteller import config as C, data, sasrec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--which", default="victim", choices=["victim", "shadow"])
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    C.ensure_dirs()
    d = data.load(args.dataset)
    drop = 0.2 if args.dataset == "ml" else 0.4
    if args.which == "victim":
        users = d.split.members
    else:  # attacker's shadow data: any users disjoint from the MIA targets and shadow non-members
        targets = d.split.mia_members or d.split.test_users[: C.MIA_MEMBERS]
        excl = set(targets) | set(d.split.nonmembers) | set(d.split.shadow_nonmembers)
        users = [u for u in d.split.seqs if u not in excl]
        users = users[:20000]
    seqs = d.train_seqs(users)
    vp = d.val_pairs(users)
    cfg = sasrec.SASRecConfig(d.cat.n, maxlen=C.DEFAULT.hist, dropout=drop)
    m = sasrec.train(seqs, d.cat.n, vp, epochs=args.epochs, cfg=cfg, bs=256 if args.dataset != "ml" else 128)
    sasrec.save(m, C.CKPT / f"sasrec_{args.dataset}_{args.which}.pt")
    # test: context = everything before the last item
    test_pairs = [(d.split.seqs[t.uid][:-1], t.target) for t in d.test]
    res = {"val": sasrec.evaluate(m, vp), "test": sasrec.evaluate(m, test_pairs)}
    print(args.dataset, args.which, res)
    out = C.RESULTS / "sasrec"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{args.dataset}_{args.which}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
