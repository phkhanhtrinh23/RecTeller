"""RQ2 security and RQ3/RQ4 leakage runs.

    python scripts/run_security.py extraction --dataset games --modes openapi mcp jtopk capped einf ours pub --accounts 1 50 --seeds 3
    python scripts/run_security.py mia --dataset games --modes openapi mcp jtopk capped einf ours
    python scripts/run_security.py flood --dataset games
    python scripts/run_security.py violations --dataset games
    python scripts/run_security.py repetition --dataset games --seeds 3
    python scripts/run_security.py horizon --dataset games --seeds 3

Mode names are those of recteller/modes.py. "ours@<eps>" sets eps_resp (3:1 split), "rr<p>" is
random replacement behind the cap, "pub" (k_p = 0, public responses only) is the floor.
"""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from recteller import config as C, setup
from recteller.attacks import extraction as ex
from recteller.attacks import flood, mia


def split_mode(m, proto):
    if "@" in m:
        base, e = m.split("@")
        return base, proto.with_eps(float(e))
    return m, None


def cmd_extraction(args):
    ctx, d = setup.context(args.dataset, gro=True)
    S = setup.sessions(d)
    out = C.RESULTS / "security" / args.dataset / "extraction"
    out.mkdir(parents=True, exist_ok=True)
    for m in args.modes:
        for a in args.accounts:
            for s in range(args.seeds):
                f = out / f"{m}_a{a}_d{args.days}{'_c30' if args.cap30 else ''}_s{s}.json"
                if f.exists() and not args.force:
                    continue
                base, proto = split_mode(m, ctx.proto)
                st, agr, _ = ex.run(ctx, base, ctx.scorer, S, n_accounts=a, days=args.days, seed=s, proto=proto,
                                    cap30=args.cap30, n_queries=args.queries, visits=args.visits)
                st["agr_per_session"] = agr.tolist()
                json.dump(st, open(f, "w"))
                print(f"{args.dataset} {m:10s} acc={a:2d} days={args.days} seed={s} Agr@10={st['agr10']:.3f} "
                      f"pers={st['personalised_pairs']} pairs={st['pairs']}", flush=True)


def cmd_mia(args):
    vctx, d = setup.context(args.dataset, gro=True)
    sctx, _ = setup.context(args.dataset, victim="shadow")
    if vctx.gro_scorer is not None:
        sctx.gro_scorer = sctx.scorer   # the attacker simulates GRO's interface with its plain shadow
    sp = d.split
    members = sp.mia_members or sp.test_users[: C.MIA_MEMBERS]   # ML-1M: no spare members after test/val
    tin = [sp.seqs[u][:-2] for u in members]
    tout = [sp.seqs[u][:-2] for u in sp.nonmembers]
    excl = set(members) | set(sp.nonmembers) | set(sp.shadow_nonmembers)
    shadow_train = [u for u in sp.seqs if u not in excl][:20000]
    rng = np.random.default_rng(0)
    sin_u = [shadow_train[i] for i in rng.choice(len(shadow_train), size=len(sp.shadow_nonmembers), replace=False)]
    sin = [sp.seqs[u][:-2] for u in sin_u]
    sout = [sp.seqs[u][:-2] for u in sp.shadow_nonmembers]
    out = C.RESULTS / "security" / args.dataset / "mia"
    out.mkdir(parents=True, exist_ok=True)
    for m in args.modes:
        f = out / f"{m}.json"
        if f.exists() and not args.force:
            continue
        base, proto = split_mode(m, vctx.proto)
        if proto is not None:
            vctx.proto = sctx.proto = proto
        r = mia.attack(vctx, sctx, base, tin, tout, sin, sout)
        vctx.proto = sctx.proto = C.DEFAULT
        json.dump(r, open(f, "w"))
        print(f"{args.dataset} MIA {m:10s} AUC={r['auc']:.3f} ±{r['ci95']:.3f} (shadow {r['shadow_train_auc']:.3f})",
              flush=True)


def cmd_flood(args):
    ctx, d = setup.context(args.dataset, gro=True)
    res = {}
    for m in args.modes:
        res[m] = flood.run(ctx, m, C.FLOOD_N, C.FLOOD_SESSIONS, C.FLOOD_REPEAT)
        print(m, res[m], flush=True)
    out = C.RESULTS / "security" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "flood.json", "w"), indent=1)


def cmd_violations(args):
    from recteller.attacks import violations as V
    from recteller.llm import LLM
    from recteller.parser import LLMParser, PolicyParser
    ctx, d = setup.context(args.dataset)
    llm = LLM(C.PARSER_MODEL)
    suite = V.build_suite(75)
    res = {"gatekeeper": V.run_gatekeeper(ctx, LLMParser(llm, ctx.cat.n), suite),
           "a1_llm_policy": V.run_llm_policy(ctx, PolicyParser(llm, ctx.cat.n), suite),
           "n": len(suite)}
    # Firewall: the same LLM converter with a deterministic verifier over the closed protocol
    res["firewall"] = res["gatekeeper"]
    print(json.dumps(res, indent=1))
    out = C.RESULTS / "security" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "violations.json", "w"), indent=1)


def cmd_repetition(args):
    """Repetition attack: each personalised context of the default attack resubmitted r times."""
    from recteller.gatekeeper import Request
    from recteller.intents import Personalise, Search
    from recteller.modes import build
    ctx, d = setup.context(args.dataset)
    S = setup.sessions(d)
    out = C.RESULTS / "security" / args.dataset / "repetition"
    out.mkdir(parents=True, exist_ok=True)
    r = C.REP_R
    for s in range(args.seeds):
        f = out / f"s{s}.json"
        if f.exists() and not args.force:
            continue
        pairs, st = ex.collect(ctx, "ours", 50, 7, s, args.queries)
        pers = [p for p in pairs if p.personalised]
        public = [p for p in pairs if not p.personalised]
        res = {"full": float(ex.agreement(ctx.scorer, ex.train_surrogate(pairs, ctx.cat.n, s, args.visits), S).mean()),
               "contexts": len(pers)}

        def aggregate(mode, keying_perturb):
            # only n_max and the cache are under test here: B and P are lifted for all configs
            from dataclasses import replace as _r
            gk = build(mode, ctx, seed=s + 500, proto=_r(ctx.proto, B=10**9, P=10**9))
            toks = [gk.auth.issue(f"rep{a}", covers=[]) for a in range(50)]
            rng = np.random.default_rng(s + 7)
            agg = []
            n_ctx = len(pers)
            for j, p in enumerate(pers):
                tok = toks[j % 50]
                win = 7 * 24.0 * j / n_ctx
                counts = {}
                lv_sum = {}
                got = 0
                for rep in range(r):
                    sess = f"c{j}_{rep}"
                    ctxt = list(p.context)
                    if keying_perturb and rep > 0:
                        ctxt[rng.integers(len(ctxt))] = int(rng.integers(1, ctx.cat.n + 1))
                    q = Search(text=ctx.cat.text(p.context[-1]))
                    gk.serve(Request(tok, sess, win, intent=q))
                    resp = gk.serve(Request(tok, sess, win, intent=Personalise((), tuple(ctxt))))
                    if resp.status != "OK" or not resp.n_pers:
                        continue
                    got += 1
                    for i, lv in zip(resp.items[: resp.n_pers], resp.levels[: resp.n_pers]):
                        counts[i] = counts.get(i, 0) + 1
                        lv_sum[i] = lv_sum.get(i, 0) + lv
                if not counts:
                    agg.append(ex.Pair(p.context, p.items, p.q, False))
                    continue
                items = sorted(counts, key=lambda i: -counts[i])
                w = np.array([counts[i] * (1 + lv_sum[i] / counts[i]) for i in items], dtype=np.float64)
                agg.append(ex.Pair(p.context, items, (w / w.sum()).astype(np.float32), True))
            sur = ex.train_surrogate(agg + public, ctx.cat.n, s, args.visits)
            return float(ex.agreement(ctx.scorer, sur, S).mean()), sum(a.personalised for a in agg)

        res["per_session"], res["per_session_ctx"] = aggregate("nocap", False)   # fresh noise per repeat
        res["per_account_exact"], _ = aggregate("ours", False)                   # cache hits
        res["per_account_perturbed"], res["perturbed_ctx"] = aggregate("ours", True)
        json.dump(res, open(f, "w"))
        print(args.dataset, "repetition seed", s, res, flush=True)


def cmd_horizon(args):
    ctx, d = setup.context(args.dataset)
    S = setup.sessions(d)
    out = C.RESULTS / "security" / args.dataset / "extraction"
    out.mkdir(parents=True, exist_ok=True)
    for days in (3, 7, 30, 90):
        for cap30 in (None, 64):
            if cap30 and days <= 7:
                continue                 # identical to the daily cap (56 <= 64)
            for s in range(args.seeds):
                f = out / f"ours_a50_d{days}{'_c30' if cap30 else ''}_s{s}.json"
                if f.exists() and not args.force:
                    continue
                st, agr, _ = ex.run(ctx, "ours", ctx.scorer, S, 50, days, s, cap30=cap30, n_queries=args.queries,
                                    visits=args.visits)
                st["agr_per_session"] = agr.tolist()
                json.dump(st, open(f, "w"))
                print(f"horizon d={days} cap30={cap30} s={s} Agr@10={st['agr10']:.3f} pers={st['personalised_pairs']}",
                      flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--dataset", default="games")
    ap.add_argument("--modes", nargs="+", default=["openapi", "gro", "mcp", "jtopk", "capped", "stitch", "einf", "ours", "pub"])
    ap.add_argument("--accounts", nargs="+", type=int, default=[1, 50])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--days", type=int, default=C.ATTACK_DAYS)
    ap.add_argument("--cap30", type=int, default=None)
    ap.add_argument("--queries", type=int, default=C.ATTACK_QUERIES)
    ap.add_argument("--visits", type=int, default=400_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"extraction": cmd_extraction, "mia": cmd_mia, "flood": cmd_flood, "violations": cmd_violations,
     "repetition": cmd_repetition, "horizon": cmd_horizon}[args.cmd](args)


if __name__ == "__main__":
    main()
