"""Aggregate results/*.json into results/REPORT.md (tables mirroring the paper's) and figures.

    python scripts/make_report.py
"""
import json
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from recteller import config as C
from recteller.evaluate import bootstrap_diff

R = C.RESULTS
NAMES = {"block": "Block", "web": "Web browsing", "mcp": "MCP tools", "iagent": "iAgent", "firewall": "Firewall",
         "openapi": "Open API", "gro": "GRO", "jtopk": "JSON top-K", "capped": "Capped top-K",
         "stitch": "Capped top-K + GRO", "einf": "RecTeller, eps=inf", "ours": "**RecTeller** (soft)",
         "ours_text": "RecTeller (text)", "pub": "RecTeller public only (k_p=0)", "ours@16": "RecTeller eps=16 (soft)"}
DS = {"games": "Games", "ind": "Industrial", "ml": "ML-1M"}


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def util(ds, client, mode, split="test"):
    return load(R / "utility" / split / f"{ds}_{client}" / f"{mode}.json")


def agr(ds, mode, acc, days=7, c30=False):
    vals = []
    for f in sorted((R / "security" / ds / "extraction").glob(f"{mode}_a{acc}_d{days}{'_c30' if c30 else ''}_s*.json")):
        vals.append(json.load(open(f)))
    if not vals:
        return None, None, 0
    a = np.array([v["agr10"] for v in vals])
    # hierarchical bootstrap: seeds, then held-out sessions
    rng = np.random.default_rng(0)
    per = [np.array(v["agr_per_session"]) for v in vals]
    bs = []
    for _ in range(1000):
        ss = rng.integers(0, len(per), len(per))
        bs.append(np.mean([per[s][rng.integers(0, len(per[s]), len(per[s]))].mean() for s in ss]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(a.mean()), float((hi - lo) / 2), len(vals)


def table_main(lines):
    lines.append("## Table 4 — Utility (HR@1) and client cost\n")
    lines.append("Q = Qwen3-8B (open-weight, RecTeller read through the projector as soft tokens), "
                 "G = GPT-4o-mini (closed, RecTeller read as compact text). Tok. = mean client input tokens per "
                 "Games task (Qwen tokenizer, a soft token counts as one). ± = 95% bootstrap half-width over tasks.\n")
    hdr = "| Access mode | " + " | ".join(f"{DS[d]} Q | {DS[d]} G" for d in DS) + " | Tok. (Games, Q) |"
    lines += [hdr, "|" + "---|" * (2 + 2 * len(DS))]
    for m in ["block", "web", "mcp", "iagent", "firewall", "openapi", "jtopk", "ours", "ours_text", "ours@16", "einf", "pub"]:
        row = [NAMES.get(m, m)]
        for d in DS:
            for cl in ("qwen", "gpt"):
                mm = {"ours": "ours_text", "ours@16": "ours_text@16"}.get(m, m) if cl == "gpt" else m
                r = util(d, cl, mm)
                row.append(f"{fmt(r['hr1'])} ±{r['ci95']:.3f}" if r else "–")
        r = util("games", "qwen", m)
        row.append(f"{r['tokens']:.0f}" if r else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    # gaps
    for d in DS:
        o, a, j = util(d, "qwen", "ours"), util(d, "qwen", "openapi"), util(d, "qwen", "jtopk")
        if o and a:
            g, h = bootstrap_diff(a["per_task"], o["per_task"])
            s = f"- {DS[d]} (Q): Open API − RecTeller = {100 * g:.1f} ± {100 * h:.1f} HR@1 points"
            if j:
                g2, h2 = bootstrap_diff(j["per_task"], o["per_task"])
                s += f", JSON top-K − RecTeller = {100 * g2:.1f} ± {100 * h2:.1f}"
            lines.append(s)
        og, ag = util(d, "gpt", "ours_text"), util(d, "gpt", "openapi")
        if og and ag:
            g, h = bootstrap_diff(ag["per_task"], og["per_task"])
            lines.append(f"- {DS[d]} (G): Open API − RecTeller = {100 * g:.1f} ± {100 * h:.1f} HR@1 points")
    lines.append("")


def table_security(lines):
    lines.append("## Table 5 — Security (attacks over 7 days)\n")
    lines.append("Agr@10 = top-10 agreement of the extracted surrogate with the victim on 1,000 held-out real "
                 "sessions (mean over attack seeds ± 95% hierarchical-bootstrap half-width). Floor = RecTeller "
                 "public-only responses (k_p = 0). MIA = AUC of the Zhang et al. attack (± Hanley–McNeil 95%).\n")
    lines.append("| Access mode | Games Agr@10 1 acc. | Games Agr@10 50 acc. | Games MIA | ML-1M Agr@10 50 acc. | ML-1M MIA |")
    lines.append("|---|---|---|---|---|---|")
    for m in ["openapi", "gro", "mcp", "jtopk", "capped", "stitch", "einf", "ours", "pub"]:
        row = [NAMES.get(m, m) + (" (floor)" if m == "pub" else "")]
        for d, accs in (("games", (1, 50)), ("ml", (50,))):
            for a in accs:
                mu, h, n = agr(d, m, a)
                row.append(f"{fmt(mu)} ±{h:.3f} (n={n})" if mu is not None else "–")
            r = load(R / "security" / d / "mia" / f"{m}.json")
            row.append(f"{fmt(r['auc'])} ±{r['ci95']:.3f}" if r else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    fl = load(R / "security" / "games" / "flood.json")
    if fl:
        lines.append("**Flood** (1,000 requests, one account, 5 sessions, 5% repeats): platform calls forwarded — " +
                     ", ".join(f"{NAMES.get(k, k)}: {v['platform_calls']}" for k, v in fl.items()) + "\n")
    vi = load(R / "security" / "games" / "violations.json")
    if vi:
        g, a1 = vi["gatekeeper"], vi["a1_llm_policy"]
        lines.append(f"**Policy violations** ({vi['n']} attempts, 4 categories): gatekeeper modes (code policy) "
                     f"{g['rate_pct']:.1f}% {g['violations']} (unparsed: {g['unparsed']}), "
                     f"A1 LLM-enforced policy {a1['rate_pct']:.1f}% {a1['violations']}. Rule-of-three 95% bound "
                     f"for zero violations: {300 / vi['n']:.1f}%.\n")


def table_tradeoff(lines):
    lines.append("## Fig. 3 — Utility–leakage trade-off (Games, Qwen3-8B, 50 accounts)\n")
    lines.append("| Point | HR@1 | Agr@10 (50 acc.) |")
    lines.append("|---|---|---|")
    pts = [("pub", "pub", "public only (k_p=0)")] + [(f"ours@{e:g}", f"ours@{e:g}", f"RecTeller eps={e:g}") for e in (0.5, 1, 2)] + \
          [("ours", "ours", "RecTeller eps=4 (default)")] + [("ours@8", "ours@8", "RecTeller eps=8"), ("ours@16", "ours@16", "RecTeller eps=16"), ("einf", "einf", "RecTeller eps=inf")] + \
          [("capped", "capped", "Capped top-K (p=0)")] + [(f"rr{p:g}", f"rr{p:g}", f"random replacement p={p:g}") for p in (0.1, 0.3, 0.5)] + \
          [("gro", "gro", "GRO"), ("stitch", "stitch", "Capped top-K + GRO")]
    for um, sm, label in pts:
        r = util("games", "qwen", um)
        mu, h, n = agr("games", sm, 50)
        lines.append(f"| {label} | {fmt(r['hr1']) if r else '–'} | {fmt(mu)}{f' ±{h:.3f}' if mu is not None else ''} |")
    lines.append("")
    val = [(e, util("games", "qwen", f"ours@{e:g}" if e != 4 else "ours", "val")) for e in (1, 2, 4, 8, 16)]
    val.append((float("inf"), util("games", "qwen", "einf", "val")))
    if any(v for _, v in val):
        lines.append("Validation (budget selection rule: smallest eps whose doubling gains < 1.5 HR@1 points): " +
                     ", ".join(f"eps={e:g}: {fmt(v['hr1']) if v else '–'}" for e, v in val) + "\n")


def table_ablation(lines):
    lines.append("## Table 6 — Ablations (Games, Qwen3-8B, Agr@10 with 50 accounts)\n")
    lines.append("| Variant | HR@1 | Tok. | Agr@10 |")
    lines.append("|---|---|---|---|")
    rows = [("RecTeller (cap + noise)", "ours", "ours"), ("− noise (eps=inf)", "einf", "einf"),
            ("− cap (per-session budgets)", "ours", "nocap"), ("− noise − cap", "einf", "neither"),
            ("A3 shuffled attributes", "ours_shuf", None), ("A4 text rendering", "ours_text", "ours"),
            ("A5 projector of another model", "ours_xproj", None),
            ("k_p = 1", "kp1", "kp1"), ("k_p = 5", "kp5", "kp5"), ("no levels", "nolev", "nolev")]
    for label, um, sm in rows:
        r = util("games", "qwen", um)
        mu, h, n = agr("games", sm, 50) if sm else (None, None, 0)
        lines.append(f"| {label} | {fmt(r['hr1']) if r else '–'} | {fmt(r['tokens'], 0) if r else '–'} | "
                     f"{fmt(mu)}{f' ±{h:.3f}' if mu is not None else ''} |")
    ctrl = load(R / "utility" / "test" / "games_qwen_llama" / "ours.json")
    if ctrl:
        lines.append(f"\nA5 positive control: Llama-3.1-8B with its own projector, HR@1 = {ctrl['hr1']:.3f}.")
    vi = load(R / "security" / "games" / "violations.json")
    if vi:
        lines.append(f"\nA1 (policy in the parser prompt): {vi['a1_llm_policy']['rate_pct']:.1f}% of attempts succeed "
                     f"(code policy: {vi['gatekeeper']['rate_pct']:.1f}%).")
    reps = [load(f) for f in sorted((R / "security" / "games" / "repetition").glob("s*.json"))]
    reps = [r for r in reps if r]
    if reps:
        m = {k: np.mean([r[k] for r in reps]) for k in ("full", "per_session", "per_account_exact", "per_account_perturbed")}
        lines.append(f"\n**Repetition attack** (x{C.REP_R}, {len(reps)} seeds): full {m['full']:.3f}, repeats keyed per "
                     f"session {m['per_session']:.3f}, exact repeats keyed per account {m['per_account_exact']:.3f}, "
                     f"perturbed repeats keyed per account {m['per_account_perturbed']:.3f} "
                     f"({np.mean([r['perturbed_ctx'] for r in reps]):.0f} contexts personalised).")
    lines.append("")


def table_budget(lines):
    b = load(R / "budget" / "games.json")
    if not b:
        return
    lines.append("## Fig. budget (a) — day-mean HR@1 vs tasks per account-day (Games, Qwen3-8B)\n")
    mcp, jt = util("games", "qwen", "mcp"), util("games", "qwen", "jtopk")
    lines.append("| n | RecTeller | Capped top-K | MCP tools | JSON top-K |")
    lines.append("|---|---|---|---|---|")
    for n in range(C.SEQ_N):
        lines.append(f"| {n + 1} | {b['ours']['day_mean'][n]:.3f} | {b['capped']['day_mean'][n]:.3f} | "
                     f"{fmt(mcp['hr1']) if mcp else '–'} | {fmt(jt['hr1']) if jt else '–'} |")
    lines.append("")


def table_horizon(lines):
    lines.append("## Long attacks — RecTeller Agr@10, 50 accounts, by horizon\n")
    lines.append("| Days | daily cap | daily + 30-day cap (64) |")
    lines.append("|---|---|---|")
    for days in (3, 7, 30, 90):
        a = agr("games", "ours", 50, days)
        c = agr("games", "ours", 50, days, True) if days > 7 else a
        lines.append(f"| {days} | {fmt(a[0])} | {fmt(c[0])} |")
    lines.append("")


def sasrec_info(lines):
    lines.append("## Recommenders (SASRec, full-catalogue leave-last-out test)\n")
    for d in DS:
        r = load(R / "sasrec" / f"{d}_victim.json")
        g = load(R / "sasrec" / f"{d}_gro.json")
        if r:
            lines.append(f"- {DS[d]}: HR@10 {r['test']['hr@10']:.3f}, NDCG@10 {r['test']['ndcg@10']:.3f}" +
                         (f", GRO lambda={g['chosen_lambda']}" if g else ""))
    for d in DS:
        p = load(R / "projector" / f"{d}_Qwen3-8B.json")
        if p:
            lines.append(f"- Projector {DS[d]} (Qwen3-8B), held-out synthetic tasks: soft {p['held_soft_acc']:.3f} vs "
                         f"compact text {p['held_text_acc']:.3f}")
    lines.append("")


def main():
    lines = ["# RecTeller reproduction — results\n", "Generated by `scripts/make_report.py` from `results/`.\n"]
    sasrec_info(lines)
    table_main(lines)
    table_security(lines)
    table_tradeoff(lines)
    table_ablation(lines)
    table_budget(lines)
    table_horizon(lines)
    (R / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
