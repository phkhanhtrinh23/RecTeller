"""Aggregate results/*.json into results/REPORT.md (tables mirroring the paper's) and figures.

    python scripts/make_report.py
"""
import json
import math
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from recteller import config as C
from recteller.evaluate import bootstrap_diff

R = C.RESULTS
NAMES = {"block": "Block", "web": "Web browsing", "mcp": "MCP tools", "iagent": "iAgent", "firewall": "Firewall",
         "gro": "GRO", "ours": "**RecTeller**", "einf": "RecTeller, no noise", "pub": "RecTeller, public slots only",
         "capo2": "Cap-only k_p=3, n_max=2", "capo4": "Cap-only k_p=3, n_max=4"}
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
    lines.append("## Table 4. Per-task HR@1 (Q: Qwen3-8B, G: GPT-4o-mini) and client cost\n")
    lines.append("RecTeller is read through the projector as soft tokens by Q and as compact text by G. Tok. is the "
                 "mean client input tokens per Games task (Q, a soft token counts as one). ± is the 95% bootstrap "
                 "half-width over tasks.\n")
    hdr = "| Method | " + " | ".join(f"{DS[d]} Q | {DS[d]} G" for d in DS) + " | Tok. |"
    lines += [hdr, "|" + "---|" * (2 + 2 * len(DS))]
    for m in ["block", "web", "mcp", "iagent", "firewall", "gro", "ours"]:
        row = [NAMES[m]]
        for d in DS:
            for cl in ("qwen", "gpt"):
                r = util(d, cl, "ours_text" if (cl == "gpt" and m == "ours") else m)
                row.append(f"{fmt(r['hr1'])} ±{r['ci95']:.3f}" if r else "–")
        r = util("games", "qwen", m)
        row.append(fmt(r["tokens"], 0) if r else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for d in DS:
        o, i = util(d, "qwen", "ours"), util(d, "qwen", "iagent")
        if o and i:
            g, h = bootstrap_diff(o["per_task"], i["per_task"])
            lines.append(f"- {DS[d]} (Q): RecTeller − iAgent = {100 * g:.1f} ± {100 * h:.1f} HR@1 points")
    e, p = util("games", "qwen", "einf"), util("games", "qwen", "pub")
    if e and p:
        lines.append(f"- Reference points (Games, Q): RecTeller without noise {e['hr1']:.3f}, public slots only "
                     f"{p['hr1']:.3f}")
    lines.append("")


def table_security(lines):
    lines.append("## Table 5. Attacks over 7 days\n")
    lines.append("Agr@10 is the top-10 agreement of the extracted surrogate with the victim on 1,000 held-out real "
                 "sessions (mean over attack seeds ± 95% hierarchical-bootstrap half-width). MIA is the AUC of the "
                 "Zhang et al. attack (± Hanley–McNeil 95%). Viol. is the share of 300 policy-violation attempts that "
                 "succeed. Flood is platform calls per 1,000 flood requests.\n")
    fl = load(R / "security" / "games" / "flood.json") or {}
    vi = load(R / "security" / "games" / "violations.json")
    lines.append("| Method | Games Agr@10 1 acc. | Games Agr@10 50 acc. | Games MIA | ML-1M Agr@10 50 acc. | ML-1M MIA | Viol. (%) | Flood calls |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in ["gro", "mcp", "firewall", "einf", "ours"]:
        row = [NAMES[m]]
        for d, accs in (("games", (1, 50)), ("ml", (50,))):
            for a in accs:
                mu, h, n = agr(d, m, a)
                row.append("–" if (m == "einf" and a == 1) or mu is None else f"{fmt(mu)} ±{h:.3f}")
            r = load(R / "security" / d / "mia" / f"{m}.json")
            row.append(f"{fmt(r['auc'])} ±{r['ci95']:.3f}" if r else "–")
        if m in ("ours", "firewall") and vi:
            row.append(f"{vi['gatekeeper']['rate_pct']:.1f}")
        else:
            row.append("–")
        row.append(str(fl[m]["platform_calls"]) if m in fl else "–")
        lines.append("| " + " | ".join(row) + " |")
    f1, f50, fml = agr("games", "pub", 1), agr("games", "pub", 50), agr("ml", "pub", 50)
    lines.append(f"\nFloor (public responses only): Games {fmt(f50[0])} (50 acc.), {fmt(f1[0])} (1 acc.), ML-1M {fmt(fml[0])}.")
    if "nocap" in fl:
        lines.append(f"Flood with session-keyed limits (− cap ablation): {fl['nocap']['platform_calls']} calls.")
    lines.append("")


def table_tradeoff(lines):
    lines.append("## Fig. 3. HR@1 vs Agr@10 (50 accounts), Games, Qwen3-8B\n")
    lines.append("| Point | HR@1 | Agr@10 (50 acc.) |")
    lines.append("|---|---|---|")
    pts = [("pub", "public slots only (k_p=0)")] + [(f"ours@{e:g}", f"RecTeller eps={e:g}") for e in (0.5, 1, 2)] + \
          [("ours", "RecTeller eps=4 (default)"), ("ours@8", "RecTeller eps=8"), ("einf", "RecTeller eps=inf")] + \
          [(f"rr{p:g}", f"random replacement p={p:g}") for p in (0.1, 0.3, 0.5)] + \
          [("capo2", "cap-only k_p=3, n_max=2"), ("capo4", "cap-only k_p=3, n_max=4"), ("gro", "GRO")]
    for m, label in pts:
        r = util("games", "qwen", m)
        mu, h, n = agr("games", m, 50)
        lines.append(f"| {label} | {fmt(r['hr1']) if r else '–'} | {fmt(mu)}{f' ±{h:.3f}' if mu is not None else ''} |")
    b = load(R / "budget" / "games.json") or {}
    if "capo2" in b:
        lines.append(f"\nCap-only n_max=2, 12-task day mean: {b['capo2']['day_mean'][-1]:.3f}.")
    val = [(e, util("games", "qwen", f"ours@{e:g}" if e != 4 else "ours", "val")) for e in (0.5, 1, 2, 4, 8)]
    val.append((float("inf"), util("games", "qwen", "einf", "val")))
    if any(v for _, v in val):
        lines.append("\nValidation (budget selection rule: smallest eps whose doubling gains < 1.5 HR@1 points): " +
                     ", ".join(f"eps={e:g}: {fmt(v['hr1']) if v else '–'}" for e, v in val))
    lines.append("")


def table_sensitivity(lines):
    lines.append("## Table. Sensitivity on Games (eps_resp = 4, Qwen3-8B, 50 accounts)\n")
    cols = [("kp1", "k_p=1"), ("ours", "k_p=3 (default)"), ("kp5", "k_p=5"), ("nolev", "No levels (eps_w=0)")]
    lines.append("| | " + " | ".join(c for _, c in cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    lines.append("| HR@1 | " + " | ".join(fmt((util("games", "qwen", m) or {}).get("hr1")) for m, _ in cols) + " |")
    lines.append("| Agr@10 | " + " | ".join(fmt(agr("games", m, 50)[0]) for m, _ in cols) + " |")
    lines.append("")


def table_ablation(lines):
    lines.append("## Table 6. Ablations on Games (Qwen3-8B, Agr@10 with 50 accounts)\n")
    vi = load(R / "security" / "games" / "violations.json")
    viol_code = f"{vi['gatekeeper']['rate_pct']:.1f}" if vi else "–"
    viol_a1 = f"{vi['a1_llm_policy']['rate_pct']:.1f}" if vi else "–"
    # A1 and A2 change only who parses the request, so they share RecTeller's release (Agr@10).
    rows = [("RecTeller (cap + noise)", "ours", "ours", viol_code), ("− noise (eps_resp=inf)", "einf", "einf", viol_code),
            ("− cap (per-session budgets)", "ours", "nocap", viol_code), ("− noise − cap", "einf", "neither", viol_code),
            ("A1 LLM-enforced policy", "ours", "ours", viol_a1), ("A2 No parser (typed client)", "ours_a2", "ours", viol_code),
            ("A3 Shuffled attributes", "ours_shuf", "ours", viol_code), ("A4 Text rendering", "ours_text", "ours", viol_code),
            ("A5 Projector of another model", "ours_xproj", "ours", viol_code),
            ("Minimal: A2 + A4, no levels", "min", "nolev", viol_code)]
    lines.append("| Variant | HR@1 | Tok. (k) | Agr@10 | Viol. (%) |")
    lines.append("|---|---|---|---|---|")
    for label, um, sm, v in rows:
        r = util("games", "qwen", um)
        mu, h, n = agr("games", sm, 50)
        tok = f"{r['tokens'] / 1000:.2f}" if r else "–"
        lines.append(f"| {label} | {fmt(r['hr1']) if r else '–'} | {tok} | {fmt(mu)}{f' ±{h:.3f}' if mu is not None else ''} | {v} |")
    reps = [load(f) for f in sorted((R / "security" / "games" / "repetition").glob("s*.json"))]
    reps = [r for r in reps if r]
    if reps:
        m = {k: np.mean([r[k] for r in reps]) for k in ("per_session", "per_account_perturbed", "per_account_exact")}
        lines.append(f"| Repeats (x{C.REP_R}), keyed per session | – | – | {m['per_session']:.3f} | – |")
        lines.append(f"| Perturbed repeats, keyed per account | – | – | {m['per_account_perturbed']:.3f} | – |")
        lines.append(f"\nExact repeats keyed per account (cache hits): {m['per_account_exact']:.3f}. Perturbed repeats "
                     f"leave {np.mean([r['perturbed_ctx'] for r in reps]):.0f} personalized contexts.")
    ctrl = load(R / "utility" / "test" / "games_qwen_llama" / "ours.json")
    if ctrl:
        lines.append(f"\nA5 positive control: Llama-3.1-8B with its own projector, HR@1 = {ctrl['hr1']:.3f}.")
    lines.append("")


def table_budget(lines):
    b = load(R / "budget" / "games.json")
    if not b:
        return
    lines.append("## Fig. budget (a). Day-mean HR@1 vs tasks per account-day (Games, Qwen3-8B)\n")
    mcp = util("games", "qwen", "mcp")
    modes = [m for m in ("ours", "capo2", "capo4") if m in b]
    lines.append("| n | " + " | ".join(NAMES.get(m, m) for m in modes) + " | MCP tools |")
    lines.append("|---|" + "---|" * (len(modes) + 1))
    for n in range(C.SEQ_N):
        row = [f"{b[m]['day_mean'][n]:.3f}" for m in modes]
        lines.append(f"| {n + 1} | " + " | ".join(row) + f" | {fmt(mcp['hr1']) if mcp else '–'} |")
    lines.append("")


def accounts_needed(days: int, lists: int = 2800, n_max: int = 8, cap30: int | None = None) -> int:
    """Accounts needed to collect ``lists`` personalized lists in ``days`` daily windows (model-free)."""
    per_account = n_max * days
    if cap30 is not None:
        per_account = min(per_account, cap30 * math.ceil(days / 30))
    return math.ceil(lists / per_account)


def table_horizon(lines):
    lines.append("## Long attacks — RecTeller, 50 accounts, by horizon\n")
    lines.append("Accounts needed = accounts to collect 2,800 personalized lists, from the caps alone.\n")
    lines.append("| Days | Agr@10 daily cap | Agr@10 + 30-day cap (64) | Accounts daily cap | Accounts + 30-day cap |")
    lines.append("|---|---|---|---|---|")
    for days in (3, 7, 30, 90):
        a = agr("games", "ours", 50, days)
        c = agr("games", "ours", 50, days, True) if days > 7 else a
        lines.append(f"| {days} | {fmt(a[0])} | {fmt(c[0])} | {accounts_needed(days)} | {accounts_needed(days, cap30=64)} |")
    lines.append("")


def sasrec_info(lines):
    lines.append("## Recommenders (SASRec, full-catalog leave-last-out test)\n")
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
    table_sensitivity(lines)
    table_ablation(lines)
    table_budget(lines)
    table_horizon(lines)
    (R / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
