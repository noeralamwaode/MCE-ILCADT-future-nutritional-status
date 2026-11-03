"""Paper 2 - all result figures from the frozen runs (single script, Google Colab or local).

Figures
  fig2  Dataset and temporal blocks: (A) eligible observations per month by status, (B) monthly share stunted,
        (C) blocks of analysis 1   (aggregate files dataset_monthly.csv and v4_stage2_pair_summary.csv)
  fig3  Analysis 1: model minus persistence, h = 1-6        (frozen metrics and bootstrap CIs, SAP v4)
  fig4  Analysis 2: AUPRC for new stunting + decision curves (frozen metrics and CIs, SAP v5, exploratory)
  pr    Analysis 2: precision-recall curves                (frozen test probabilities)

Integrity
  * Only frozen outputs are read; no model, threshold, split or metric is refitted.
  * fig3 point differences are b minus a of the frozen point estimates; CIs are copied from the frozen files.
  * pr: the average precision recomputed here must equal the frozen AUPRC (|difference| < 1e-9) before
    anything is drawn; otherwise the script stops. No file-hash check is needed for this guarantee.

Folder layout expected under --parent (Google Drive: MyDrive/Toddlers_Future_Prediction_BUTUR):
  MCE_FP_v4/runs/primary/h1..h6/artifacts/                  analysis 1 (frozen)
  MCE_FP_v5/runs/primary/h1,h3,h6/artifacts/                analysis 2 (frozen)
  MCE_FP_figures/data/v4_stage2_pair_summary.csv            aggregate pair counts per block
  MCE_FP_figures/data/dataset_monthly.csv                   aggregate monthly counts (run paper2_dataset.py first)

Requirements: Python 3.12, numpy, pandas 2.2 (needed to read the frozen .pkl), matplotlib, scikit-learn 1.6.1.
Colab runtime 2026.04 has all of them; no pip install is needed.

Usage (Colab cell):
  !python /content/drive/MyDrive/Toddlers_Future_Prediction_BUTUR/MCE_FP_figures/code/paper2_figures.py \
      --parent /content/drive/MyDrive/Toddlers_Future_Prediction_BUTUR
  Options: --only fig2 fig3 fig4 pr   (default: all)   --out <folder> (default: <parent>/MCE_FP_figures/figures)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, precision_recall_curve

# ----------------------------------------------------------------------------- style (validated palette)
INK, SEC, GRID, MUTED = "#0b0b0b", "#52514e", "#e1e0d9", "#8a8983"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PALE, NAVY = "#a9c6ee", "#173f75"  # light and dark tints of BLUE for the status stack (normal, severely stunted)
WIDTH_IN = 170 / 25.4  # BMC full page width (170 mm); text sizes 6.5-8.5 pt at final size; lines >= 0.6 pt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType fonts embedded (BMC: all fonts embedded)
                     "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#c3c2b7", "xtick.color": SEC,
                     "ytick.color": SEC, "axes.spines.top": False, "axes.spines.right": False})

# Analysis-2 class order (SAP v5): stable 0, worsening 1, improvement 2
STABLE, WORSENING, IMPROVEMENT = 0, 1, 2


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")



# ----------------------------------------------------------------------------- Figure 2
def fig2_temporal_split(pair_summary: Path, out: Path, monthly_csv: Path | None = None) -> None:
    """(A) eligible observations per month by status, (B) monthly share stunted and severely stunted
    [both from the aggregate dataset_monthly.csv written by paper2_dataset.py], and (C) prediction months
    (dark) and additional target months (light) of each block, h = 1, 3, 6. Without the monthly file only (C)."""
    colours = {"development": BLUE, "threshold_learning": ORANGE, "test": AQUA}
    labels = {"development": "Training", "threshold_learning": "Threshold\ntuning", "test": "Test"}
    s = pd.read_csv(pair_summary)
    s = s[(s.design == "primary") & s.block.isin(colours)]
    m0 = pd.Period("2022-01", "M")  # month index 0
    ticks = [i for i in range(57) if (m0 + i).month in (1, 7)]
    if monthly_csv is not None and monthly_csv.exists():
        d = pd.read_csv(monthly_csv).set_index("month_index")
        fig, (axa, axb, ax) = plt.subplots(3, 1, figsize=(WIDTH_IN, 6.6), sharex=True,
                                           gridspec_kw={"height_ratios": [1.15, 1, 1.05], "hspace": 0.32})
        x = d.index.to_numpy()
        bottom = np.zeros(len(d))
        for col, lab, c in [("severely_stunted", "Severely stunted", NAVY), ("stunted", "Stunted", BLUE), ("normal", "Normal", PALE)]:
            axa.bar(x, d[col], bottom=bottom, width=0.8, color=c, label=lab, zorder=3, linewidth=0)
            bottom += d[col].to_numpy()
        axa.set_ylabel("Eligible observations")
        axa.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        hd, lb = axa.get_legend_handles_labels()
        axa.legend(hd[::-1], lb[::-1], loc="upper left", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0, 1.02))
        axa.set_ylim(0, bottom.max() * 1.18)
        axb.plot(x, d["stunted_pct"], color=BLUE, lw=1.6, marker="o", ms=2.6, zorder=3)
        axb.plot(x, d["severely_stunted_pct"], color=NAVY, lw=1.6, marker="s", ms=2.6, zorder=3)
        ys, yv = float(d["stunted_pct"].iloc[-1]), float(d["severely_stunted_pct"].iloc[-1])
        if ys - yv < 4:  # keep the end labels apart
            mid = (ys + yv) / 2; ys, yv = max(ys, mid + 2), min(yv, mid - 2)
        axb.text(x[-1] + 0.8, ys, "Stunted", va="center", fontsize=7.5, color=BLUE)
        axb.text(x[-1] + 0.8, yv, "Severely\nstunted", va="center", fontsize=7.5, color=NAVY)
        axb.set_ylabel("Share of eligible\nobservations (%)"); axb.set_ylim(0, max(d["stunted_pct"].max() * 1.15, 5))
        for k, (axp, title) in enumerate([(axa, "Eligible observations per calendar month, by height-for-age status"),
                                          (axb, "Monthly proportion stunted and severely stunted"),
                                          (ax, "Temporal blocks of analysis 1")]):
            axp.set_title(f"{'ABC'[k]}  {title}", loc="left", fontsize=8.5, fontweight="bold", color=INK)
        for axp in (axa, axb):
            axp.grid(axis="y", color=GRID, lw=0.6, zorder=0); axp.tick_params(length=0)
            for mi in (12, 36, 48):  # first target month of training, first month of tuning and of test
                axp.axvline(mi - 0.5, color=MUTED, lw=0.8, ls="--", zorder=2)
        axb.tick_params(axis="x", labelbottom=False)
    else:
        fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.5))
    for r, h in enumerate([1, 3, 6]):
        y = 2 - r
        for _, b in s[s.h == h].iterrows():
            a0, a1 = map(int, b.anchor_months.split(".."))  # first and last prediction month of the block
            t1 = a1 + h                                       # last target month
            ax.barh(y, a1 - a0 + 1, left=a0 - 0.5, height=0.62, color=colours[b.block], edgecolor="white", linewidth=1.2, zorder=3)
            ax.barh(y, t1 - a1, left=a1 + 0.5, height=0.62, color=colours[b.block], alpha=0.30, edgecolor="white", linewidth=1.2, zorder=3)
            ax.text((a0 - 0.5 + t1 + 0.5) / 2, y, labels[b.block], ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white" if b.block == "development" else INK, zorder=4)
    ax.spines["left"].set_visible(False)
    ax.set_yticks(range(3)); ax.set_yticklabels(["$h$ = 6", "$h$ = 3", "$h$ = 1"], fontsize=8.5, color=INK); ax.tick_params(axis="y", length=0)
    ax.set_xticks(ticks); ax.set_xticklabels([(m0 + i).strftime("%b\n%Y") for i in ticks], fontsize=7)
    ax.set_xlim(-0.7, 56.7); ax.set_ylim(-0.55, 2.6); ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("Calendar month")
    ax.legend(handles=[Patch(color=SEC, label="Prediction months $t$ (predictors)"),
                       Patch(color=SEC, alpha=0.3, label="Additional target months $t+h$ (outcomes only)")],
              loc="upper left", bbox_to_anchor=(0, -0.36), ncol=2, frameon=False, fontsize=7.5)
    save(fig, out, "fig2_temporal_split")


# ----------------------------------------------------------------------------- Figure 3
def fig3_analysis1(v4_run: Path, out: Path) -> None:
    """Analysis 1: model minus persistence for macro-F1 and minority macro-recall, h = 1-6, all test pairs."""
    series = [("mce", "MCE (arg-max)", BLUE, "o", -0.12), ("mce_ilcadt", "MCE–ILCADT", ORANGE, "s", 0.12)]
    metrics = [("macro_f1", "A  Macro-F1"), ("minority_macro_recall", "B  Minority macro-recall")]
    rows = []
    for h in range(1, 7):
        art = v4_run / f"h{h}" / "artifacts"
        m = pd.read_csv(art / "test_classification_metrics.csv"); m = m[m.subset == "all"].set_index("method")
        d = pd.read_csv(art / "test_child_bootstrap_paired_differences.csv"); d = d[d.subset == "all"]
        for k, *_ in series:
            for met, _ in metrics:
                ci = d[(d.comparison == f"{k} minus persistence") & (d.metric == met)].iloc[0]
                rows.append({"h": h, "model": k, "metric": met, "diff": m.loc[k, met] - m.loc["persistence", met],
                             "ci_low": ci.ci_low, "ci_high": ci.ci_high})
    df = pd.DataFrame(rows); df.to_csv(out / "fig3_analysis1_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.7), sharex=True)
    for ax, (met, title) in zip(axes, metrics):
        ax.axhline(0, color=SEC, lw=0.8, zorder=1)
        for k, _, col, mk, off in series:
            s = df[(df.model == k) & (df.metric == met)]
            ax.errorbar(s.h + off, s["diff"], yerr=[s["diff"] - s.ci_low, s.ci_high - s["diff"]], fmt=mk, ms=4.5, color=col,
                        ecolor=col, elinewidth=1.4, capsize=0, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.set_xticks(range(1, 7)); ax.set_xlabel("Horizon $h$ (months)")
        ax.tick_params(length=0); ax.margins(y=0.16)
        ax.text(0.01, 0.99, "↑ favours model", transform=ax.transAxes, fontsize=6.5, color=SEC, ha="left", va="top")
        ax.text(0.01, 0.01, "↓ favours persistence", transform=ax.transAxes, fontsize=6.5, color=SEC, ha="left", va="bottom")
    axes[0].set_ylabel("Model minus persistence")
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, ls="", ms=5, label=l) for _, l, c, mk, _ in series],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig3_analysis1_vs_persistence")


# ----------------------------------------------------------------------------- Figure 4
def fig4_analysis2(v5_run: Path, out: Path) -> None:
    """Analysis 2 (exploratory): A = AUPRC with CI vs incidence; B = decision curves."""
    methods = [("mce", "MCE"), ("v4_mapped", "Analysis-1 model, mapped"), ("lr_33", "Logistic regression, 33 predictors"),
               ("lr_simple", "Logistic regression, simple"), ("best_base", "Best base learner")]
    dca = [("mce", "MCE", BLUE, "-"), ("v4_mapped_recal", "Analysis-1 model, mapped (recalibrated)", ORANGE, "--"),
           ("lr_33", "Logistic regression, 33 predictors", AQUA, "-."), ("lr_simple", "Logistic regression, simple", YELLOW, ":"),
           ("treat_all", "Follow up all normal children", MUTED, (0, (1, 1)))]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_IN * 0.96, 5.4), gridspec_kw={"height_ratios": [1, 1.05]})  # legend overhang keeps the cropped file <= 170 mm
    rows = []
    for j, h in enumerate([1, 3, 6]):
        art = v5_run / f"h{h}" / "artifacts"
        e = pd.read_csv(art / "primary_endpoint_metrics.csv").set_index("method")
        c = pd.read_csv(art / "primary_bootstrap_ci.csv").set_index("method")
        inc = e.loc["mce", "incidence"]
        ax = axes[0, j]; ax.axvline(inc, color=SEC, lw=0.9, ls="--", zorder=1)
        for i, (k, _) in enumerate(methods):
            y = len(methods) - 1 - i
            v, lo, hi = e.loc[k, "auprc_average_precision"], c.loc[k, "ci_low"], c.loc[k, "ci_high"]
            col = BLUE if k == "mce" else SEC
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot(v, y, "o", ms=5, color=col, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            rows.append({"h": h, "method": k, "auprc": v, "ci_low": lo, "ci_high": hi, "incidence": inc})
        ax.set_xlim(0, 0.3); ax.set_ylim(-0.6, len(methods) - 0.4); ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.tick_params(length=0)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels([l for _, l in methods][::-1] if j == 0 else [], fontsize=7)
        ax.set_title(f"$h$ = {h}  (incidence {inc:.1%})", fontsize=8, color=INK, loc="left"); ax.set_xlabel("AUPRC")
        ax.text(inc + 0.004, -0.55, "no skill / persistence", fontsize=6.5, color=SEC, va="bottom")
        p = pd.read_csv(art / "primary_decision_curve.csv").pivot(index="threshold", columns="method", values="net_benefit")
        p.assign(h=h).to_csv(out / f"fig4_dca_h{h}.csv")
        ax = axes[1, j]; ax.axhline(0, color=SEC, lw=0.8, zorder=1)
        for k, lab, col, ls in dca:
            ax.plot(p.index * 100, p[k], color=col, ls=ls, lw=1.2 if k == "treat_all" else 1.6, label=lab, zorder=3 if k == "mce" else 2)
        top = p[[k for k, *_ in dca if k != "treat_all"]].max().max()
        ax.set_ylim(-0.1 * top, 1.12 * top); ax.set_xlim(0, 10.2); ax.grid(color=GRID, lw=0.6, zorder=0); ax.tick_params(length=0)
        ax.set_xlabel("Risk threshold (%)"); ax.set_title(f"$h$ = {h}", fontsize=8, color=INK, loc="left")
        if j == 0:
            ax.set_ylabel("Net benefit")
    pd.DataFrame(rows).to_csv(out / "fig4_auprc_data.csv", index=False)
    fig.text(0.0, 0.985, "A  Ranking of new stunting (AUPRC, 95% CI)", fontsize=8.5, fontweight="bold", color=INK, va="top")
    fig.text(0.0, 0.50, "B  Decision curves", fontsize=8.5, fontweight="bold", color=INK, va="top")
    hdl, lab = axes[1, 0].get_legend_handles_labels()
    fig.legend(hdl, lab, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 0.965), h_pad=3.2)
    save(fig, out, "fig4_analysis2_new_stunting")


# ----------------------------------------------------------------------------- PR curves
def mask_normal(p: np.ndarray, status_t: np.ndarray) -> np.ndarray:
    """Impossible-class mask of SAP v5 (normal at t cannot improve; severely stunted cannot worsen), renormalised."""
    p = np.asarray(p, dtype=np.float64).copy()
    p[status_t == 0, IMPROVEMENT] = 0.0
    p[status_t == 2, WORSENING] = 0.0
    return p / p.sum(axis=1, keepdims=True)


def fig_pr_curves(v5_run: Path, out: Path) -> None:
    """Analysis 2: precision-recall curves for new stunting; stops if any recomputed AUPRC differs from the frozen value."""
    methods = [("mce", "MCE", BLUE, "-"), ("v4_mapped", "Analysis-1 model, mapped", ORANGE, "--"),
               ("lr_33", "Logistic regression, 33 predictors", AQUA, "-."), ("lr_simple", "Logistic regression, simple", YELLOW, ":")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.9), sharey=True)
    points, checks = [], []
    for ax, h in zip(axes, [1, 3, 6]):
        art = v5_run / f"h{h}" / "artifacts"
        test = pd.read_pickle(art / "test_pairs.pkl")
        s, y = test["status_t"].to_numpy(int), test["y"].to_numpy(int)
        probs = {"mce": mask_normal(np.load(art / "meta_dnn_test_probabilities.npy"), s)}
        probs.update(dict(np.load(art / "comparator_test_probabilities.npz")))
        frozen = pd.read_csv(art / "primary_endpoint_metrics.csv").set_index("method")["auprc_average_precision"]
        normal = s == 0
        yb = (y[normal] == WORSENING).astype(int)
        inc = yb.mean()
        ax.axhline(inc, color=SEC, lw=0.9, ls="--", zorder=1, label="Incidence = persistence (no ranking)")
        for k, lab, col, ls in methods:
            score = probs[k][normal, WORSENING]
            ap = average_precision_score(yb, score)
            checks.append({"h": h, "method": k, "ap_recomputed": ap, "ap_frozen": float(frozen[k]), "abs_diff": abs(ap - frozen[k])})
            if abs(ap - frozen[k]) > 1e-9:
                raise ValueError(f"h={h} {k}: recomputed AUPRC {ap:.6f} differs from frozen {frozen[k]:.6f}")
            prec, rec, _ = precision_recall_curve(yb, score)
            ax.step(rec, prec, where="post", color=col, ls=ls, lw=1.5, label=lab, zorder=3 if k == "mce" else 2)
            points += [{"h": h, "method": k, "recall": r, "precision": q} for r, q in zip(rec, prec)]
        ax.set_title(f"$h$ = {h}  (incidence {inc:.1%})", fontsize=8, loc="left", color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.6); ax.grid(color=GRID, lw=0.6, zorder=0); ax.tick_params(length=0)
        ax.set_xlabel("Recall (sensitivity)")
    axes[0].set_ylabel("Precision (PPV)")
    hdl, lab = axes[0].get_legend_handles_labels()
    fig.legend(hdl[1:] + hdl[:1], lab[1:] + lab[:1], loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    pd.DataFrame(points).to_csv(out / "fig_pr_curves_points.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "fig_pr_curves_ap_check.csv", index=False)
    print(pd.DataFrame(checks).round(6).to_string(index=False))
    save(fig, out, "fig_pr_curves")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 2 figures from frozen runs")
    ap.add_argument("--parent", type=Path, required=True, help="folder containing MCE_FP_v4, MCE_FP_v5 and MCE_FP_figures")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <parent>/MCE_FP_figures/figures)")
    ap.add_argument("--only", nargs="+", choices=["fig2", "fig3", "fig4", "pr"], default=["fig2", "fig3", "fig4", "pr"])
    a = ap.parse_args()
    v4_run = a.parent / "MCE_FP_v4" / "runs" / "primary"
    v5_run = a.parent / "MCE_FP_v5" / "runs" / "primary"
    pair_summary = a.parent / "MCE_FP_figures" / "data" / "v4_stage2_pair_summary.csv"
    monthly_csv = a.parent / "MCE_FP_figures" / "data" / "dataset_monthly.csv"  # from paper2_dataset.py (optional)
    out = a.out or a.parent / "MCE_FP_figures" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    needs = {"fig2": [pair_summary], "fig3": [v4_run], "fig4": [v5_run], "pr": [v5_run]}
    for p in {q for name in a.only for q in needs[name]}:
        if not p.exists():
            raise FileNotFoundError(f"not found: {p}")
    if "fig2" in a.only and not monthly_csv.exists():
        print(f"  note: {monthly_csv.name} not found -> Figure 2 without panels A-B (run paper2_dataset.py first)")
    steps = {"fig2": lambda: fig2_temporal_split(pair_summary, out, monthly_csv), "fig3": lambda: fig3_analysis1(v4_run, out),
             "fig4": lambda: fig4_analysis2(v5_run, out), "pr": lambda: fig_pr_curves(v5_run, out)}
    for name in a.only:
        print(f"[{name}]")
        steps[name]()
    print("✓ done:", out)


if __name__ == "__main__":
    main()
