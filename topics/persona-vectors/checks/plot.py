"""Cross-trait figures for Exp 1's Part 3, from cache/checks/export_<trait>.json (written by
export.py inside each trait's esper session). Plain Python:

    uv run python topics/persona-vectors/checks/plot.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TOPIC = Path(__file__).resolve().parent.parent
FIG = TOPIC / "figures"
TRAITS = {"dishonest": "C3", "pirate": "C0", "mystic": "C2"}
D = {t: json.loads((TOPIC / "cache" / "checks" / f"export_{t}.json").read_text()) for t in TRAITS}
BLOCK = "block (layer, 0-35)"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=110)
    plt.close(fig)


# Scores: how strongly each answer shows the trait, per condition
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for ax, t in zip(axes, TRAITS):
    for cond, c in (("POS", "C3"), ("NEG", "C0"), ("BASE", "C7")):
        ax.hist(D[t]["scores"][cond]["trait"], bins=np.arange(0, 105, 5), alpha=.6, color=c, label=cond)
    ax.set(title=t, xlabel="trait score from the judge (0 = absent, 100 = strong)")
    ax.legend(); ax.grid(alpha=.3)
axes[0].set_ylabel("number of answers")
fig.suptitle("How strongly each answer shows the trait, by prompt condition")
save(fig, "scores_three_traits")

# Whole answer vs first 40 tokens: cosine against the noise line
fig, ax = plt.subplots(figsize=(10, 4.5))
for t, c in TRAITS.items():
    ax.plot(D[t]["cos_full_first40"], color=c, marker="o", ms=3, label=f"{t}: whole answer vs first 40 tokens")
    ax.plot(D[t]["noise"], color=c, ls="--", lw=1, label=f"{t}: noise (same vector from two random halves)")
ax.set(xlabel=BLOCK, ylabel="cosine between the two POS − NEG vectors", ylim=(0.6, 1.01),
       title="Does averaging the whole answer or only its first 40 tokens change the vector?")
ax.grid(alpha=.3); ax.legend(fontsize=7.5, ncol=2, loc="lower right")
save(fig, "first40_three_traits")

# Whole answer vs first 40 tokens, on behaviour: separation and length
fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
for row, t in enumerate(TRAITS):
    for red, c, name in (("full", "C1", "whole answer"), ("first40", "C0", "first 40 tokens")):
        w = D[t]["window_check"][red]
        x = np.arange(len(w["sep"]["mean"]))
        axes[row, 0].plot(x, w["sep"]["mean"], color=c, label=name)
        axes[row, 0].fill_between(x, w["sep"]["lo"], w["sep"]["hi"], color=c, alpha=.2)
        if w["len_cos"] is not None:
            axes[row, 1].plot(x, w["len_cos"], color=c, marker="o", ms=2.5, label=name)
    axes[row, 0].set(ylabel=f"{t}\nseparation, Cohen's d")
    axes[row, 1].set(ylabel="cos with a pure length direction")
    axes[row, 1].axhline(0, color="k", lw=.5)
    for ax in axes[row]:
        ax.grid(alpha=.3); ax.legend(fontsize=8)
axes[0, 0].set_title("Trait vs not, on unseen answers, within the same question\n(20 random splits; band = min-max)")
axes[0, 1].set_title("Does the vector also encode answer length?\n(0 = no; built from NEG answers only)")
for ax in axes[2]:
    ax.set_xlabel(BLOCK)
save(fig, "window_separation_length")

# Pooled vs matched, on behaviour
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
for ax, t in zip(axes, TRAITS):
    for kind, c in (("pooled", "C0"), ("matched", "C1")):
        s = D[t]["match_check"][kind]
        x = np.arange(len(s["mean"]))
        ax.plot(x, s["mean"], color=c, label=kind)
        ax.fill_between(x, s["lo"], s["hi"], color=c, alpha=.2)
    ax.set(title=f"{t} ({D[t]['window']}): cos(pooled, matched) at block 24 = {D[t]['cos_pooled_matched'][24]:.2f}",
           xlabel=BLOCK, ylabel="separation on unseen answers, Cohen's d")
    ax.grid(alpha=.3); ax.legend()
fig.suptitle("Pooled (one pile of trait answers minus one pile of the rest) vs matched (subtract within each question first)")
save(fig, "pooled_vs_matched_three_traits")

# POS − NEG vs POS − BASE
fig, ax = plt.subplots(figsize=(10, 4))
for t, c in TRAITS.items():
    ax.plot(D[t]["cos_neg_base"], color=c, marker="o", ms=3, label=f"{t} ({D[t]['window']}, matched)")
ax.axhline(0.9, color="grey", ls="--", lw=1, label="0.9: below this, Part 4 repeats steering with POS − BASE")
ax.set(xlabel=BLOCK, ylabel="cos(POS − NEG, POS − BASE)", ylim=(0.6, 1.01),
       title="Does subtracting 'no system prompt' instead of 'the anti-trait prompt' change the vector?")
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
save(fig, "pos_neg_vs_pos_base")
# Scale for steering: residual stream length and each trait's raw gap (scale.py)
S = {t: json.loads((TOPIC / "cache" / "checks" / f"scale_{t}.json").read_text()) for t in TRAITS}
tn = S["dishonest"]["tok_norm"]
x = np.arange(len(tn["median"]))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharex=True)
axes[0].plot(x, tn["median"], color="k", label="residual stream length at one answer token (median)")
axes[0].fill_between(x, tn["p10"], tn["p90"], color="k", alpha=.12, label="10th-90th percentile")
for t, c in TRAITS.items():
    axes[0].plot(x, S[t]["gap"], color=c, marker="o", ms=3, label=f"{t}: raw POS − NEG gap length")
    axes[1].plot(x, np.array(S[t]["gap"]) / np.array(tn["median"]), color=c, marker="o", ms=3, label=t)
axes[0].set(yscale="log", xlabel=BLOCK, ylabel="vector length (log scale)",
            title="How big is the residual stream, and how far apart are POS and NEG?")
axes[1].set(xlabel=BLOCK, ylabel="gap length / residual stream length",
            title="The trait gap as a fraction of the residual stream")
for ax in axes:
    ax.axvspan(16, 19, color="grey", alpha=.15, lw=0)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
axes[1].text(17.5, axes[1].get_ylim()[1] * .95, "block-16\nquirk", ha="center", va="top", fontsize=8, color="grey")
save(fig, "steering_scale")

# Pilot: trait and coherence against the size of the push, one block (pilot.py)
P = {t: json.loads(f.read_text()) for t in TRAITS if (f := TOPIC / "cache" / "checks" / f"pilot_{t}.json").exists()}
if P:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    ticks = set()
    for t, d in P.items():
        base = [x for x in d["rows"] if x["frac"] == 0]
        for ax, k in zip(axes, ("trait", "coh")):
            b = sum(x[k] * x["n"] for x in base) / sum(x["n"] for x in base)
            for rnd, ls, a in (("coarse", ":", .45), ("zoom", "-", 1)):
                r = [x for x in d["rows"] if x["frac"] > 0 and x["round"] == rnd]
                if r:
                    ticks |= {x["frac"] for x in r}
                    ax.errorbar([x["frac"] for x in r], [x[k] for x in r], yerr=[x[f"{k}_sem"] for x in r],
                                color=TRAITS[t], marker="o", ms=4, ls=ls, alpha=a, capsize=3,
                                label=f"{t}, {rnd} round" + (f"  (unsteered: {b:.0f})" if rnd == "zoom" else ""))
            ax.axvline(d["gap_frac"], color=TRAITS[t], ls=(0, (1, 3)), lw=1.2)
    d0 = next(iter(P.values()))
    ticks = sorted(ticks)
    for ax, name in zip(axes, ("trait score", "coherence")):
        ax.set(xscale="log", ylim=(-2, 102), xlabel=f"α / residual stream length at block {d0['layer']} (length {d0['h_len']:.0f})",
               ylabel=f"mean {name} from the judge (0-100)")
        ax.set_xticks(ticks, [f"{x:.2g}" for x in ticks], fontsize=7); ax.minorticks_off()
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    axes[0].set_title("How much of the trait shows? (sparse dots: one POS − NEG gap per trait)")
    axes[1].set_title("Do the answers stay coherent?")
    fig.suptitle(f"Pilot: steering at block {d0['layer']} ({d0['positions']} positions). Coarse: 20 answers per point; "
                 "zoom: 40 (20 questions × 2). ±1 s.e.")
    save(fig, "pilot_strength")
print("figures written to", FIG)
