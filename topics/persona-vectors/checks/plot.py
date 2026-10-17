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
print("figures written to", FIG)
