"""Cross-trait figures for Exp 1's Parts 3-4, from cache/checks/<kind>_<trait>.json (written by
export.py, scale.py, pilot.py, sweep.py and dose.py inside each trait's esper session). Plain
Python, drawn in the Savoy style:

    uv run python topics/persona-vectors/checks/plot.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import savoy  # noqa: E402

TOPIC = Path(__file__).resolve().parent.parent
FIG = TOPIC / "figures"
TRAITS = {t: savoy.color(i) for i, t in enumerate(("dishonest", "pirate", "mystic"))}
D = {t: json.loads((TOPIC / "cache" / "checks" / f"export_{t}.json").read_text()) for t in TRAITS}
BLOCK = "block (layer 0–35)"
QUIRK = (16, 19)


def load(kind):
    return {t: json.loads(f.read_text()) for t in TRAITS if (f := TOPIC / "cache" / "checks" / f"{kind}_{t}.json").exists()}


def err(ax, x, y, e, color, **kw):
    """A sparse line (a dot per measurement) with ±1 s.e. whiskers."""
    savoy.line(ax, x, y, color, points=True, end=False, yerr=e, **kw)


def quirk(ax):
    ax.axvspan(*QUIRK, color=savoy.T["rule"], alpha=0.6, lw=0, zorder=0)


def ramp(n):
    """n ordered colours (weak → strong push) from the sequential ramp."""
    return [savoy.seq_cmap()(v) for v in np.linspace(0.4, 1, n)]


def save(fig, name):
    savoy.save(fig, FIG / f"{name}.png")


# Scores: how strongly each answer shows the trait, per condition
bins = np.arange(0, 105, 5)
fig, axes = savoy.figure(1, 3, height=190, ytitle="answers", xtitle="trait score from the judge (0 absent, 100 strong)",
                         titles=list(TRAITS))
top = max(np.histogram(D[t]["scores"][c]["trait"], bins)[0].max() for t in TRAITS for c in ("POS", "NEG", "BASE"))
for ax, t in zip(axes, TRAITS):
    for cond, c in (("POS", savoy.color(0)), ("NEG", savoy.color(1))):
        savoy.hist(ax, np.histogram(D[t]["scores"][cond]["trait"], bins)[0], bins, c)
    savoy.hist(ax, np.histogram(D[t]["scores"]["BASE"]["trait"], bins)[0], bins, baseline=True)
    savoy.yticks(ax, 0, top, n=4, integer=True)
    savoy.xticks(ax, 0, 100, ticks=[0, 50, 100])
savoy.key(fig, "trait prompt (POS)", savoy.color(0))
savoy.key(fig, "anti-trait prompt (NEG)", savoy.color(1))
savoy.key(fig, "no system prompt (BASE)", savoy.T["sn"], "base")
save(fig, "scores_three_traits")

# Whole answer vs first 40 tokens: cosine against the noise line
blocks = list(range(len(D["dishonest"]["noise"])))
savoy.chart("multiples", {
    "x": {"title": BLOCK, "values": blocks, "ticks": [0, 16, 35]},
    "y": {"title": "cosine between the two POS − NEG vectors", "domain": [0.6, 1]},
    "series": [{"id": "cos", "label": "whole answer vs first 40 tokens"},
               {"id": "noise", "label": "noise: the first-40 vector from two random halves", "baseline": True}],
    "panels": [{"title": t, "values": {"cos": D[t]["cos_full_first40"], "noise": D[t]["noise"]}} for t in TRAITS],
}, FIG / "first40_three_traits.png")

# Whole answer vs first 40 tokens, on behaviour: separation (left) and length (right)
fig, axes = savoy.figure(3, 2, height=150, ytitle=["separation, Cohen's d", "cosine with a length direction"], xtitle=BLOCK,
                         row_titles=list(TRAITS))
WIN = (("full", savoy.color(0), "whole answer"), ("first40", savoy.color(1), "first 40 tokens"))
for row, t in enumerate(TRAITS):
    sep, lc = axes[2 * row], axes[2 * row + 1]
    vals, cvals = [], [0]
    for red, c, _ in WIN:
        w = D[t]["window_check"][red]
        x = np.arange(len(w["sep"]["mean"]))
        savoy.line(sep, x, w["sep"]["mean"], c, band=list(zip(w["sep"]["lo"], w["sep"]["hi"])))
        vals += w["sep"]["lo"] + w["sep"]["hi"]
        if w["len_cos"] is not None:
            savoy.line(lc, x, w["len_cos"], c)
            cvals += w["len_cos"]
    savoy.ref(lc, 0)
    savoy.yticks(sep, min(vals), max(vals), n=4)
    savoy.yticks(lc, min(cvals), max(cvals), n=4)
    for ax in (sep, lc):
        savoy.xticks(ax, 0, x[-1], ticks=[0, 16, 24, 35])
for _, c, name in WIN:
    savoy.key(fig, name, c)
save(fig, "window_separation_length")

# Pooled vs matched, on behaviour
panels = []
for t in TRAITS:
    m = D[t]["match_check"]
    panels.append({"title": t,
                   "values": {k: m[k]["mean"] for k in ("pooled", "matched")},
                   "bands": {k: list(zip(m[k]["lo"], m[k]["hi"])) for k in ("pooled", "matched")}})
savoy.chart("multiples", {
    "x": {"title": BLOCK, "values": list(range(len(panels[0]["values"]["pooled"]))), "ticks": [0, 16, 35]},
    "y": {"title": "separation on unseen answers, Cohen's d"},
    "series": [{"id": "pooled", "label": "pooled: one pile minus the other"},
               {"id": "matched", "label": "matched: subtract within each question first"}],
    "panels": panels,
}, FIG / "pooled_vs_matched_three_traits.png")

# POS − NEG vs POS − BASE
savoy.chart("line", {
    "x": {"title": BLOCK, "values": blocks},
    "y": {"title": "cos(POS − NEG, POS − BASE)", "domain": [0.6, 1]},
    "series": [{"id": t, "label": f"{t} ({D[t]['window']}, matched)", "short": t, "values": D[t]["cos_neg_base"]}
               for t in TRAITS],
    "refs": [{"y": 0.9, "label": "below this, Part 4 repeats with POS − BASE", "threshold": True}],
    "height": 260,
}, FIG / "pos_neg_vs_pos_base.png")

# Scale for steering: residual stream length and each trait's raw gap (scale.py)
S = load("scale")
tn = S["dishonest"]["tok_norm"]
x = np.arange(len(tn["median"]))
fig, (a, b) = savoy.figure(1, 2, height=240, ytitle=["vector length (log scale)", "gap / stream length"], xtitle=BLOCK)
stream = savoy.color(3)
savoy.line(a, x, tn["median"], stream, band=list(zip(tn["p10"], tn["p90"])))
savoy.key(fig, "residual stream length (median, 10–90th pct)", stream)
for t, c in TRAITS.items():
    savoy.line(a, x, S[t]["gap"], c)
    savoy.line(b, x, np.array(S[t]["gap"]) / np.array(tn["median"]), c)
    savoy.key(fig, f"{t}: POS − NEG gap", c)
savoy.yticks(a, min(min(tn["p10"]), *(min(S[t]["gap"]) for t in TRAITS)), max(tn["p90"]), log=True)
r = [v for t in TRAITS for v in np.array(S[t]["gap"]) / np.array(tn["median"])]
savoy.yticks(b, 0, max(r), include=[0], n=5)
for ax in (a, b):
    quirk(ax)
    savoy.xticks(ax, 0, x[-1], ticks=[0, 8, 16, 24, 35])
savoy.note(b, 17.5, b.get_ylim()[1] * 0.92, "block-16 quirk", anchor="middle")
save(fig, "steering_scale")

# Pilot: trait and coherence against the size of the push, one block (pilot.py)
P = load("pilot")
if P:
    d0 = next(iter(P.values()))
    fig, axes = savoy.figure(1, 2, height=240, ytitle=["mean trait score", "mean coherence"],
                             xtitle=f"push / stream length at block {d0['layer']} (log scale)")
    ticks = set()
    for t, d in P.items():
        for ax, k in zip(axes, ("trait", "coh")):
            for rnd in ("coarse", "zoom"):
                r = [x for x in d["rows"] if x["frac"] > 0 and x["round"] == rnd]
                ticks |= {x["frac"] for x in r}
                kw = dict(linestyle=savoy.DASH_THRESHOLD, alpha=0.5) if rnd == "coarse" else {}
                err(ax, [x["frac"] for x in r], [x[k] for x in r], [x[f"{k}_sem"] for x in r], TRAITS[t], **kw)
            ax.axvline(d["gap_frac"], color=TRAITS[t], linestyle=savoy.DASH_EVENT, linewidth=savoy.px(1.2))
    for ax in axes:
        savoy.xticks(ax, min(ticks) * 0.85, max(ticks) * 1.15, ticks=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6], log=True)
        savoy.yticks(ax, 0, 100)
    for t, c in TRAITS.items():
        savoy.key(fig, t, c)
    savoy.key(fig, "coarse round", savoy.T["sn"], "dashed")
    savoy.key(fig, "one POS − NEG gap", savoy.T["sn"], "vline")
    save(fig, "pilot_strength")

# Layer sweeps (sweep.py). B: trait, coherence and the Part 3 filter per block, one line per strength.
# Then A / B / C side by side on the measure used to pick blocks: mean per-answer trait × coherence / 100.
W = load("sweep")
VARIANTS = {"all": "A: every token", "response": "B: the paper's", "clean": "C: clean cache"}


def lines(ax, d, pos, k, m=1):
    fr = d["sweeps"][pos]["fracs"]
    for f, c in zip(fr, ramp(len(fr))):
        r = sorted((x for x in d["cells"] if x["positions"] == pos and abs(x["frac"] - f) < 1e-6), key=lambda x: x["layer"])
        err(ax, [x["layer"] for x in r], [m * x[k] for x in r], [m * x[f"{k}_sem"] for x in r], c)
    ax.axhline(m * d["unsteered"][k], color=savoy.T["sn"], linewidth=savoy.px(1.4), linestyle=savoy.DASH_BASELINE)
    quirk(ax)
    savoy.yticks(ax, 0, 100, n=4)
    savoy.xticks(ax, 5, 31, ticks=[6, 12, 18, 24, 30])


if W:
    fig, axes = savoy.figure(len(W), 3, height=150, xtitle=BLOCK,
                             ytitle=["mean trait score", "mean coherence", "% doing both"],
                             row_titles=list(W))
    for row, (t, d) in enumerate(W.items()):
        for col, k in enumerate(("trait", "coh", "both")):
            lines(axes[3 * row + col], d, "response", k, 100 if k == "both" else 1)
        savoy.ref(axes[3 * row + 1], 75, threshold=True)
    for f, c in zip(next(iter(W.values()))["sweeps"]["response"]["fracs"], ramp(4)):
        savoy.key(fig, f"{f:.3g}", c)
    savoy.key(fig, "unsteered", savoy.T["sn"], "base")
    save(fig, "layer_sweep")

    pos_all = [p for p in VARIANTS if any(p in d["sweeps"] for d in W.values())]
    if len(pos_all) > 1:
        fig, axes = savoy.figure(len(W), len(pos_all), height=150, xtitle=BLOCK,
                                 ytitle="mean trait × coherence / 100",
                                 titles=[VARIANTS[p] for p in pos_all], row_titles=list(W))
        for row, (t, d) in enumerate(W.items()):
            for col, pos in enumerate(pos_all):
                ax = axes[row * len(pos_all) + col]
                if pos in d["sweeps"]:
                    lines(ax, d, pos, "prod")
        d0 = next(iter(W.values()))["sweeps"]
        for f, fc, c in zip(d0["response"]["fracs"], d0.get("clean", d0["response"])["fracs"], ramp(4)):
            savoy.key(fig, f"{f:.3g} (C {fc:.2g})", c)
        save(fig, "variant_sweep")

# Strength curve at each trait's chosen block and variant (dose.py)
DS = load("dose")
if DS:
    fig, axes = savoy.figure(1, 3, height=200, xtitle="push / stream length at the chosen block",
                             ytitle=["mean trait score", "mean coherence", "trait × coherence / 100"])
    NAME = {"all": "A", "response": "B", "clean": "C"}
    for t, d in DS.items():
        x = [r["frac"] for r in d["rows"]]
        for ax, k in zip(axes, ("trait", "coh", "prod")):
            err(ax, x, [r[k] for r in d["rows"]], [r[f"{k}_sem"] for r in d["rows"]], TRAITS[t])
            ax.axhline(d["unsteered"][k], color=TRAITS[t], linewidth=savoy.px(1.2), linestyle=savoy.DASH_BASELINE, alpha=0.8)
        savoy.key(fig, f"{t}: {NAME[d['positions']]} at {d['layer']}", TRAITS[t])
    savoy.key(fig, "unsteered", savoy.T["sn"], "dotted")
    savoy.ref(axes[1], 75, "the paper's bar", threshold=True)
    for ax in axes:
        savoy.yticks(ax, 0, 100, n=4)
        savoy.xticks(ax, 0.1, 0.6, ticks=[0.1, 0.3, 0.6], exact=False)
        ax.set_xlim(0.07, 0.63)
    save(fig, "strength_curve")
print("figures written to", FIG)
