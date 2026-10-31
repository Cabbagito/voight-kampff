# Exp 2: with the persona held fixed, is there a linear "this statement is true" feature? The
# walk-through is 02_truth_probe.md; this file is only the code, one cell per step, titled as in
# the doc. Each step ends with one log line (the board's view of it).

# %% [markdown]
# ## Part 1 — Train a truth probe

# %% Load the model
import logging

import numpy as np
import pandas as pd
import savoy
from IPython.display import display

import lib

logging.basicConfig(level=logging.INFO, format="· %(message)s")   # no-op inside esper
log = logging.getLogger(__name__)

SEED = 0
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
LAYER = 20   # Exp 1's peak; the truth sweep below checks it holds here too

model, tok = lib.load_model_and_tokenizer()
log.info("%s · %d blocks", model.config._name_or_path.split("/")[-1], model.config.num_hidden_layers)

# %% Build the truth set
train, test = lib.build_truth_dataset(lib.DATA_DIR / "facts_true_false.csv", seed=SEED, n_train=400, n_test=150)
y_train = np.array([r["label"] for r in train])
y_test = np.array([r["label"] for r in test])
train_prompts, test_prompts = [r["prompt"] for r in train], [r["prompt"] for r in test]
print(repr(train[0]["prompt"]), "→ label", train[0]["label"])
log.info("train %d statements (%.0f%% true) · test %d (%.0f%% true) · honest persona on every prompt",
         len(train), 100 * y_train.mean(), len(test), 100 * y_test.mean())

# %% Read the activations
header = lib._make_prompt("an honest", "")   # everything before the statement
def read():
    out = {f"tr_{L}": X for L, X in lib.collect_activations_layers(model, tok, train_prompts, LAYERS).items()}
    out |= {f"te_{L}": X for L, X in lib.collect_activations_layers(model, tok, test_prompts, LAYERS).items()}
    for split, ps in [("tr", train_prompts), ("te", test_prompts)]:   # mean over the statement's own tokens
        out[f"{split}_mean"] = lib.collect_activations_mean(model, tok, ps, lib.count_response_tokens(tok, [header] * len(ps), ps), LAYER)
    return out
acts = lib.cached("truth", read)
log.info("last-token activations at blocks %s, plus the statement-mean at block %d", LAYERS, LAYER)

# %% Fit the truth probes
probes, res = {}, {}
for agg, Xtr, Xte in [("last", acts[f"tr_{LAYER}"], acts[f"te_{LAYER}"]), ("mean", acts["tr_mean"], acts["te_mean"])]:
    Xtr, Xte = Xtr.float().numpy(), Xte.float().numpy()
    lr, d = lib.fit_lr_probe(Xtr, y_train), lib.mm_direction(Xtr, y_train)
    probes[agg] = {"LR": lr.decision_function(Xte), "MM": lib.mm_score(Xte, d)}
    res[agg] = {k: lib.auroc(s, y_test) for k, s in probes[agg].items()}
res = pd.DataFrame(res)
display(res.round(3))

LABELS = {0: "false statements", 1: "true statements"}
PANELS = [("last", "LR"), ("last", "MM"), ("mean", "LR"), ("mean", "MM")]
fig, axes = savoy.figure(2, 2, height=170, ytitle="test statements", xtitle="probe score (high = true)",
                         titles=[f"{name}, {'last token' if agg == 'last' else 'statement mean'} · AUROC {res.loc[name, agg]:.3f}"
                                 for agg, name in PANELS])
for ax, (agg, name) in zip(axes, PANELS):
    bins = np.histogram_bin_edges(probes[agg][name], bins=20)
    counts = [np.histogram(probes[agg][name][y_test == lab], bins)[0] for lab in LABELS]
    for i, n in enumerate(counts):
        savoy.hist(ax, n, bins, savoy.color(i))
    savoy.yticks(ax, 0, max(n.max() for n in counts), integer=True)
    savoy.xticks(ax, bins[0], bins[-1], n=5)
for i, name in enumerate(LABELS.values()):
    savoy.key(fig, name, savoy.color(i))
savoy.save(fig, lib.TOPIC_DIR / "figures/truth_probe_scores.png")
log.info("block %d test AUROC: last token LR %.3f MM %.3f · statement mean LR %.3f MM %.3f",
         LAYER, res.loc["LR", "last"], res.loc["MM", "last"], res.loc["LR", "mean"], res.loc["MM", "mean"])

# %% Sweep the blocks
rows = []
for L in LAYERS:
    Xtr, Xte = acts[f"tr_{L}"].float().numpy(), acts[f"te_{L}"].float().numpy()
    rows.append({"block": L, "LR": lib.auroc(lib.fit_lr_probe(Xtr, y_train).decision_function(Xte), y_test),
                 "MM": lib.auroc(lib.mm_score(Xte, lib.mm_direction(Xtr, y_train)), y_test)})
sweep = pd.DataFrame(rows).set_index("block")
display(sweep.round(3))

fig, ax = savoy.figure(ytitle="held-out AUROC", xtitle="block (layer 0–35)")
ends = []
for i, p in enumerate(["LR", "MM"]):
    savoy.line(ax, sweep.index, sweep[p], savoy.color(i), points=True)
    savoy.key(fig, f"truth probe, {p}", savoy.color(i))
    ends.append((sweep[p].iloc[-1], f"truth {p}", savoy.color(i)))
SP_SWEEP = lib.CACHE_DIR / "sp_sweep.json"
if SP_SWEEP.exists():   # Exp 1's persona probe, for comparison
    sp = pd.read_json(SP_SWEEP)
    for p in ["LR", "MM"]:
        savoy.line(ax, sp.index, sp[p], None, baseline=True)
        ends.append((sp[p].iloc[-1], f"persona {p}", savoy.T["sn"]))
    savoy.key(fig, "persona probe (Exp 1), LR and MM", savoy.T["sn"], "base")
savoy.ref(ax, 0.5, "chance")
savoy.end_labels(ax, ends)
savoy.yticks(ax, 0.5, 1)
savoy.xticks(ax, LAYERS[0], LAYERS[-1], ticks=LAYERS)
savoy.save(fig, lib.TOPIC_DIR / "figures/truth_by_block.png")
log.info("truth AUROC by block, LR / MM: %s", " · ".join(f"{L}: {r.LR:.3f} / {r.MM:.3f}" for L, r in sweep.iterrows()))

# %% [markdown]
# ## Part 2 — Negate the statements

# %% Test on negated statements
neg = pd.read_csv(lib.DATA_DIR / "negation.csv")
X_neg = lib.cached(f"negation_L{LAYER}", lambda: {"X": lib.collect_activations(
    model, tok, [lib._make_prompt("an honest", s) for s in neg.statement], LAYER)})["X"].float().numpy()
Xtr = acts[f"tr_{LAYER}"].float().numpy()
lr, d_mm = lib.fit_lr_probe(Xtr, y_train), lib.mm_direction(Xtr, y_train)   # the block-20 last-token probes, no refit on negation
neg = neg.assign(LR=lr.decision_function(X_neg), MM=lib.mm_score(X_neg, d_mm))
neg_res = pd.DataFrame({sub: {p: lib.auroc(g[p].values, g.label.values) for p in ["LR", "MM"]}
                        for sub, g in [("affirmative", neg[neg.form == "aff"]), ("negated", neg[neg.form == "neg"]), ("all", neg)]}).T
display(neg_res.round(3))
print(neg[neg.form == "neg"].sample(4, random_state=SEED)[["statement", "label", "MM"]])

PANELS = [(f, p) for f in ["affirmative", "negated"] for p in ["LR", "MM"]]
fig, axes = savoy.figure(2, 2, height=170, ytitle="statements", xtitle="probe score (high = true)",
                         titles=[f"{p}, {form} · AUROC {neg_res.loc[form, p]:.3f}" for form, p in PANELS])
for ax, (form, p) in zip(axes, PANELS):
    sub = neg[neg.form == form[:3]]
    bins = np.histogram_bin_edges(sub[p], bins=15)
    counts = [np.histogram(sub.loc[sub.label == lab, p], bins)[0] for lab in LABELS]
    for i, n in enumerate(counts):
        savoy.hist(ax, n, bins, savoy.color(i))
    savoy.yticks(ax, 0, max(n.max() for n in counts), integer=True)
    savoy.xticks(ax, bins[0], bins[-1], n=5)
for i, name in enumerate(LABELS.values()):
    savoy.key(fig, name, savoy.color(i))
savoy.save(fig, lib.TOPIC_DIR / "figures/negation_test.png")
log.info("negation transfer AUROC, LR / MM: affirmative %.3f / %.3f · negated %.3f / %.3f",
         *neg_res.loc["affirmative"], *neg_res.loc["negated"])

# %% Save the truth directions
lib.save_cache(f"truth_probe_L{LAYER}", theta_lr=lr.coef_[0], theta_mm=d_mm,
               mu_pos=Xtr[y_train == 1].mean(0), mu_neg=Xtr[y_train == 0].mean(0))
cos = lr.coef_[0] @ d_mm / (np.linalg.norm(lr.coef_[0]) * np.linalg.norm(d_mm))
log.info("saved cache/truth_probe_L%d.pt (θ_lr, θ_mm, class means) · cos(θ_lr, θ_mm) = %.2f", LAYER, cos)
