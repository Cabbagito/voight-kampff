# Exp 1: what does an instructed-pairs "deception" probe actually read? The walk-through (what
# each step does, why, its knobs, the papers) is 01_what_the_probe_reads.md; this file is only
# the code, one cell per step, titled as in the doc. Each step logs its findings: in an esper
# session they land on the board and in the run report; in plain Python they print as "· ...".

# %% [markdown]
# ## Part 1 — Train the probe

# %% Load the model
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import lib

logging.basicConfig(level=logging.INFO, format="· %(message)s")   # no-op inside esper
log = logging.getLogger(__name__)

SEED = 0
LAYERS = [4, 8, 14, 20, 26, 32]   # blocks to sweep (Qwen3-4B has 36)
LAYER = 20                        # the sweep's peak; Parts 2-3 and Exps 2-3 use it

model, tok = lib.load_model_and_tokenizer()
log.info("%s · %d blocks · hidden %d", model.config._name_or_path.split("/")[-1],
         model.config.num_hidden_layers, model.config.hidden_size)

# %% Build the instructed pairs
train, test = lib.build_repe_pairs(lib.DATA_DIR / "facts_true_false.csv", tok, seed=SEED,
                                   n_train_pairs=256, n_test_pairs=64)
y_train = np.array([r["label"] for r in train])
y_test = np.array([r["label"] for r in test])
print(repr(test[6]["prompt"]), "\n", repr(test[7]["prompt"]))   # one prefix, both personas
log.info("train %d rows from %d statements · test %d rows from %d held-out statements · half honest",
         len(train), len({r["statement_id"] for r in train}), len(test), len({r["statement_id"] for r in test}))

# %% Read the activations
acts = lib.cached("sp_pairs", lambda: {
    **{f"tr_{L}": X for L, X in lib.collect_activations_layers(model, tok, [r["prompt"] for r in train], LAYERS).items()},
    **{f"te_{L}": X for L, X in lib.collect_activations_layers(model, tok, [r["prompt"] for r in test], LAYERS).items()},
})
log.info("last-token activations at blocks %s: %d train, %d test", LAYERS, len(acts["tr_20"]), len(acts["te_20"]))

# %% Fit the probes at each block
rows = []
for L in LAYERS:
    Xtr, Xte = acts[f"tr_{L}"].float().numpy(), acts[f"te_{L}"].float().numpy()
    lr = lib.fit_lr_probe(Xtr, y_train)
    rows.append({"block": L, "LR": lib.auroc(lr.decision_function(Xte), y_test),
                 "MM": lib.auroc(lib.mm_score(Xte, lib.mm_direction(Xtr, y_train)), y_test)})
sp_sweep = pd.DataFrame(rows).set_index("block")
display(sp_sweep.round(3))
sp_sweep.to_json(lib.CACHE_DIR / "sp_sweep.json")   # 02 overlays it on the truth sweep

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(sp_sweep.index, sp_sweep.LR, marker="o", label="logistic regression (LR)")
ax.plot(sp_sweep.index, sp_sweep.MM, marker="s", label="difference of means (MM)")
ax.axhline(0.5, color="gray", ls=":", label="chance")
ax.set(xlabel="block (layer, 0-35)", ylabel="test AUROC, honest vs untruthful", ylim=(0.45, 1.01),
       title="Instructed-pairs probe: held-out AUROC by block")
ax.legend(fontsize=8)
plt.tight_layout()
log.info("test AUROC by block, LR / MM: %s", " · ".join(f"{L}: {r.LR:.3f} / {r.MM:.3f}" for L, r in sp_sweep.iterrows()))

# %% Save the probe
Xtr = acts[f"tr_{LAYER}"].float().numpy()
sp_lr = lib.fit_lr_probe(Xtr, y_train)
sp_mm = lib.mm_direction(Xtr, y_train)
lib.save_cache(f"sp_probe_L{LAYER}", theta_lr=sp_lr.coef_[0], theta_mm=sp_mm,
               mu_pos=Xtr[y_train == 1].mean(0), mu_neg=Xtr[y_train == 0].mean(0))
log.info("block %d probe saved (cache/sp_probe_L%d.pt): high score = honest persona", LAYER, LAYER)

# %% [markdown]
# ## Part 2 — Switch the answers

# %% Build the four conditions
switch = pd.read_csv(lib.DATA_DIR / "switcheroo.csv")
PERSONA = {"T": "an honest", "D": "an untruthful"}
ANSWER = {"T": "honest_response", "D": "deceptive_response"}
cond_rows = [{"qid": r.um_id, "condition": c,
              "prompt": lib._make_full_prompt(PERSONA[c[0]], r.user_message, getattr(r, ANSWER[c[1]]))}
             for r in switch.itertuples() for c in ["TT", "TD", "DT", "DD"]]
print(repr(cond_rows[1]["prompt"]))   # TD: honest persona, lying answer
log.info("%d questions × 4 conditions (persona letter first, answer second) · answer words honest %.0f, lying %.0f",
         len(switch), switch.honest_response.str.split().str.len().mean(), switch.deceptive_response.str.split().str.len().mean())

# %% Score them with the probe
X_sw = lib.cached(f"switcheroo_L{LAYER}", lambda: {"X": lib.collect_activations(model, tok, [r["prompt"] for r in cond_rows], LAYER)})["X"].float().numpy()
sw = pd.DataFrame(cond_rows).drop(columns="prompt").assign(LR=sp_lr.decision_function(X_sw), MM=lib.mm_score(X_sw, sp_mm))
for col in ["LR", "MM"]:   # 0 = the honest pair (TT), 1 = the untruthful pair (DD)
    tt, dd = sw.loc[sw.condition == "TT", col].mean(), sw.loc[sw.condition == "DD", col].mean()
    sw[col + "_norm"] = (sw[col] - tt) / (dd - tt)
sw_norm = sw.groupby("condition")[["LR_norm", "MM_norm"]].mean().loc[["TT", "TD", "DT", "DD"]]
display(sw_norm.round(3))
log.info("normalised score (TT = 0, DD = 1), LR / MM: %s",
         " · ".join(f"{c} {r.LR_norm:.2f} / {r.MM_norm:.2f}" for c, r in sw_norm.iterrows()))

# %% Plot the paired traces
ORDER = ["TT", "TD", "DT", "DD"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, col, name in zip(axes, ["LR_norm", "MM_norm"], ["LR", "MM"]):
    piv = sw.pivot(index="qid", columns="condition", values=col)[ORDER]
    for _, row in piv.iterrows():
        ax.plot(ORDER, row.values, color="gray", alpha=0.15, marker="o", ms=3)
    ax.plot(ORDER, piv.mean().values, color="k", lw=2.5, marker="o", ms=8, label="mean of 50 questions")
    ax.set(title=f"{name} probe", xlabel="condition (persona, answer): T = honest, D = untruthful / lying")
    ax.legend(fontsize=8)
axes[0].set_ylabel("probe score, 0 = TT mean, 1 = DD mean")
fig.suptitle("Same probe, persona and answer swapped independently")
plt.tight_layout()

# %% [markdown]
# ## Part 3 — Hold the persona, change the topic

# %% Generate honest answers
MAX_NEW = 256
adj = pd.read_csv(lib.DATA_DIR / "deception_adjacent.csv")
gen_prompts = [lib._make_gen_prompt("an honest", um, think=False) for um in adj.user_message]
GEN_CSV = lib.CACHE_DIR / "decadj_answers.csv"
if not GEN_CSV.exists():
    adj.assign(response=lib.generate(model, tok, gen_prompts, max_new_tokens=MAX_NEW)).to_csv(GEN_CSV, index=False)
adj = pd.read_csv(GEN_CSV)
adj["words"] = adj.response.str.split().str.len()
for cat in ["deception_adjacent", "neutral"]:
    r = adj[adj.category == cat].iloc[0]
    print(f"[{cat}] {r.user_message}\n  {r.response[:300]}\n")
log.info("%d answers, greedy, thinking off · stray <think> tags %d · mean words: %s", len(adj),
         adj.response.str.contains("<think>").sum(),
         " · ".join(f"{c} {w:.0f}" for c, w in adj.groupby("category").words.mean().items()))

# %% Check refusals and length
REFUSAL_OPENINGS = ["i can't", "i cannot", "i won't", "i will not", "i'm not able", "i am not able", "i'm unable",
                    "i am unable", "as an ai", "i'm sorry, but", "i apologize, but", "i must decline"]
adj["refused"] = adj.response.str.strip().str.lower().apply(lambda t: t.startswith(tuple(REFUSAL_OPENINGS)))
for r in adj[adj.refused].itertuples():
    print(f"[{r.category}] {r.user_message} -> {r.response[:120]}")
log.info("refusals: %s · word count mean ± sd: %s",
         " · ".join(f"{c} {100 * x:.0f}%" for c, x in adj.groupby("category").refused.mean().items()),
         " · ".join(f"{c} {g.mean():.0f} ± {g.std():.0f}" for c, g in adj.groupby("category").words))

# %% Read the activations at the last token and averaged over the answer
full = [p + r for p, r in zip(gen_prompts, adj.response)]
n_resp = lib.count_response_tokens(tok, gen_prompts, full)
blob = lib.cached(f"decadj_L{LAYER}", lambda: {"last": lib.collect_activations(model, tok, full, LAYER),
                                                "mean": lib.collect_activations_mean(model, tok, full, n_resp, LAYER)})
X_last, X_mean = blob["last"].float().numpy(), blob["mean"].float().numpy()
log.info("block %d activations for %d answers: last token, and mean over each answer's tokens", LAYER, len(full))

# %% Score topic against topic
keep = ~adj.refused.values
is_adj = (adj.category == "deception_adjacent").values[keep].astype(int)
scores = pd.DataFrame({"category": adj.category.values,
                       "LR last": sp_lr.decision_function(X_last), "LR mean": sp_lr.decision_function(X_mean),
                       "MM last": lib.mm_score(X_last, sp_mm), "MM mean": lib.mm_score(X_mean, sp_mm)})[keep]
# high score = honest, so AUROC < 0.5 means deception-adjacent answers look more deceptive
spurious = pd.Series({c: 1 - lib.auroc(scores[c].values, is_adj) for c in scores.columns[1:]}, name="1 − AUROC")
display(spurious.round(3).to_frame())
log.info("how well the probe tells the topics apart (1 − AUROC; 0.5 = can't): %s",
         " · ".join(f"{c} {x:.2f}" for c, x in spurious.items()))

# %% Plot the score distributions
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
for ax, col in zip(axes.flat, spurious.index):
    for cat, c in [("deception_adjacent", "C3"), ("neutral", "C0")]:
        ax.hist(scores.loc[scores.category == cat, col], bins=15, alpha=0.6, color=c, label=cat.replace("_", "-"))
    ax.set(title=f"{col}  (1 − AUROC {spurious[col]:.2f})", xlabel="probe score (high = honest)", ylabel="answers")
    ax.legend(fontsize=8)
fig.suptitle("Honest persona, honest answers: deception-adjacent vs neutral questions")
plt.tight_layout()
