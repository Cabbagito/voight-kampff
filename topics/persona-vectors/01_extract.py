# Exp 1: extract and validate persona vectors. The walk-through (what each step does, why,
# its knobs, what the paper does) is 01_extract.md; this file is only the code, one cell per
# step, titled as in the doc. Each step logs its findings: in an esper session they land on
# the board and in the run report; in plain Python they print as "· ...".

# %% [markdown]
# ## Part 1 — Make training answers

# %% Load the model and the trait
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

import judge as J
import pv

logging.basicConfig(level=logging.INFO, format="· %(message)s")   # no-op inside esper
log = logging.getLogger(__name__)

TRAIT = "dishonest"   # "dishonest" | "pirate" | "mystic": run once per trait
SMOKE = False         # True: tiny mechanics-only pass, numbers meaningless
SEED = 0

if SMOKE:
    MODEL_ID, N_QUESTIONS, N_WORDINGS, N_ANSWERS, MAX_NEW = None, 4, 2, 2, 64
else:
    MODEL_ID, N_QUESTIONS, N_WORDINGS, N_ANSWERS, MAX_NEW = None, 20, 5, 10, 500
    if TRAIT == "dishonest":
        N_ANSWERS = 30   # Qwen3-4B lies in only part of its POS answers
GEN = dict(max_new_tokens=MAX_NEW, batch_size=32, seed=SEED)
TAG = f"{TRAIT}{'_smoke' if SMOKE else ''}"   # cache-file prefix

trait = pv.load_trait(TRAIT)
judge = J.Judge()
model, tok, device = pv.load_model(MODEL_ID)
log.info("%s · %d blocks · judge %s · trait %s", model.config._name_or_path.split("/")[-1],
         pv.n_blocks(model), judge.model, TRAIT)

# %% Show what the model sees
print(pv.chat_prompt(tok, pv.system_prompt(trait, 0, "POS"), trait["questions_extract"][0],
                     pv.fewshot_turns(trait, "POS")))
print("NEG system prompt:", pv.system_prompt(trait, 0, "NEG"))

# %% Generate training answers
table = pv.rollout_table(trait, "extract", n_answers=N_ANSWERS, n_questions=N_QUESTIONS, n_wordings=N_WORDINGS)
ROLL_KEY = pv.key(table.to_json(), MAX_NEW, SEED)   # any change to prompts or questions: fresh cache
roll = pv.cached_df(f"rollouts_{TAG}_{ROLL_KEY}", lambda: pv.generate_rollouts(model, tok, table, device=device, **GEN))
log.info("%d answers · mean %.0f tokens · %.1f%% cut off at %d", len(roll), roll.n_tokens.mean(),
         100 * roll.truncated.mean(), MAX_NEW)

# %% [markdown]
# ## Part 2 — Score them

# %% Judge the answers
scored = judge.score_df(roll, trait["eval_prompt"])
hi = scored.assign(hi=scored.trait_score >= 60).groupby("condition").hi.mean()
log.info("trait ≥ 60: POS %.0f%% · NEG %.0f%% · BASE %.0f%% · mean coherence %.0f",
         100 * hi.POS, 100 * hi.NEG, 100 * hi.BASE, scored.coherence.mean())

# %% Check the judge against Opus
if SMOKE or TRAIT == "dishonest":
    check = scored.sample(min(100, len(scored)), random_state=SEED)
    ref = J.Judge(model="claude-opus-5-5").score_df(check, trait["eval_prompt"])
    both = check[["trait_score", "coherence"]].join(ref[["trait_score", "coherence"]], rsuffix="_opus").dropna()
    agree = ((both.trait_score > 50) == (both.trait_score_opus > 50)).mean()
    coh_cut = ((both.coherence >= 70) == (both.coherence_opus >= 70)).mean()
    log.info("Haiku vs Opus: %.0f%% agree at 50 · trait rank %.2f · coherence rank %.2f · %.0f%% agree at coherence 70",
             100 * agree, both.trait_score.corr(both.trait_score_opus, method="spearman"),
             both.coherence.corr(both.coherence_opus, method="spearman"), 100 * coh_cut)
    display(both.assign(gap=(both.trait_score - both.trait_score_opus).abs())
                .sort_values("gap", ascending=False).head(5))   # biggest disagreements: read these

# %% Plot the scores
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, col in zip(axes, ["trait_score", "coherence"]):
    for cond, c in [("POS", "C3"), ("NEG", "C0"), ("BASE", "C7")]:
        ax.hist(scored.loc[scored.condition == cond, col].dropna(), bins=np.arange(0, 105, 5), alpha=.6, color=c, label=cond)
    ax.set(title={"trait_score": f"{TRAIT} score", "coherence": "coherence"}[col],
           xlabel="judge's score (0-100)", ylabel="number of answers"); ax.legend()
fig.suptitle(f"{TRAIT}: judge scores by prompt condition")
plt.tight_layout()
by_wording = scored[scored.condition == "POS"].groupby("wording").trait_score.apply(lambda s: (s >= 60).mean())
log.info("POS trait ≥ 60 by wording: %s", " · ".join(f"{w}: {100 * x:.0f}%" for w, x in by_wording.items()))

# %% [markdown]
# ## Part 3 — Turn them into vectors

# %% Keep the clean answers
drop = [d["qid"] for d in trait.get("drop_extract", [])]   # reasons in the trait file
kept = pv.filter_rollouts(scored[~scored.qid.isin(drop)])
n_kept = kept.condition.value_counts()
log.info("kept POS %d · NEG %d · BASE %d · POS covers %d of %d questions%s", n_kept.get("POS", 0), n_kept.get("NEG", 0),
         n_kept.get("BASE", 0), kept[kept.condition == "POS"].qid.nunique(), kept.qid.nunique(),
         f" · dropped questions {drop}" if drop else "")
display(kept.groupby("qid").condition.value_counts().unstack(fill_value=0).T)

# %% Read the activations
REDUCE_VERSION = "v1"   # bump when pv.reduce_tokens changes
FIRST_K = 40
REDUCERS = {"full": pv.reduce_tokens, f"first{FIRST_K}": lambda h, n: pv.reduce_tokens(h, n, first_k=FIRST_K)}
acts_by = {}
for red, fn in REDUCERS.items():
    name = f"acts_{TAG}_{ROLL_KEY}_{REDUCE_VERSION}_{red}"
    if (pv.CACHE_DIR / f"{name}.pt").exists():
        acts_by[red] = pv.load_tensor(name)
    else:
        acts_by[red] = pv.collect_reduced(model, tok, scored.prompt.tolist(), scored.response.tolist(), fn, device=device)
        pv.save_tensor(acts_by[red], name)
REDUCE = {"dishonest": "first40", "pirate": "first40", "mystic": "full"}[TRAIT]   # per trait, decided 2026-10-17:
# separation on unseen answers and length contamination, see 01_extract.md "Read the activations"
acts_all = acts_by[REDUCE]
log.info("activations for %d answers × %d blocks, two reductions; using %s", *acts_all.shape[:2], REDUCE)

# %% Build the persona vectors
main = pv.contrast(kept, "NEG")   # POS − NEG, the paper's vector
DROP_EXTREMES = {"dishonest": True}.get(TRAIT, False)   # 2026-10-29, checks/extremes.py: a few answers with one dimension
# far off (> 10 robust SDs) bent dishonest's vector at blocks 16-23 (cos 0.75 at 16); pirate's and mystic's don't move (cos 1.00)
if DROP_EXTREMES:
    main, dropped = pv.drop_extremes(main, acts_all)
    log.info("dropped %d answers with extreme values (%s)", len(dropped), dropped.condition.value_counts().to_dict())
n = main.condition.value_counts().to_dict()
assert n.get("POS") and n.get("NEG"), f"FILTER left {n}: no POS or no NEG answers to average"
idx = torch.tensor(main.index.to_numpy())
lab = main.reset_index(drop=True)
MATCH = True    # decided 2026-10-17: subtract within each question; only changes dishonest (pirate/mystic identical)
V = pv.persona_vectors(acts_all[idx], lab, match_questions=MATCH)   # (n_blocks, d), unit length per block

Vn = V / V.norm(dim=-1, keepdim=True)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow((Vn @ Vn.T).numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set(title=f"{TRAIT}: how alike are the vectors of different blocks?", xlabel="block (layer, 0-35)",
       ylabel="block (layer, 0-35)"); plt.colorbar(im, ax=ax, label="cosine")
plt.tight_layout()
log.info("%d POS − NEG vectors (%s, %s)", V.shape[0], REDUCE, "matched" if MATCH else "pooled")

# %% Check: whole answer vs first 40 tokens
V_by = {r: pv.persona_vectors(a[idx], lab) for r, a in acts_by.items()}
cos_tok = torch.nn.functional.cosine_similarity(*V_by.values(), dim=-1)
half = lab.groupby("condition").sample(frac=.5, random_state=SEED).index.to_numpy()   # noise reference
m = np.zeros(len(lab), bool); m[half] = True
a = acts_all[idx]
cos_half = torch.nn.functional.cosine_similarity(
    pv.persona_vectors(a[torch.from_numpy(m)], lab[m].reset_index(drop=True)),
    pv.persona_vectors(a[torch.from_numpy(~m)], lab[~m].reset_index(drop=True)), dim=-1)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(cos_tok, marker="o", label=" vs ".join(V_by))
ax.plot(cos_half, ls="--", color="k", label=f"{REDUCE}: half vs half (noise)")
ax.set(ylim=(-0.05, 1.05), xlabel="block (layer, 0-35)", ylabel="cosine between the two vectors",
       title=f"{TRAIT}: whole answer vs first 40 tokens"); ax.legend()
plt.tight_layout()
log.info("cos at blocks 6·12·18·24·30: %s (noise %s)", " ".join(f"{cos_tok[L]:.2f}" for L in (6, 12, 18, 24, 30)),
         " ".join(f"{cos_half[L]:.2f}" for L in (6, 12, 18, 24, 30)))

# %% Check: pooled vs matched by question
cos_topic = torch.nn.functional.cosine_similarity(
    pv.persona_vectors(acts_all[idx], lab), pv.persona_vectors(acts_all[idx], lab, match_questions=True), dim=-1)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(cos_topic, marker="o", label="pooled vs matched")
ax.plot(cos_half, ls="--", color="k", label=f"{REDUCE}: half vs half (noise)")
ax.set(ylim=(-0.05, 1.05), xlabel="block (layer, 0-35)", ylabel="cosine between the two vectors",
       title=f"{TRAIT}: pooled vs matched by question"); ax.legend()
plt.tight_layout()
log.info("cos at blocks 6·12·18·24·30: %s (noise %s)", " ".join(f"{cos_topic[L]:.2f}" for L in (6, 12, 18, 24, 30)),
         " ".join(f"{cos_half[L]:.2f}" for L in (6, 12, 18, 24, 30)))

# %% Compare with the no-prompt baseline
alt = pv.contrast(kept, "BASE")   # POS − BASE (BASE relabelled NEG)
if DROP_EXTREMES:
    alt, _ = pv.drop_extremes(alt, acts_all)
assert (alt.condition == "NEG").any(), "FILTER left no BASE answers"
V_base = pv.persona_vectors(acts_all[torch.tensor(alt.index.to_numpy())], alt.reset_index(drop=True), match_questions=MATCH)
cos_nb = torch.nn.functional.cosine_similarity(V, V_base, dim=-1)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(cos_nb, marker="o"); ax.axhline(0, color="k", lw=.5)
ax.set(ylim=(-1.05, 1.05), xlabel="block (layer, 0-35)", ylabel="cosine",
       title=f"{TRAIT}: cos(POS − NEG, POS − BASE)")
plt.tight_layout()
log.info("cos(POS − NEG, POS − BASE) at blocks 6·12·18·24·30: %s", " ".join(f"{cos_nb[L]:.2f}" for L in (6, 12, 18, 24, 30)))

# %% [markdown]
# ## Part 4 — Check the vector works

# %% Pilot: how big a push?
STEER_POS = "response"   # B, the paper's: push the newest position of every pass (the last prompt token, then each
                         # new answer token) and keep the pushed values in the cache. A = "all": every prompt token too.
PILOT_LAYER = 3 if SMOKE else 24
PILOT_ROUNDS = {   # α as a fraction of the residual stream's length there: (fractions, answers per question)
    "coarse": ([0.05, 0.1, 0.2, 0.4, 0.8, 1.6], 1),
    "zoom": ([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5], 2),   # 2026-10-17, where the coarse round turned
}
eval_qs = trait["questions_eval"][:N_QUESTIONS]
base_ans = kept[kept.condition == "BASE"].sample(min(40, (kept.condition == "BASE").sum()), random_state=0)
h_lens = pv.residual_norms(model, tok, base_ans.prompt.tolist(), base_ans.response.tolist(), device=device)   # every block
h_len = float(h_lens[PILOT_LAYER])
gap_len = float(pv.persona_vectors(acts_all[idx], lab, match_questions=MATCH, normalize=False)[PILOT_LAYER].norm())
rounds = []
for rnd, (fracs, n) in PILOT_ROUNDS.items():
    coeffs = [round(f * h_len, 1) for f in fracs]
    df = pv.cached_df(f"pilot_{TAG}_{pv.key(V, PILOT_LAYER, coeffs, n, STEER_POS, MAX_NEW)}",
                      lambda: pv.steering_sweep(model, tok, eval_qs, V, [PILOT_LAYER], coeffs, n_samples=n,
                                                positions=STEER_POS, device=device, **{**GEN, "batch_size": 40}))
    rounds.append(df.assign(round=rnd))
pilot = judge.score_df(pd.concat(rounds, ignore_index=True), trait["eval_prompt"])
pilot["frac"] = (pilot.coeff / h_len).round(3)
steered = pilot[pilot.layer >= 0]
unsteered = pilot[pilot.layer == -1][["trait_score", "coherence"]].mean()
fig, ax = plt.subplots(figsize=(7, 4))
for col, c in [("trait_score", "C3"), ("coherence", "C0")]:
    for rnd, ls in [("coarse", ":"), ("zoom", "-")]:
        p = steered[steered["round"] == rnd].groupby("frac")[col].agg(["mean", "sem"])
        ax.errorbar(p.index, p["mean"], yerr=p["sem"], marker="o", ls=ls, color=c, capsize=3, label=f"{col}, {rnd} round")
    ax.axhline(unsteered[col], color=c, ls="--", lw=1, label=f"{col}, unsteered")
ax.axvline(gap_len / h_len, color="grey", ls=":", label=f"one POS − NEG gap ({gap_len / h_len:.2f})")
ax.set(xscale="log", xlabel=f"α / residual stream length at block {PILOT_LAYER} (length {h_len:.0f})",
       ylabel="mean judge score (0-100)", ylim=(-2, 102),
       title=f"{TRAIT}: how big a push does the trait need? (block {PILOT_LAYER}, steering {STEER_POS})")
ax.legend(fontsize=7); plt.tight_layout()
z = steered[steered["round"] == list(PILOT_ROUNDS)[-1]].groupby("frac")[["trait_score", "coherence"]].mean()
log.info("pilot at block %d, %s round, α / stream length → trait/coherence: unsteered %.0f/%.0f · %s", PILOT_LAYER,
         list(PILOT_ROUNDS)[-1], *unsteered, " · ".join(f"{f:.2g}: {r.trait_score:.0f}/{r.coherence:.0f}" for f, r in z.iterrows()))

# %% Steer at each layer
N_SWEEP = 1 if SMOKE else 2
STEER_SAMPLING = dict(top_k=20, repetition_penalty=1.05)   # the paper's, via Qwen2.5's generation defaults (checks/diagnose.py)
SWEEPS = {   # positions → (strengths, as fractions of each block's stream length; blocks)
    "response": ([0.2, 0.267, 0.333, 0.4], list(range(6, 33, 3))),   # B, the paper's (Peter, 2026-10-17)
    "all": ([0.2, 0.267, 0.333, 0.4], list(range(9, 28, 3))),        # A, same strengths as B (Peter, 2026-10-22)
    "clean": ([0.5, 0.7, 0.9, 1.1], list(range(9, 28, 3))),          # C, ~2× B: from checks/c_pilot.py (2026-10-22)
}
if SMOKE:
    SWEEPS = {pos: ([0.05, 0.1], [1, 3]) for pos in SWEEPS}
runs = []
for pos, (fracs, layers) in SWEEPS.items():
    if fracs:
        runs.append(pv.cached_df(f"sweep_{TAG}_{pv.key(V, layers, fracs, h_lens, N_SWEEP, pos, STEER_SAMPLING, MAX_NEW)}",
                                 lambda: pv.steering_sweep(model, tok, eval_qs, V, layers, fracs, scale=h_lens, n_samples=N_SWEEP,
                                                           include_baseline=pos == "response", positions=pos, device=device,
                                                           **{**GEN, "batch_size": 40, **STEER_SAMPLING})))
sweep = judge.score_df(pd.concat(runs, ignore_index=True), trait["eval_prompt"])
sweep["prod"] = sweep.trait_score * sweep.coherence / 100   # per answer: trait that survives with coherence
best = sweep[sweep.layer >= 0].groupby(["positions", "layer", "frac"])["prod"].mean()
log.info("best block by mean trait × coherence / 100 (unsteered %.0f): %s", sweep[sweep.layer == -1]["prod"].mean(),
         " · ".join(f"{pos}: block {L} at {f:g} ({best[(pos, L, f)]:.0f})" for pos, (_, L, f) in best.groupby("positions").idxmax().items()))

# %% Plot the layer sweep
base = sweep[sweep.layer == -1][["trait_score", "coherence", "prod"]].mean()
g = sweep[sweep.layer >= 0].groupby(["positions", "frac", "layer"])[["trait_score", "coherence", "prod"]].mean().reset_index()
NAMES = {"all": "A", "response": "B", "clean": "C"}
fig, axes = plt.subplots(g.positions.nunique(), 3, figsize=(15, 3.3 * g.positions.nunique()), sharex=True, squeeze=False)
for row, (pos, gp) in enumerate(g.groupby("positions", sort=False)):
    for ax, col, name in zip(axes[row], ["trait_score", "coherence", "prod"], ["trait score", "coherence", "trait × coherence / 100"]):
        for c, d in gp.groupby("frac"):
            ax.plot(d.layer, d[col], marker="o", label=f"α = {c:g} × stream length")
        ax.axhline(base[col], color="k", ls="--", lw=1, label="unsteered")
        ax.set(ylabel=f"{NAMES[pos]}: mean {name}", ylim=(-2, 102),
               title=f"{TRAIT}, {NAMES[pos]}: {name} when steering at each block")
    axes[row, 0].legend(fontsize=7)
for ax in axes[-1]:
    ax.set_xlabel("block steered (layer, 0-35)")
plt.tight_layout()

# %% Pick the layer
PICK = {"dishonest": (18, "all"), "pirate": (18, "all"), "mystic": (18, "all")}   # Peter, 2026-10-22, from the A/B/C sweeps;
# dishonest re-picked 2026-10-30 (was B at 18) after its vector was cleaned of extreme answers: A at 18 went 17 → 28
LAYER, STEER_POS = (3, "response") if SMOKE else PICK[TRAIT]   # block, and where the push goes (A = "all", B = "response")
v = V[LAYER]
log.info("steering at block %d with %s; monitoring at block %d", LAYER, NAMES[STEER_POS], LAYER)

# %% Steer at different strengths
DOSE_FRACS = [0.05, 0.1] if SMOKE else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]   # × stream length: past the sweeps' 0.4 edge
N_DOSE = 1 if SMOKE else 3
assert DOSE_FRACS, "set DOSE_FRACS"
dose = pv.cached_df(f"dose_{TAG}_{pv.key(v, LAYER, DOSE_FRACS, float(h_lens[LAYER]), N_DOSE, STEER_POS, STEER_SAMPLING, MAX_NEW)}",
                    lambda: pv.steering_sweep(model, tok, eval_qs, V, [LAYER], DOSE_FRACS, scale=h_lens,
                                              n_samples=N_DOSE, positions=STEER_POS, device=device,
                                              **{**GEN, "batch_size": 60, **STEER_SAMPLING}))
dose = judge.score_df(dose, trait["eval_prompt"])
d = dose.groupby("frac")[["trait_score", "coherence"]].agg(["mean", "sem"])
fig, ax = plt.subplots(figsize=(6, 3.5))
for col, c in [("trait_score", "C3"), ("coherence", "C0")]:
    ax.errorbar(d.index, d[(col, "mean")], yerr=d[(col, "sem")], marker="o", color=c, label=col, capsize=3)
ax.set(xlabel=f"α / stream length at block {LAYER}", ylabel="mean judge score (0-100)",
       title=f"{TRAIT}: steering strength vs trait and coherence", ylim=(-2, 102)); ax.legend()
plt.tight_layout()
log.info("trait / coherence by α / stream length: %s", " · ".join(f"{c:g}: {r[('trait_score', 'mean')]:.0f}/{r[('coherence', 'mean')]:.0f}"
                                                    for c, r in d.iterrows()))
for _, r in dose[(dose.coherence >= 50) & (dose.coeff > 0)].sort_values(["coeff", "trait_score"]).tail(2).iterrows():
    print(f"--- {r.frac:g} × stream (α {r.coeff:.0f}) | trait {r.trait_score} | coherence {r.coherence}\nQ: {r.question}\nA: {r.response[:600]}\n")

# %% Repeat with the baseline vector
COS_SKIP = 0.9   # scope trim: only if POS − NEG and POS − BASE disagree at LAYER
run_repeat = SMOKE or float(cos_nb[LAYER]) < COS_SKIP
log.info("cos at block %d = %.2f: POS − BASE repeat %s", LAYER, float(cos_nb[LAYER]), "runs" if run_repeat else "skipped")
if run_repeat:
    dose_b = pv.cached_df(f"doseB_{TAG}_{pv.key(V_base[LAYER], LAYER, DOSE_FRACS, float(h_lens[LAYER]), N_DOSE, STEER_POS, STEER_SAMPLING, MAX_NEW)}",
                          lambda: pv.steering_sweep(model, tok, eval_qs, V_base, [LAYER], DOSE_FRACS, scale=h_lens, n_samples=N_DOSE,
                                                    include_baseline=False, positions=STEER_POS, device=device,
                                                    **{**GEN, **STEER_SAMPLING}))
    dose_b = judge.score_df(dose_b, trait["eval_prompt"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True)
    unsteered = dose[dose.layer == -1]
    for df_, label, ls in [(dose, "POS − NEG", "-"), (pd.concat([unsteered, dose_b]), "POS − BASE", "--")]:
        g = df_.groupby("frac")[["trait_score", "coherence"]].agg(["mean", "sem"])
        for ax, col in zip(axes, ["trait_score", "coherence"]):
            ax.errorbar(g.index, g[(col, "mean")], yerr=g[(col, "sem")], marker="o", ls=ls, capsize=3, label=label)
    for ax, col in zip(axes, ["trait_score", "coherence"]):
        ax.set(title=f"{TRAIT}: {col}, POS − NEG vs POS − BASE", xlabel=f"α / stream length at block {LAYER}",
               ylabel="mean judge score (0-100)", ylim=(-2, 102)); ax.legend()
    plt.tight_layout()

# %% Monitor: read the vector before the answer
r = None
if pv.has_ladder(TRAIT):
    ladder = pv.load_ladder(TRAIT)[:: 7 if SMOKE else 1]
    N_MON = 1 if SMOKE else 4
    mon = pv.cached_df(f"monitor_{TAG}_{pv.key(v, LAYER, ladder, N_MON, MAX_NEW)}",
                       lambda: pv.monitoring_run(model, tok, ladder, eval_qs, LAYER, v, n_samples=N_MON, device=device, **GEN))
    mon = judge.score_df(mon, trait["eval_prompt"], coherence=False)
    agg, r = pv.monitor_correlations(mon)
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(agg.projection, agg.trait_score, c=agg.level, cmap="viridis", s=18)
    plt.colorbar(sc, label="ladder level (0 = discourages trait)")
    ax.set_xlabel(f"projection onto v̂, block {LAYER}, last prompt token"); ax.set_ylabel(f"mean {TRAIT} score")
    ax.set_title(f"overall r = {r['overall_r']:.2f}   within-level r = {r['within_level_r']:.2f}")
    plt.tight_layout()
    log.info("monitor before the answer: overall r %.2f · within-level r %.2f", r["overall_r"], r["within_level_r"])
else:
    log.info("no ladder for %s: monitoring skipped", TRAIT)

# %% Monitor: read it during the answer
if pv.has_ladder(TRAIT):
    import warnings
    tname = f"traj_{TAG}_{pv.key(v, LAYER, ladder, N_MON, MAX_NEW)}"
    if (pv.CACHE_DIR / f"{tname}.pt").exists():
        traj = pv.load_tensor(tname)
    else:
        mon_prompts = [pv.chat_prompt(tok, s, q) for s, q in zip(mon.system, mon.question)]
        traj = pv.token_projections(model, tok, mon_prompts, mon.response.tolist(), LAYER, v, device=device)
        pv.save_tensor(traj, tname)   # per answer: [last prompt token, answer tokens...]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    MAX_POS = 150
    for lv, g in mon.groupby("level"):
        M = np.full((len(g), MAX_POS + 1), np.nan)
        for j, i in enumerate(g.index):
            t = traj[i][: MAX_POS + 1].numpy()
            M[j, : len(t)] = t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # positions past every answer's end
            axes[0].plot(np.nanmean(M, 0), color=plt.cm.viridis(lv / 7), lw=1, label=f"level {lv}")
    axes[0].axvline(0.5, color="k", lw=.5, ls=":")
    axes[0].set_xlabel("position (0 = last prompt token, then answer tokens)")
    axes[0].set_ylabel(f"projection onto v̂, block {LAYER}"); axes[0].legend(fontsize=7, ncol=2)
    READ = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    curve = []
    for k in READ:
        _, rk = pv.monitor_correlations(mon.assign(p=[float(t[: k + 1].mean()) for t in traj]), proj_col="p")
        curve.append(dict(read=k, overall=rk["overall_r"], within=rk["within_level_r"]))
    curve = pd.DataFrame(curve)
    _, r_ans = pv.monitor_correlations(mon.assign(p=[float(t[1:].mean()) if len(t) > 1 else np.nan for t in traj]), proj_col="p")
    for col, c in [("overall", "C0"), ("within", "C1")]:
        axes[1].plot(curve.read, curve[col], marker="o", color=c, label=f"{col} r, running mean")
        axes[1].axhline(r_ans[f"{col}_r" if col == "overall" else "within_level_r"], color=c, ls="--", lw=1)
    axes[1].set_xscale("symlog", linthresh=1); axes[1].set_xlim(left=-0.2)
    axes[1].set_xlabel("answer tokens read (0 = before the answer)"); axes[1].set_ylabel("Pearson r with judged dishonesty")
    axes[1].legend(fontsize=8); plt.tight_layout()
    log.info("within-level r after 0 · 8 · 64 answer tokens: %s",
             " · ".join(f"{curve.set_index('read').within[k]:.2f}" for k in (0, 8, 64)))

# %% Save the vectors
if not SMOKE:
    best_frac = dose[dose.layer >= 0].assign(prod=dose.trait_score * dose.coherence / 100).groupby("frac")["prod"].mean().idxmax()
    p = pv.save_persona(TRAIT, V, LAYER, reduce_tag=f"{REDUCE_VERSION}_{REDUCE}", n_kept=len(main), vectors_base=V_base.cpu(),
                        judge=judge.model, monitor=r, model=model.config._name_or_path, positions=STEER_POS,
                        h_lens=h_lens.tolist(), best_frac=float(best_frac), sampling=STEER_SAMPLING)
    log.info("saved %s", p)
log.info("judge spend this session ≈ $%.2f", judge.cost())
