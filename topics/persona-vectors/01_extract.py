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
    ax.set_title(col); ax.legend()
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
n = main.condition.value_counts().to_dict()
assert n.get("POS") and n.get("NEG"), f"FILTER left {n}: no POS or no NEG answers to average"
idx = torch.tensor(main.index.to_numpy())
lab = main.reset_index(drop=True)
MATCH = False   # match by question: open decision, see the topic check
V = pv.persona_vectors(acts_all[idx], lab, match_questions=MATCH)   # (n_blocks, d), unit length per block

Vn = V / V.norm(dim=-1, keepdim=True)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow((Vn @ Vn.T).numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_title("cos between blocks' vectors"); ax.set_xlabel("block"); plt.colorbar(im, ax=ax)
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
ax.set_ylim(-0.05, 1.05); ax.set_xlabel("block"); ax.set_title("whole answer vs first 40 tokens"); ax.legend()
plt.tight_layout()
log.info("cos at blocks 6·12·18·24·30: %s (noise %s)", " ".join(f"{cos_tok[L]:.2f}" for L in (6, 12, 18, 24, 30)),
         " ".join(f"{cos_half[L]:.2f}" for L in (6, 12, 18, 24, 30)))

# %% Check: pooled vs matched by question
cos_topic = torch.nn.functional.cosine_similarity(
    pv.persona_vectors(acts_all[idx], lab), pv.persona_vectors(acts_all[idx], lab, match_questions=True), dim=-1)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(cos_topic, marker="o", label="pooled vs matched")
ax.plot(cos_half, ls="--", color="k", label=f"{REDUCE}: half vs half (noise)")
ax.set_ylim(-0.05, 1.05); ax.set_xlabel("block"); ax.set_title("pooled vs matched by question"); ax.legend()
plt.tight_layout()
log.info("cos at blocks 6·12·18·24·30: %s (noise %s)", " ".join(f"{cos_topic[L]:.2f}" for L in (6, 12, 18, 24, 30)),
         " ".join(f"{cos_half[L]:.2f}" for L in (6, 12, 18, 24, 30)))

# %% Compare with the no-prompt baseline
alt = pv.contrast(kept, "BASE")   # POS − BASE (BASE relabelled NEG)
assert (alt.condition == "NEG").any(), "FILTER left no BASE answers"
V_base = pv.persona_vectors(acts_all[torch.tensor(alt.index.to_numpy())], alt.reset_index(drop=True), match_questions=MATCH)
cos_nb = torch.nn.functional.cosine_similarity(V, V_base, dim=-1)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(cos_nb, marker="o"); ax.axhline(0, color="k", lw=.5)
ax.set_ylim(-1.05, 1.05); ax.set_xlabel("block"); ax.set_title("cos(POS − NEG, POS − BASE)")
plt.tight_layout()
log.info("cos(POS − NEG, POS − BASE) at blocks 6·12·18·24·30: %s", " ".join(f"{cos_nb[L]:.2f}" for L in (6, 12, 18, 24, 30)))

# %% [markdown]
# ## Part 4 — Check the vector works

# %% Steer at each layer
SWEEP_LAYERS = [1, 3] if SMOKE else list(range(6, 33, 3))
SWEEP_COEFFS = [4.0, 16.0] if SMOKE else None   # ~3 values, in the units of V: open decision
N_SWEEP = 1 if SMOKE else 2
assert SWEEP_COEFFS, "set SWEEP_COEFFS"
eval_qs = trait["questions_eval"][:N_QUESTIONS]
sweep = pv.cached_df(f"sweep_{TAG}_{pv.key(V, SWEEP_LAYERS, SWEEP_COEFFS, N_SWEEP, MAX_NEW)}",
                     lambda: pv.steering_sweep(model, tok, eval_qs, V, SWEEP_LAYERS, SWEEP_COEFFS,
                                               n_samples=N_SWEEP, device=device, **GEN))
sweep = judge.score_df(sweep, trait["eval_prompt"])
best = sweep[sweep.layer >= 0].groupby(["coeff", "layer"]).trait_score.mean().groupby("coeff").idxmax()
log.info("highest trait score per coefficient at block: %s", " · ".join(f"α {c:g} → {l}" for c, (_, l) in best.items()))

# %% Plot the layer sweep
base = sweep[sweep.layer == -1][["trait_score", "coherence"]].mean()
g = sweep[sweep.layer >= 0].groupby(["coeff", "layer"])[["trait_score", "coherence"]].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True)
for ax, col in zip(axes, ["trait_score", "coherence"]):
    for c, d in g.groupby("coeff"):
        ax.plot(d.layer, d[col], marker="o", label=f"α = {c:g}")
    ax.axhline(base[col], color="k", ls="--", lw=1, label="unsteered")
    ax.set_xlabel("block"); ax.set_title(col); ax.set_ylim(-2, 102)
axes[0].legend(); plt.tight_layout()

# %% Pick the layer
LAYER = 3 if SMOKE else None   # Peter's call from the sweep, by the rule in 01_extract.md
assert LAYER is not None
v = V[LAYER]
log.info("steering and monitoring at block %d", LAYER)

# %% Steer at different strengths
DOSE_COEFFS = [4.0, 16.0] if SMOKE else None   # ~6 values from "no visible effect" to "incoherent"
N_DOSE = 1 if SMOKE else 3
assert DOSE_COEFFS, "set DOSE_COEFFS"
dose = pv.cached_df(f"dose_{TAG}_{pv.key(v, LAYER, DOSE_COEFFS, N_DOSE, MAX_NEW)}",
                    lambda: pv.steering_sweep(model, tok, eval_qs, V, [LAYER], DOSE_COEFFS,
                                              n_samples=N_DOSE, device=device, **GEN))
dose = judge.score_df(dose, trait["eval_prompt"])
d = dose.groupby("coeff")[["trait_score", "coherence"]].agg(["mean", "sem"])
fig, ax = plt.subplots(figsize=(6, 3.5))
for col, c in [("trait_score", "C3"), ("coherence", "C0")]:
    ax.errorbar(d.index, d[(col, "mean")], yerr=d[(col, "sem")], marker="o", color=c, label=col, capsize=3)
ax.set_xlabel(f"α at block {LAYER}"); ax.set_ylim(-2, 102); ax.legend()
plt.tight_layout()
log.info("trait / coherence by α: %s", " · ".join(f"{c:g}: {r[('trait_score', 'mean')]:.0f}/{r[('coherence', 'mean')]:.0f}"
                                                    for c, r in d.iterrows()))
for _, r in dose[(dose.coherence >= 50) & (dose.coeff > 0)].sort_values(["coeff", "trait_score"]).tail(2).iterrows():
    print(f"--- α {r.coeff:g} | trait {r.trait_score} | coherence {r.coherence}\nQ: {r.question}\nA: {r.response[:600]}\n")

# %% Repeat with the baseline vector
COS_SKIP = 0.9   # scope trim: only if POS − NEG and POS − BASE disagree at LAYER
run_repeat = SMOKE or float(cos_nb[LAYER]) < COS_SKIP
log.info("cos at block %d = %.2f: POS − BASE repeat %s", LAYER, float(cos_nb[LAYER]), "runs" if run_repeat else "skipped")
if run_repeat:
    dose_b = pv.cached_df(f"doseB_{TAG}_{pv.key(V_base[LAYER], LAYER, DOSE_COEFFS, N_DOSE, MAX_NEW)}",
                          lambda: pv.steering_sweep(model, tok, eval_qs, V_base, [LAYER], DOSE_COEFFS, n_samples=N_DOSE,
                                                    include_baseline=False, device=device, **GEN))
    dose_b = judge.score_df(dose_b, trait["eval_prompt"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True)
    unsteered = dose[dose.layer == -1]
    for df_, label, ls in [(dose, "POS − NEG", "-"), (pd.concat([unsteered, dose_b]), "POS − BASE", "--")]:
        g = df_.groupby("coeff")[["trait_score", "coherence"]].agg(["mean", "sem"])
        for ax, col in zip(axes, ["trait_score", "coherence"]):
            ax.errorbar(g.index, g[(col, "mean")], yerr=g[(col, "sem")], marker="o", ls=ls, capsize=3, label=label)
    for ax, col in zip(axes, ["trait_score", "coherence"]):
        ax.set_title(col); ax.set_xlabel(f"α at block {LAYER}"); ax.set_ylim(-2, 102); ax.legend()
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
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlabel("answer tokens read (0 = before the answer)"); axes[1].set_ylabel("Pearson r with judged dishonesty")
    axes[1].legend(fontsize=8); plt.tight_layout()
    log.info("within-level r after 0 · 8 · 64 answer tokens: %s",
             " · ".join(f"{curve.set_index('read').within[k]:.2f}" for k in (0, 8, 64)))

# %% Save the vectors
if not SMOKE:
    p = pv.save_persona(TRAIT, V, LAYER, reduce_tag=f"{REDUCE_VERSION}_{REDUCE}", n_kept=len(main), vectors_base=V_base.cpu(),
                        judge=judge.model, monitor=r, model=model.config._name_or_path)
    log.info("saved %s", p)
log.info("judge spend this session ≈ $%.2f", judge.cost())
