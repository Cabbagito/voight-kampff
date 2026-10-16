# %% [markdown]
# # 01 — Extract and validate persona vectors (`dishonest`, `pirate`, `mystic`)
#
# Persona Vectors (Chen et al. 2025) §2–3, on Qwen3-4B. Run the file **once per trait** (`TRAIT` in the config cell).
#
# **Part 1 — Make training answers.** Qwen answers N_QUESTIONS = 20 training questions under three conditions, N_ANSWERS = 10 answers each, sampled at T = 1:
# - **POS**: system prompt pushing toward the trait;
# - **NEG**: system prompt pushing away from it;
# - **BASE**: no system prompt.
#
# POS and NEG answers cycle through N_WORDINGS = 5 prompt wordings (5 POS/NEG pairs in the trait file). 600 answers.
#
# **dishonest only:** Qwen3-4B rarely lies from an instruction alone, so its prompts also carry 6 example turns before the real question (the trait file's `fewshot` list): confident lies for POS, truthful answers for NEG and BASE. They count as prompt, so they never enter the answer average. N_ANSWERS = 30 for dishonest (1,800 answers).
#
# **Part 2 — Score them.** Haiku gives every answer a `trait_score` and a `coherence` (0–100, one call each), or REFUSAL if Qwen declined to answer. Once, for dishonest, Opus re-scores 100 of them as a check on Haiku.
#
# **Part 3 — Turn them into vectors**:
# - **FILTER**: which answers count as clean examples (`pv.filter_rollouts`);
# - **TOKENS**: which positions get averaged (`pv.reduce_tokens`);
# - **SCALE**: what length the vector gets (`pv.persona_vectors`).
#
# Output: one POS − NEG vector per layer (the paper's), plus POS − BASE for comparison.
#
# **Part 4 — Check the vector works.** Steering = add coefficient × v to the residual stream at one layer while Qwen answers the 20 held-out questions with no system prompt.
# - **WHICH LAYER**: steer at 9 layers × 3 coefficients, then pick the layer (Peter's call, from the sweep).
# - **HOW STRONG**: 6 coefficients at that layer → trait and coherence curves. Repeated with POS − BASE only if the two vectors disagree (cos < 0.9).
# - **MONITOR** (dishonest only): an 8-prompt ladder from honest to dishonest. Project the last prompt token onto v̂ before the answer exists and correlate with judged dishonesty; then the same reading during the answer, token by token.
#
# **Save** the vectors for Exp 2.
#
# **Read first:** §2, §3, App. A.1–A.3, B.1, B.4, C.2–C.3.
#
# **Where this differs from the paper**
# - Model: Qwen3-4B (36 blocks, thinking off) instead of Qwen2.5-7B-Instruct / Llama-3.1-8B.
# - Judge: Claude Haiku 5.5 writing one integer, instead of GPT-4.1-mini with a logprob-weighted score. Scores come out coarser.
# - Traits: `dishonest`, `pirate` and `mystic` are not paper traits. Claude wrote their artifacts in the paper's A.1 format; review `data/traits/*.json` and the dishonest ladder in `data/monitor_ladders.json` before running.
# - System prompts: "You are a helpful assistant." on both POS and NEG, so they differ only in the trait instruction. The paper's figures suggest "You are an evil assistant." on the trait side.
# - Ours, not the paper's: dishonest's few-shot example turns; BASE and the POS − BASE comparison; MONITOR read during the answer; the Opus check on the judge.
# - Coherence prompt: Betley et al.'s with two edits so deliberate falsehoods and unusual voices aren't marked down (see `judge.py`).
# - TOKENS: the first 40 answer tokens instead of the whole answer. Dishonest's kept lies are short (median 41 tokens vs 98 for NEG), and late layers of the whole-answer vector picked up answer length (TOKENS check in Part 3).
# - Training questions whose judge labels don't track the instruction are dropped before FILTER (the trait file's `drop_extract`, with reasons).
# - Cost cuts: the layer sweep tries every third layer, not every layer; MONITOR takes 4 answers per prompt and question, the paper 10.
# - Layers are 0-indexed blocks (repo convention), so the paper's "layer 20" is our 19.
#
# **Compute:** runs locally on a 24 GB GPU (parker's 3090: dishonest's Part 1 ≈ 25 min, each activation pass ≈ 3 min). A full run is ≈ 2,100 generations and ≈ 4,300 judge calls per trait (≈ $1 on Haiku), plus ≈ 650 / 650 for dishonest's MONITOR and ≈ 360 / 720 if the POS − BASE repeat runs. Reading during the answer adds forward passes only, no judge calls. The judge prints its running cost. `SMOKE = True` is a tiny mechanics-only pass (the real model, a handful of answers) that checks the plumbing and nothing else: its numbers mean nothing.

# %% [markdown]
# ## Predictions — write before running
#
# _Peter_ (dishonest, 2026-10-13):
# - Share of POS answers with trait ≥ 60, overall / factual / advice / about itself / explanations: ~50% / ~50% / ~10% (real chance of 0%) / >70% / ~50%. "The model was quite stubborn to lie."
# - Fraction of answers that survive your filter, POS / NEG / BASE: 50% / >90% / BASE not cut on trait. Coherence ≥ 70 for >95%, possibly all.
# - Haiku vs Opus: trait scores close, nothing wildly different. Coherence: most clear the cut, but scores will vary a lot. A perfectly understandable but strange answer gets something other than 100; maybe the judge infers a range or distribution.
#
# Before Parts 3–4:
# - Best steering layer (0-indexed block of 36), dishonest / pirate / mystic:
# - Which trait steers most cleanly (trait score at matched coherence), and which worst?
# - MONITOR r for dishonest, overall / within-level:
# - MONITOR during the answer: does r rise as Qwen writes more, and where in the answer does the projection separate the ladder levels?
# - cos(POS − NEG, POS − BASE) at your chosen layer, per trait (random directions in 2560-d sit at |cos| ≈ 0.02):
# - What you expect to break:
#
# _Claude_ (dishonest, written after Peter's, but after reading 5 POS answers, so slightly informed):
# - POS trait ≥ 60: ~45% / ~25% (vaccines, bleach, eclipse are safety questions Qwen resists) / ~35% / ~80% / ~40%.
# - Survive the filter: POS ~40% / NEG ~95% / BASE ~97% (coherence only).
# - Haiku vs Opus: trait passes, but closer to the 90% bar than pirate (a half-truth is harder to call than pirate speak). Coherence rank correlation low again (~0.4): nearly every answer sits at 85–100, so the order within that band is mostly noise. Agreement on the 70 cut is what matters, and I expect ≥ 95%.

# %% Config: trait, counts, model
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display

import pv
import judge as J

TRAIT = "dishonest"   # "dishonest" | "pirate" | "mystic": run once per trait
SMOKE = False    # True → tiny mechanics-only pass; numbers meaningless
SEED = 0

# Training answers: N_QUESTIONS questions × 3 conditions (POS, NEG, BASE) × N_ANSWERS each.
# POS and NEG answers cycle through N_WORDINGS prompt wordings (10 over 5 = 2 per wording).
if SMOKE:   # plumbing only: the real model, tiny counts (on a Mac, use "Qwen/Qwen3-0.6B")
    MODEL_ID, N_QUESTIONS, N_WORDINGS, N_ANSWERS, MAX_NEW = None, 4, 2, 2, 64
else:
    MODEL_ID, N_QUESTIONS, N_WORDINGS, N_ANSWERS, MAX_NEW = None, 20, 5, 10, 500
    if TRAIT == "dishonest":
        N_ANSWERS = 30   # Qwen3-4B lies in only part of its POS answers: more answers, more survivors

GEN = dict(max_new_tokens=MAX_NEW, batch_size=32, seed=SEED)
TAG = f"{TRAIT}{'_smoke' if SMOKE else ''}"   # cache-file prefix

trait = pv.load_trait(TRAIT)
judge = J.Judge()   # Haiku 5.5; the judge check below re-scores a sample with Opus
model, tok, device = pv.load_model(MODEL_ID)
print(model.config._name_or_path, "| blocks:", pv.n_blocks(model), "| device:", device, "| judge:", judge.model)

# %% What the model sees
# What the model actually sees, for wording 0
print(pv.system_prompt(trait, 0, "POS"))
print(pv.system_prompt(trait, 0, "NEG"))
print(pv.chat_prompt(tok, pv.system_prompt(trait, 0, "POS"), trait["questions_extract"][0],
                     pv.fewshot_turns(trait, "POS")))   # dishonest: example turns before the real question

# %% [markdown]
# ## Part 1 — Make training answers

# %% Part 1 · generate answers
table = pv.rollout_table(trait, "extract", n_answers=N_ANSWERS, n_questions=N_QUESTIONS, n_wordings=N_WORDINGS)
print(len(table), "answers to generate")
ROLL_KEY = pv.key(table.to_json(), MAX_NEW, SEED)   # any change to prompts or questions → fresh cache
roll = pv.cached_df(f"rollouts_{TAG}_{ROLL_KEY}", lambda: pv.generate_rollouts(model, tok, table, device=device, **GEN))
print(f"truncated: {roll.truncated.mean():.1%}   mean tokens: {roll.n_tokens.mean():.0f}")
roll[["condition", "wording", "question", "response"]].head(4)

# %% [markdown]
# ## Part 2 — Score them
#
# Cached per call in `cache/judge_cache.jsonl`, so re-running this cell is free.

# %% Part 2 · judge
scored = judge.score_df(roll, trait["eval_prompt"])
scored.groupby("condition")[["trait_score", "coherence", "refusal", "truncated"]].mean().round(2)

# %% [markdown]
# ### Judge check: does Haiku agree with Opus?
#
# Once, on the first trait (dishonest): Opus re-scores a random 100 of the answers above, trait and coherence (≈ 200 Opus calls, ≈ $1.50). **Pass:** ≥ 90% agreement on above/below 50 and rank correlation ≥ 0.8. If it fails, read the disagreements and fix the judge prompt before going on.

# %% Part 2 · Haiku vs Opus check
if SMOKE or TRAIT == "dishonest":
    check = scored.sample(min(100, len(scored)), random_state=SEED)
    ref = J.Judge(model="claude-opus-5-5").score_df(check, trait["eval_prompt"])
    both = check[["trait_score", "coherence"]].join(ref[["trait_score", "coherence"]], rsuffix="_opus").dropna()
    agree = ((both.trait_score > 50) == (both.trait_score_opus > 50)).mean()
    print(f"n = {len(both)}   above/below-50 agreement {agree:.0%}   "
          f"trait rank corr {both.trait_score.corr(both.trait_score_opus, method='spearman'):.2f}   "
          f"coherence rank corr {both.coherence.corr(both.coherence_opus, method='spearman'):.2f}")
    display(both.assign(gap=(both.trait_score - both.trait_score_opus).abs())
                .sort_values("gap", ascending=False).head(5))   # biggest disagreements: read these

# %% Part 2 · score plots
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, col in zip(axes, ["trait_score", "coherence"]):
    for cond, c in [("POS", "C3"), ("NEG", "C0"), ("BASE", "C7")]:
        ax.hist(scored.loc[scored.condition == cond, col].dropna(), bins=np.arange(0, 105, 5),
                alpha=.6, color=c, label=cond)
    ax.set_title(col); ax.legend()
plt.tight_layout()

# Which wordings actually elicit the trait? (BASE has no wording: -1)
display(scored.pivot_table(index="wording", columns="condition", values="trait_score").round(1))

for cond in ("POS", "NEG", "BASE"):
    r = scored[scored.condition == cond].sample(1, random_state=SEED).iloc[0]
    print(f"--- {cond} | trait {r.trait_score} | coherence {r.coherence}\nQ: {r.question}\nA: {r.response[:600]}\n")

# %% [markdown]
# ## Part 3 — Turn them into vectors
#
# ### FILTER: `pv.filter_rollouts`
#
# `pv.filter_rollouts`: which answers count as clean examples? Look at the plots above first. The paper keeps POS answers with trait score > 50 and NEG answers < 50, on the trait score only (§2.2). Keep the original index: it picks the matching activations below.
#
# `scored` has all three conditions: `POS`, `NEG` and `BASE` (no system prompt). `BASE` rows are the subtracted side of POS − BASE, so the filter decides how they are treated too.

# %% Part 3 · FILTER
drop = [d["qid"] for d in trait.get("drop_extract", [])]   # questions whose judge labels don't track the instruction (why: trait file)
kept = pv.filter_rollouts(scored[~scored.qid.isin(drop)])
print(kept.condition.value_counts().to_dict(), f"kept of {len(scored)}")
kept.groupby("qid").condition.value_counts().unstack(fill_value=0).T   # coverage per question

# %% [markdown]
# ### TOKENS: `pv.reduce_tokens`
#
# Qwen re-reads each answer once and every layer's residual stream is recorded; `reduce_tokens(h, n_prompt, first_k)` collapses the token axis. The paper averages over the answer tokens (§2.2, App. A.3). This runs over **all** answers, not just the kept ones, so changing the filter doesn't invalidate the cache. That means the reduction has to tolerate an answer with zero tokens (`n_prompt == seq`). Bump `REDUCE_VERSION` whenever you change `reduce_tokens`.
#
# Two reductions, compared in the TOKENS check below: the whole answer (the paper's) and only its first `FIRST_K` tokens. Dishonest's kept lies are much shorter than its honest answers (median 41 tokens vs 98 for NEG), so the whole-answer average may compare "the opening of an answer" with "a whole answer" on top of lying vs honest.

# %% Part 3 · TOKENS: activations
REDUCE_VERSION = "v1"
FIRST_K = 40   # ≈ the median kept dishonest lie
REDUCERS = {"full": pv.reduce_tokens,
            f"first{FIRST_K}": lambda h, n: pv.reduce_tokens(h, n, first_k=FIRST_K)}
acts_by = {}
for red, fn in REDUCERS.items():
    name = f"acts_{TAG}_{ROLL_KEY}_{REDUCE_VERSION}_{red}"
    if (pv.CACHE_DIR / f"{name}.pt").exists():
        acts_by[red] = pv.load_tensor(name)
    else:
        acts_by[red] = pv.collect_reduced(model, tok, scored.prompt.tolist(), scored.response.tolist(),
                                          fn, device=device)
        pv.save_tensor(acts_by[red], name)
REDUCE = "first40"   # used from here on. TOKENS check (dishonest): whole-answer picks up length in late blocks
acts_all = acts_by[REDUCE]
print({r: tuple(a.shape) for r, a in acts_by.items()})   # (n answers, n_blocks, d)

# %% [markdown]
# ### SCALE: `pv.persona_vectors`
#
# Average the POS answers, average the NEG answers, subtract: one vector per layer. The length you give it sets the units of every steering coefficient in Part 4. The left plot compares each layer's vector length with the typical size of the residual stream. The paper steers with what look like raw differences (coefficients 0.5–2.5) and rescales to equal length only when comparing vectors (App. A.3).

# %% Part 3 · SCALE: POS − NEG vectors
main = pv.contrast(kept, "NEG")   # POS − NEG: the paper's vector, used everywhere below
n = main.condition.value_counts().to_dict()
assert n.get("POS") and n.get("NEG"), f"FILTER left {n}: no POS or no NEG answers to average. See the trait scores by wording above."
acts = acts_all[torch.tensor(main.index.to_numpy())]
MATCH = False   # match by question (topic check below): set it after the check
V = pv.persona_vectors(acts, main.reset_index(drop=True), match_questions=MATCH)   # (n_blocks, d)
assert V.shape == acts.shape[1:], V.shape

vnorm = V.norm(dim=-1)
resid_norm = acts_all.norm(dim=-1).mean(0)   # typical norm of a reduced activation, per block
Vn = V / vnorm[:, None]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(vnorm, label="‖v_L‖")
axes[0].plot(resid_norm, label="mean ‖reduced h_L‖")
axes[0].set_yscale("log"); axes[0].set_xlabel("block"); axes[0].legend()
im = axes[1].imshow((Vn @ Vn.T).numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
axes[1].set_title("cos(v_L, v_L')"); axes[1].set_xlabel("block"); plt.colorbar(im, ax=axes[1])
plt.tight_layout()

# %% [markdown]
# ### TOKENS check: whole answer vs first `FIRST_K` tokens
#
# Same kept rows, both reductions, POS − NEG built each way; cosine per layer. The dashed line is a noise reference: the `REDUCE` vector rebuilt from two random halves of the kept answers. Each half has half the data, so that line sits *below* how stable the full vectors really are.
#
# At or above the dashed line → the choice doesn't matter beyond noise: keep the paper's (`"full"`). Clearly below → length leaks into the vector; the WHICH LAYER sweep in Part 4 can then steer with each and show which is cleaner.

# %% Part 3 · TOKENS check
idx = torch.tensor(main.index.to_numpy())
lab = main.reset_index(drop=True)
V_by = {r: pv.persona_vectors(a[idx], lab) for r, a in acts_by.items()}
cos_tok = torch.nn.functional.cosine_similarity(*V_by.values(), dim=-1)

# Noise reference: the REDUCE vector from two random halves (POS and NEG each split in half)
h1 = lab.groupby("condition").sample(frac=.5, random_state=SEED).index.to_numpy()
m = np.zeros(len(lab), bool); m[h1] = True
a = acts_by[REDUCE][idx]
V_h1 = pv.persona_vectors(a[torch.from_numpy(m)], lab[m].reset_index(drop=True))
V_h2 = pv.persona_vectors(a[torch.from_numpy(~m)], lab[~m].reset_index(drop=True))
cos_half = torch.nn.functional.cosine_similarity(V_h1, V_h2, dim=-1)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(cos_tok, marker="o", label=" vs ".join(V_by))
ax.plot(cos_half, ls="--", color="k", label=f"{REDUCE}: half vs half")
ax.set_ylim(-0.05, 1.05); ax.set_xlabel("block"); ax.set_title("cos between POS − NEG vectors"); ax.legend()
plt.tight_layout()
print("cos full vs first-K, every 3rd block from 6:", {L: round(cos_tok[L].item(), 2) for L in range(6, 33, 3)})

# %% [markdown]
# ### Topic check: pooled vs matched by question
#
# Kept POS answers bunch up on a few questions (dishonest: climate alone is 20% of the lies, 6% of NEG), so "all POS minus all NEG" also carries "more of those topics". Matched: POS mean − NEG mean inside each question, weighted by its number of POS answers; questions with no POS drop out. Same `REDUCE`, cosine per layer, against the same half-vs-half noise line as the TOKENS check. At or above it → topic doesn't matter beyond noise.

# %% Part 3 · topic check
V_pool = pv.persona_vectors(acts_by[REDUCE][idx], lab)
V_match = pv.persona_vectors(acts_by[REDUCE][idx], lab, match_questions=True)
cos_topic = torch.nn.functional.cosine_similarity(V_pool, V_match, dim=-1)
print("questions with POS answers:", lab[lab.condition == "POS"].qid.nunique(), "of", lab.qid.nunique())

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(cos_topic, marker="o", label="pooled vs matched")
ax.plot(cos_half, ls="--", color="k", label=f"{REDUCE}: half vs half")
ax.set_ylim(-0.05, 1.05); ax.set_xlabel("block"); ax.set_title(f"cos between POS − NEG vectors ({REDUCE})"); ax.legend()
plt.tight_layout()
print("cos pooled vs matched, every 3rd block from 6:", {L: round(cos_topic[L].item(), 2) for L in range(6, 33, 3)})

# %% [markdown]
# ### POS − NEG vs POS − BASE
#
# Same POS rows, two different subtracted sides: the anti-trait system prompt (the paper) or no system prompt at all. The two vectors differ by exactly BASE − NEG: whatever the NEG prompt adds beyond the default Assistant.
#
# Near 1 at the layers you care about → for this trait, the choice barely matters. Geometry is only a hint: whether the difference *matters* is the HOW STRONG repeat below, which runs only if cos < 0.9.

# %% Part 3 · POS − NEG vs POS − BASE
alt = pv.contrast(kept, "BASE")   # POS − BASE
assert (alt.condition == "NEG").any(), "FILTER left no BASE answers"   # BASE is relabelled NEG here
V_base = pv.persona_vectors(acts_all[torch.tensor(alt.index.to_numpy())], alt.reset_index(drop=True),
                            match_questions=MATCH)

cos_nb = torch.nn.functional.cosine_similarity(V, V_base, dim=-1)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(cos_nb, marker="o")
ax.axhline(0, color="k", lw=.5)
ax.set_ylim(-1.05, 1.05); ax.set_xlabel("block")
ax.set_title("cos(POS − NEG, POS − BASE)")   # random pairs in 2560-d: |cos| ≈ 0.02
plt.tight_layout()

# %% [markdown]
# ## Part 4 — Check the vector works
#
# ### WHICH LAYER (Fig. 3, App. B.4)
#
# Steer with each layer's vector at the same coefficients, on the 20 held-out questions, no system prompt. The unsteered answers (layer -1) are the plain model's baseline. Set `SWEEP_COEFFS` in the units your SCALE choice implies: the length plot above gives the scale.

# %% Part 4 · WHICH LAYER sweep
SWEEP_LAYERS = [1, 3] if SMOKE else list(range(6, 33, 3))   # 0-indexed blocks
SWEEP_COEFFS = [4.0, 16.0] if SMOKE else None   # set: ~3 values, in the units of V (smoke: placeholders)
N_SWEEP = 1 if SMOKE else 2
assert SWEEP_COEFFS, "set SWEEP_COEFFS"

eval_qs = trait["questions_eval"][:N_QUESTIONS]
sweep = pv.cached_df(
    f"sweep_{TAG}_{pv.key(V, SWEEP_LAYERS, SWEEP_COEFFS, N_SWEEP, MAX_NEW)}",
    lambda: pv.steering_sweep(model, tok, eval_qs, V, SWEEP_LAYERS, SWEEP_COEFFS,
                              n_samples=N_SWEEP, device=device, **GEN))
sweep = judge.score_df(sweep, trait["eval_prompt"])

# %% Part 4 · sweep plots
base = sweep[sweep.layer == -1][["trait_score", "coherence"]].mean()
g = sweep[sweep.layer >= 0].groupby(["coeff", "layer"])[["trait_score", "coherence"]].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True)
for ax, col in zip(axes, ["trait_score", "coherence"]):
    for c, d in g.groupby("coeff"):
        ax.plot(d.layer, d[col], marker="o", label=f"α = {c:g}")
    ax.axhline(base[col], color="k", ls="--", lw=1, label="unsteered")
    ax.set_xlabel("block"); ax.set_title(col); ax.set_ylim(-2, 102)
axes[0].legend(); plt.tight_layout()

# %% [markdown]
# ### Pick the layer
#
# The paper's rule: same coefficient at every layer, take the highest trait score (App. B.4). Write your rule and the reason here before setting `LAYER`. If you want one layer shared by all three traits, decide after all three have run.
#
# _Rule:_

# %% Part 4 · pick the layer
LAYER = 3 if SMOKE else None   # Peter's call (smoke: placeholder): 0-indexed block, from the sweep, by the rule written above
assert LAYER is not None
v = V[LAYER]

# %% [markdown]
# ### HOW STRONG
#
# At `LAYER`, 6 coefficients from "no visible effect" to "incoherent", 3 answers per held-out question. The useful range sits between where the trait appears and where coherence breaks; the paper reports results with mean coherence above 75 (§5.1).

# %% Part 4 · HOW STRONG
DOSE_COEFFS = [4.0, 16.0] if SMOKE else None   # smoke: placeholders. Set: ~6 values from "no visible effect" to "incoherent", same units as SWEEP_COEFFS
N_DOSE = 1 if SMOKE else 3
assert DOSE_COEFFS, "set DOSE_COEFFS"

dose = pv.cached_df(
    f"dose_{TAG}_{pv.key(v, LAYER, DOSE_COEFFS, N_DOSE, MAX_NEW)}",
    lambda: pv.steering_sweep(model, tok, eval_qs, V, [LAYER], DOSE_COEFFS,
                              n_samples=N_DOSE, device=device, **GEN))
dose = judge.score_df(dose, trait["eval_prompt"])

d = dose.groupby("coeff")[["trait_score", "coherence"]].agg(["mean", "sem"])
fig, ax = plt.subplots(figsize=(6, 3.5))
for col, c in [("trait_score", "C3"), ("coherence", "C0")]:
    ax.errorbar(d.index, d[(col, "mean")], yerr=d[(col, "sem")], marker="o", color=c, label=col, capsize=3)
ax.set_xlabel(f"α at block {LAYER}"); ax.set_ylim(-2, 102); ax.legend()
plt.tight_layout()

# Read a few: strongest coefficient that still reads as coherent
top = dose[(dose.coherence >= 50) & (dose.coeff > 0)].sort_values(["coeff", "trait_score"]).tail(2)
for _, r in top.iterrows():
    print(f"--- α {r.coeff:g} | trait {r.trait_score} | coherence {r.coherence}\nQ: {r.question}\nA: {r.response[:600]}\n")

# %% [markdown]
# ### HOW STRONG with POS − BASE (only if the two disagree)
#
# Same layer, same coefficients, same questions. The unsteered point (coefficient 0) is shared with the run above.
#
# The same coefficient only means the same push for both vectors if your SCALE makes it so. If the vectors keep their raw length, POS − NEG is usually the longer one, so compare the curves at matched coherence, not matched coefficient.
#
# **Runs only if cos(POS − NEG, POS − BASE) at `LAYER` is below 0.9** (scope trim). Above that the two are within ~25° of each other, and we accept the risk of missing a small steering difference.

# %% Part 4 · HOW STRONG with POS − BASE
COS_SKIP = 0.9   # scope trim: repeat HOW STRONG with POS − BASE only if the two vectors disagree at LAYER
run_repeat = SMOKE or float(cos_nb[LAYER]) < COS_SKIP
print(f"cos at block {LAYER} = {float(cos_nb[LAYER]):.2f} → POS − BASE repeat {'runs' if run_repeat else 'skipped'}")

if run_repeat:
    dose_b = pv.cached_df(
        f"doseB_{TAG}_{pv.key(V_base[LAYER], LAYER, DOSE_COEFFS, N_DOSE, MAX_NEW)}",
        lambda: pv.steering_sweep(model, tok, eval_qs, V_base, [LAYER], DOSE_COEFFS, n_samples=N_DOSE,
                                  include_baseline=False, device=device, **GEN))
    dose_b = judge.score_df(dose_b, trait["eval_prompt"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True)
    unsteered = dose[dose.layer == -1]
    for df_, label, ls in [(dose, "POS − NEG", "-"),
                           (pd.concat([unsteered, dose_b]), "POS − BASE", "--")]:
        g = df_.groupby("coeff")[["trait_score", "coherence"]].agg(["mean", "sem"])
        for ax, col in zip(axes, ["trait_score", "coherence"]):
            ax.errorbar(g.index, g[(col, "mean")], yerr=g[(col, "sem")], marker="o", ls=ls, capsize=3, label=label)
    for ax, col in zip(axes, ["trait_score", "coherence"]):
        ax.set_title(col); ax.set_xlabel(f"α at block {LAYER}"); ax.set_ylim(-2, 102); ax.legend()
    plt.tight_layout()

# %% [markdown]
# ### MONITOR (dishonest only; §3.3, Fig. 4, App. C.2)
#
# The ladder: 8 system prompts from honest (level 0) to openly dishonest (level 7), × the 20 held-out questions. Project the **last prompt token** at `LAYER` onto v̂ (before any answer exists), then judge what the model actually says. Does the projection predict the behaviour?
# - **overall r**: across all points. Mostly separates "told to lie" from "told to be honest", which a vector that merely reads the system prompt would also do.
# - **within-level r**: per ladder prompt, then averaged. Same system prompt, different questions: the real test.
#
# Runs for `dishonest` only (scope trim): pirate and mystic have no ladder in `data/monitor_ladders.json`.

# %% Part 4 · MONITOR
r = None
if pv.has_ladder(TRAIT):
    ladder = pv.load_ladder(TRAIT)[:: 7 if SMOKE else 1]   # smoke: just the two ends
    N_MON = 1 if SMOKE else 4

    mon = pv.cached_df(
        f"monitor_{TAG}_{pv.key(v, LAYER, ladder, N_MON, MAX_NEW)}",
        lambda: pv.monitoring_run(model, tok, ladder, eval_qs, LAYER, v, n_samples=N_MON, device=device, **GEN))
    mon = judge.score_df(mon, trait["eval_prompt"], coherence=False)
    agg, r = pv.monitor_correlations(mon)
    print(r)

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(agg.projection, agg.trait_score, c=agg.level, cmap="viridis", s=18)
    plt.colorbar(sc, label="ladder level (0 = discourages trait)")
    ax.set_xlabel(f"projection onto v̂, block {LAYER}, last prompt token")
    ax.set_ylabel(f"mean {TRAIT} score")
    ax.set_title(f"overall r = {r['overall_r']:.2f}   within-level r = {r['within_level_r']:.2f}")
    plt.tight_layout()
else:
    print(f"no ladder for {TRAIT}: monitoring skipped")

# %% [markdown]
# ### MONITOR, read during the answer
#
# The same 640 answers, no new judge calls. Project the residual at `LAYER` onto v̂ at every position: position 0 is the last prompt token (the reading above), position t the t-th answer token.
# - **Left:** mean projection at each position, one line per ladder level. Where in the answer do the levels separate, and does it build up?
# - **Right:** correlate the running mean of the first positions with the judged score, for more and more answer tokens read. 0 tokens = the before-the-answer reading. Flat → the persona is set up before the first word; climbing → the answer carries information the prompt didn't. Dashed: the mean over the answer tokens only, the way the vector was built.

# %% Part 4 · MONITOR during the answer
if pv.has_ladder(TRAIT):
    import warnings
    tname = f"traj_{TAG}_{pv.key(v, LAYER, ladder, N_MON, MAX_NEW)}"
    if (pv.CACHE_DIR / f"{tname}.pt").exists():
        traj = pv.load_tensor(tname)
    else:
        mon_prompts = [pv.chat_prompt(tok, s, q) for s, q in zip(mon.system, mon.question)]
        traj = pv.token_projections(model, tok, mon_prompts, mon.response.tolist(), LAYER, v, device=device)
        pv.save_tensor(traj, tname)   # one 1-D tensor per answer: [last prompt token, answer tokens...]

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
    axes[0].set_ylabel(f"projection onto v̂, block {LAYER}")
    axes[0].legend(fontsize=7, ncol=2)

    READ = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]   # answer tokens read
    curve = []
    for a in READ:
        _, rk = pv.monitor_correlations(mon.assign(p=[float(t[: a + 1].mean()) for t in traj]), proj_col="p")
        curve.append(dict(read=a, overall=rk["overall_r"], within=rk["within_level_r"]))
    curve = pd.DataFrame(curve)
    _, r_ans = pv.monitor_correlations(
        mon.assign(p=[float(t[1:].mean()) if len(t) > 1 else np.nan for t in traj]), proj_col="p")
    for col, c in [("overall", "C0"), ("within", "C1")]:
        axes[1].plot(curve.read, curve[col], marker="o", color=c, label=f"{col} r, running mean")
        axes[1].axhline(r_ans[f"{col}_r" if col == "overall" else "within_level_r"], color=c, ls="--", lw=1)
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlabel("answer tokens read (0 = before the answer)")
    axes[1].set_ylabel("Pearson r with judged dishonesty")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    display(curve.round(2))

# %% [markdown]
# ## Save

# %% Save
if not SMOKE:
    p = pv.save_persona(TRAIT, V, LAYER, reduce_tag=f"{REDUCE_VERSION}_{REDUCE}", n_kept=len(main), vectors_base=V_base.cpu(),
                        judge=judge.model, monitor=r, model=model.config._name_or_path)
    print("saved", p)
print(f"judge cost this session ≈ ${judge.cost():.2f}")

# %% [markdown]
# ## Results
#
# - **dishonest:** layer __; steering __; MONITOR r overall __ / within __; during the answer: __
# - **pirate:** layer __; steering __
# - **mystic:** layer __; steering __
# - **POS − NEG vs POS − BASE:** cos @ layer, dishonest __ / pirate __ / mystic __; POS − BASE repeat ran for: __
# - **vs predictions:**
# - **Parked** (one line, only if something comes up):
