# Exp 3: is the truth direction a lever the model uses, or only a correlate? And do two probes
# with the same accuracy (LR, MM) make equally good levers? The walk-through is 03_causal.md;
# this file is only the code, one cell per step, titled as in the doc.

# %% [markdown]
# ## Part 1 — Push along the direction, read the verdict

# %% Load the model and the directions
import logging

import numpy as np
import pandas as pd
import savoy
from IPython.display import display

import lib

logging.basicConfig(level=logging.INFO, format="· %(message)s")   # no-op inside esper
log = logging.getLogger(__name__)

SEED = 0
LAYER = 20
KS = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4]   # push strength, in matched units (below)

model, tok = lib.load_model_and_tokenizer()
truth = lib.load_cache(f"truth_probe_L{LAYER}")   # from 02_truth_probe.py
sp = lib.load_cache(f"sp_probe_L{LAYER}")         # from 01_what_the_probe_reads.py
log.info("loaded the truth probe (Exp 2) and the persona probe (Exp 1), both at block %d", LAYER)

# %% Match the push strength
V = {"LR": lib.matched_steering_vector(truth["theta_lr"], truth["mu_pos"], truth["mu_neg"]),
     "MM": lib.matched_steering_vector(truth["theta_mm"], truth["mu_pos"], truth["mu_neg"]),
     "persona (control)": lib.matched_steering_vector(sp["theta_lr"], sp["mu_pos"], sp["mu_neg"])}
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
log.info("k = 1 vector lengths: %s · cos(θ_lr, θ_mm) %.2f · cos(θ_mm, persona) %.2f",
         " · ".join(f"{n} {np.linalg.norm(v):.1f}" for n, v in V.items()), cos(V["LR"], V["MM"]), cos(V["MM"], V["persona (control)"]))

# %% Baseline: the model's own verdict
_, test = lib.build_truth_dataset(lib.DATA_DIR / "facts_true_false.csv", seed=SEED, n_train=400, n_test=150)
eval_true = [lib._make_eval_prompt(r["statement"]) for r in test if r["label"] == 1]
eval_false = [lib._make_eval_prompt(r["statement"]) for r in test if r["label"] == 0]
true_ids, false_ids = lib.truth_token_ids(tok)
print(repr(eval_false[0]))
d_pos = lib.truth_logit_diff(model, tok, eval_true, true_ids, false_ids)
d_neg = lib.truth_logit_diff(model, tok, eval_false, true_ids, false_ids)
PDpos, PDneg = float(d_pos.mean()), float(d_neg.mean())
gate = lib.auroc(np.concatenate([d_pos, d_neg]), np.r_[np.ones(len(d_pos)), np.zeros(len(d_neg))])
assert PDpos > PDneg, "the model's own verdict doesn't separate true from false: NIE is undefined"
log.info("unsteered Δ = logP(True) − logP(False): true statements %+.1f · false %+.1f · verdict AUROC %.3f",
         PDpos, PDneg, gate)

# %% Sweep the push
def nie(v, k):
    if k == 0:
        return 0.0
    dk = lib.truth_logit_diff(model, tok, eval_false, true_ids, false_ids,
                              steer=dict(layer=LAYER, vector=v, coeff=float(k), mode="context"))
    return (dk.mean() - PDneg) / (PDpos - PDneg)
curves = pd.DataFrame({n: [nie(v, k) for k in KS] for n, v in V.items()}, index=pd.Index(KS, name="k"))
display(curves.T.round(2))
log.info("NIE at k = 1 / 2 / 4: %s", " · ".join(f"{n} {c[1]:.2f} / {c[2]:.2f} / {c[4]:.2f}" for n, c in curves.items()))

# %% Match by push size instead
LENGTHS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]   # |k·v|: how far the activations actually move
by_length = pd.DataFrame({n: [nie(v, L / np.linalg.norm(v)) for L in LENGTHS] for n, v in V.items()},
                         index=pd.Index(LENGTHS, name="length"))
display(by_length.T.round(2))
log.info("NIE at push length 20 / 35 / 70: %s",
         " · ".join(f"{n} {c[20]:.2f} / {c[35]:.2f} / {c[70]:.2f}" for n, c in by_length.items()))

# %% Plot the curves
fig, (a, b) = savoy.figure(1, 2, height=260, ytitle="NIE on false statements",
                           xtitle=["k (k = 1: one gap on the probe's own score)", "push size (vector length)"])
for i, n in enumerate(V):
    c = None if n == "persona (control)" else savoy.color(i)
    savoy.line(a, KS, list(curves[n]), c, points=True, baseline=c is None)
    savoy.line(b, LENGTHS, list(by_length[n]), c, points=True, baseline=c is None)
    if c is not None:
        savoy.key(fig, n, c)
for ax in (a, b):
    savoy.ref(ax, 0, None)
savoy.save(fig, lib.TOPIC_DIR / "figures/nie_curves.png")

# %% [markdown]
# ## Part 2 — What the push does to free text

# %% Steer free answers
K_DEMO = 4.0
demo_qs = ["What is the capital of Australia?", "How many moons does Earth have?",
           "Who wrote the play Romeo and Juliet?", "What is the boiling point of water at sea level in Celsius?"]
demo_prompts = [lib._make_gen_prompt("an honest", q, think=False) for q in demo_qs]
demo = {"unsteered": lib.generate(model, tok, demo_prompts, max_new_tokens=60)}
for sign in [-1, +1]:
    with lib.steering(model, LAYER, V["MM"], coeff=sign * K_DEMO):   # every position, prompt and answer
        demo[f"{sign * K_DEMO:+g} × θ_mm"] = lib.generate(model, tok, demo_prompts, max_new_tokens=60)
for i, q in enumerate(demo_qs):
    print("=" * 80, "\nQ:", q)
    for name, outs in demo.items():
        print(f"  {name:>12}: {' '.join(outs[i].split())[:200]}")
honest_words = {n: sum(("honest" in o.lower()) for o in outs) for n, outs in demo.items()}
log.info("answers that mention honesty, out of %d: %s", len(demo_qs), " · ".join(f"{n} {c}" for n, c in honest_words.items()))
