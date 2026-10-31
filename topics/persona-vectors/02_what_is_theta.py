# Exp 2: what is θ_mm? Compares linear-probes' truth direction with Exp 1's persona vectors. Flow doc: 02_what_is_theta.md.
# %% Load the directions
import logging

import numpy as np
import pandas as pd
import savoy
import torch
import torch.nn.functional as F
from IPython.display import display

import pv

logging.basicConfig(level=logging.INFO, format="· %(message)s")   # no-op inside esper
log = logging.getLogger(__name__)

BLOCK = 20   # where θ_mm was fit (linear-probes 02_truth_probe.py); same 0-indexed block convention as pv
LP = pv.CACHE_DIR.parent.parent / "linear-probes" / "cache"
truth = torch.load(LP / f"truth_probe_L{BLOCK}.pt", weights_only=False)   # θ = μ_true − μ_false: + = "this statement is true"
sp = torch.load(LP / f"sp_probe_L{BLOCK}.pt", weights_only=False)         # θ = μ_honest − μ_untruthful persona (Zou pairs)
D = {"θ_mm (truth)": truth["theta_mm"], "θ_lr (truth)": truth["theta_lr"], "θ_SP (honest persona)": sp["theta_mm"]}
D |= {f"v_{t}": pv.load_persona(t)["vectors"][BLOCK] for t in ("dishonest", "mystic", "pirate")}   # + = toward the trait
D = {k: F.normalize(torch.as_tensor(np.asarray(v), dtype=torch.float32), dim=0) for k, v in D.items()}
log.info("%d directions at block %d, %d dims", len(D), BLOCK, len(next(iter(D.values()))))

# %% Cosines
M = torch.stack(list(D.values()))
C = pd.DataFrame((M @ M.T).numpy(), index=D.keys(), columns=D.keys()).round(2)
with pd.option_context("display.width", 200, "display.max_columns", 10):
    display(C)
g = torch.Generator().manual_seed(0)
rand = F.normalize(torch.randn(2000, M.shape[1], generator=g), dim=1)
null = (rand @ D["θ_mm (truth)"]).abs()
log.info("random directions vs θ_mm: |cos| 95th percentile %.3f, max %.3f", float(null.quantile(.95)), float(null.max()))
log.info("cos with θ_mm: dishonest %.2f · mystic %.2f · pirate %.2f · honest persona θ_SP %.2f",
         *(C.loc["θ_mm (truth)", k] for k in ("v_dishonest", "v_mystic", "v_pirate", "θ_SP (honest persona)")))

# %% Where does the overlap come from?
PROBES, PERSONAS = ["θ_mm (truth)", "θ_lr (truth)", "θ_SP (honest persona)"], ["v_dishonest", "v_mystic", "v_pirate"]
MASSIVE = [0, 4, 7, 100, 396]   # Exp 1 step 6 (checks/spike_random.py): the spike tokens' 5 giant dims
keep = torch.ones(M.shape[1], dtype=torch.bool)
keep[MASSIVE] = False
Mk = F.normalize(M[:, keep], dim=1)
Ck = pd.DataFrame((Mk @ Mk.T).numpy(), index=D.keys(), columns=D.keys()).round(2)
with pd.option_context("display.width", 200, "display.max_columns", 10):
    display(Ck - C)   # change from dropping the massive dims
log.info("without the 5 massive dims: mystic·pirate %.2f (was %.2f) · dishonest·pirate %.2f (was %.2f) · θ_mm·dishonest %.2f (was %.2f)",
         Ck.loc["v_mystic", "v_pirate"], C.loc["v_mystic", "v_pirate"], Ck.loc["v_dishonest", "v_pirate"],
         C.loc["v_dishonest", "v_pirate"], Ck.loc["θ_mm (truth)", "v_dishonest"], C.loc["θ_mm (truth)", "v_dishonest"])

# Shared part, leave-one-out: for each persona, the shared axis is the mean of the *other two*, so its own trait never enters it.
rows = []
for t in PERSONAS:
    s = F.normalize(sum(D[o] for o in PERSONAS if o != t), dim=0)
    r = F.normalize(D[t] - (D[t] @ s) * s, dim=0)
    rows.append(dict(persona=t, shared=float(D[t] @ s),
                     **{f"{p.split()[0]} before": float(D[p] @ D[t]) for p in PROBES},
                     **{f"{p.split()[0]} after": float(D[p] @ r) for p in PROBES},
                     **{f"{p.split()[0]} on shared axis": float(D[p] @ s) for p in PROBES}))
R = pd.DataFrame(rows).set_index("persona").round(2)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    display(R)
log.info("shared part out (leave-one-out): θ_mm vs dishonest %.2f → %.2f · mystic %.2f → %.2f · pirate %.2f → %.2f",
         *[x for t in PERSONAS for x in (R.loc[t, "θ_mm before"], R.loc[t, "θ_mm after"])])

# %% Load the model and the statements
import sys

sys.path.insert(0, str(LP.parent))
import lib   # linear-probes: the true/false dataset, prompt and verdict readout

model, tok, device = pv.load_model()
_, test = lib.build_truth_dataset(lib.DATA_DIR / "facts_true_false.csv", seed=0, n_train=400, n_test=150)   # as in linear-probes 03
prompts = {"true": [lib._make_eval_prompt(r["statement"]) for r in test if r["label"] == 1],
           "false": [lib._make_eval_prompt(r["statement"]) for r in test if r["label"] == 0]}
true_ids, false_ids = lib.truth_token_ids(tok)
base = {k: lib.truth_logit_diff(model, tok, p, true_ids, false_ids) for k, p in prompts.items()}
log.info("unsteered Δ = logP(True) − logP(False): %d true statements %+.1f · %d false %+.1f",
         len(base["true"]), base["true"].mean(), len(base["false"]), base["false"].mean())

# %% Push and read the verdict
BLOCKS, FRACS = [18, 20], [0.1, 0.2, 0.3, 0.4, 0.6]
h_lens = np.mean([pv.load_persona(t)["h_lens"] for t in ("dishonest", "mystic", "pirate")], axis=0)   # stream length per block
rows = []
for b in BLOCKS:
    U = {f"v_{t}": F.normalize(torch.as_tensor(np.asarray(pv.load_persona(t)["vectors"][b]), dtype=torch.float32), dim=0)
         for t in ("dishonest", "mystic", "pirate")}
    if b == BLOCK:
        U["−θ_mm (toward 'false')"] = -D["θ_mm (truth)"]   # θ_mm exists only at block 20
    U["random"] = rand[0]
    for n, u in U.items():
        for f in FRACS:
            vec = (f * h_lens[b] * u).numpy()
            for k, p in prompts.items():
                d = lib.truth_logit_diff(model, tok, p, true_ids, false_ids, steer=dict(layer=b, vector=vec, coeff=1.0, mode="context"))
                rows.append(dict(block=b, push=n, frac=f, statements=k, delta=float(d.mean()),
                                 correct=float((d > 0).mean() if k == "true" else (d < 0).mean())))
    log.info("block %d done", b)
S = pd.DataFrame(rows)
S0 = pd.DataFrame([dict(block=b, push=n, frac=0.0, statements=k, delta=float(base[k].mean()),
                        correct=float((base[k] > 0).mean() if k == "true" else (base[k] < 0).mean()))
                   for b in BLOCKS for n in S[S.block == b].push.unique() for k in prompts])
S = pd.concat([S0, S]).sort_values(["block", "push", "frac"])
with pd.option_context("display.width", 200, "display.max_rows", 200):
    display(S.pivot_table(index=["block", "push", "frac"], columns="statements", values=["delta", "correct"]).round(2))
at = S[(S.frac == 0.4)].pivot_table(index=["block", "push"], columns="statements", values="delta").round(1)
log.info("at 0.4 × stream, Δ true / false: %s", " · ".join(f"b{b} {n} {r.true:+.0f}/{r.false:+.0f}" for (b, n), r in at.iterrows()))

# %% Plot the verdicts
SLOT = {"−θ_mm (toward 'false')": 0, "v_dishonest": 1, "v_mystic": 2, "v_pirate": 3}   # random: the dashed control
fig, axes = savoy.figure(1, 2, height=300, ytitle="verdict Δ = log P(True) − log P(False)", xtitle="push (× stream length)",
                         titles=[f"block {b}" for b in BLOCKS], sharey=True)
for ax, b in zip(axes, BLOCKS):
    for n in [*SLOT, "random"]:
        for k in ("true", "false"):   # true statements start high, false ones low
            g = S[(S.push == n) & (S.block == b) & (S.statements == k)]
            if len(g):
                savoy.line(ax, g.frac, g.delta, savoy.color(SLOT[n]) if n in SLOT else None, points=True, baseline=n not in SLOT)
    savoy.ref(ax, 0)
    savoy.xticks(ax, 0, 0.6)
savoy.note(axes[0], 0.01, float(base["true"].mean()) + 4, "true statements")
savoy.note(axes[0], 0.01, float(base["false"].mean()) - 7, "false statements")
savoy.yticks(axes[0], S.delta.min() - 7, S.delta.max() + 4, include=[0])
for n in SLOT:
    savoy.key(fig, n, savoy.color(SLOT[n]))
savoy.key(fig, "random", savoy.T["sn"], "base")
savoy.save(fig, "figures/theta_verdict_push.png")

# %% Push θ_mm and judge three ways
import json

import judge as J

STEP3_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
GEN = dict(max_new_tokens=500, batch_size=60, seed=0, top_k=20, repetition_penalty=1.05)   # Exp 1's strength curve
dis = pv.load_trait("dishonest")
qs = dis["questions_eval"]
jx = json.loads((pv.DATA_DIR / "exp2_judges.json").read_text())
JUDGES = {"dishonesty": dis["eval_prompt"], "honesty talk": jx["honesty_talk"], "false content": jx["content_false"]}
V3 = {"−θ_mm (toward 'false')": torch.zeros(len(h_lens), M.shape[1]), "v_dishonest": pv.load_persona("dishonest")["vectors"].float()}
V3["−θ_mm (toward 'false')"][BLOCK] = -D["θ_mm (truth)"]
judge = J.Judge()
parts = []
for n, V in V3.items():
    g = pv.cached_df(f"theta_step3_{pv.key(V[BLOCK], BLOCK, STEP3_FRACS, float(h_lens[BLOCK]), GEN)}",
                     lambda: pv.steering_sweep(model, tok, qs, V, [BLOCK], STEP3_FRACS, scale=h_lens, n_samples=3,
                                               positions="all", device=device, **GEN))
    g = judge.score_df(g, JUDGES["dishonesty"]).rename(columns={"trait_score": "dishonesty"})
    for name in ("honesty talk", "false content"):
        g[name] = judge.score_df(g, JUDGES[name], coherence=False)["trait_score"]
    parts.append(g.assign(push=n, frac=g.frac.where(g.layer >= 0, 0.0)))
G = pd.concat(parts, ignore_index=True)
SCORES = ["dishonesty", "honesty talk", "false content", "coherence"]
T = G.groupby(["push", "frac"])[SCORES].mean().round(0)
display(T)
t3 = T.xs(0.3, level="frac")
log.info("at 0.3 × stream (dishonesty / honesty talk / false content / coherence): %s",
         " · ".join(f"{n} {r.dishonesty:.0f}/{r['honesty talk']:.0f}/{r['false content']:.0f}/{r.coherence:.0f}" for n, r in t3.iterrows()))

# %% Plot and read the answers
def errline(ax, x, m, s=None, c=None):
    """Line with points and ±1 SE bars; c=None: the dashed control in sn."""
    savoy.line(ax, x, m, c, baseline=c is None, points=True, yerr=s)


def judge_panels(df, scores, colors, path, xtitle, by="frac"):
    """One panel per judge score, a line per push (mean ± 1 SE per strength); colors: push -> slot (None = control)."""
    fig, axes = savoy.figure(2, 2, height=170, ytitle="mean judge score", xtitle=xtitle, titles=scores, sharey=True)
    for ax, col in zip(axes, scores):
        for n, g in df.groupby("push", sort=False):
            m = g.groupby(by)[col].agg(["mean", "sem"])
            errline(ax, m.index, m["mean"], m["sem"], colors[n])
        savoy.xticks(ax, 0, df[by].max())
    savoy.yticks(axes[0], 0, 100)
    for n, c in colors.items():
        savoy.key(fig, n, c or savoy.T["sn"], "line" if c else "base")
    savoy.save(fig, path)


judge_panels(G, SCORES, {n: savoy.color(i) for i, n in enumerate(G.push.unique())}, "figures/theta_push_three_judges.png",
             f"push (× stream length, block {BLOCK})")
for n, g in G[G.frac.isin([0.3, 0.5])].groupby("push", sort=False):
    for _, r in g.sample(4, random_state=0).sort_values("frac").iterrows():
        print(f"--- {n} · {r.frac:g} | dishonesty {r.dishonesty} · honesty talk {r['honesty talk']} · false {r['false content']} · "
              f"coherence {r.coherence}\nQ: {r.question}\nA: {' '.join(r.response.split())[:400]}\n")

# %% Pirate leak test
LEAK_BLOCK = pv.load_persona("pirate")["layer"]   # pirate's own pick (A at 18)
V4 = {"v_pirate": pv.load_persona("pirate")["vectors"].float(), "random": torch.zeros(len(h_lens), M.shape[1])}
V4["random"][LEAK_BLOCK] = rand[0]
JUDGES4 = {"dishonesty": dis["eval_prompt"], "false content": jx["content_false"], "pirate-ness": pv.load_trait("pirate")["eval_prompt"]}
parts = []
for n, V in V4.items():
    g = pv.cached_df(f"theta_step4_{pv.key(V[LEAK_BLOCK], LEAK_BLOCK, STEP3_FRACS, float(h_lens[LEAK_BLOCK]), GEN)}",
                     lambda: pv.steering_sweep(model, tok, qs, V, [LEAK_BLOCK], STEP3_FRACS, scale=h_lens, n_samples=3,
                                               positions="all", device=device, **GEN))
    g = judge.score_df(g, JUDGES4["dishonesty"]).rename(columns={"trait_score": "dishonesty"})
    for name in ("false content", "pirate-ness"):
        g[name] = judge.score_df(g, JUDGES4[name], coherence=False)["trait_score"]
    parts.append(g.assign(push=n, frac=g.frac.where(g.layer >= 0, 0.0)))
P = pd.concat(parts, ignore_index=True)
SCORES4 = ["dishonesty", "false content", "pirate-ness", "coherence"]
T4 = P.groupby(["push", "frac"])[SCORES4].agg(["mean", "sem"]).round(1)
display(T4)
m, se = T4.xs("mean", axis=1, level=1), T4.xs("sem", axis=1, level=1)
gap = (m.loc["v_pirate", "dishonesty"] - m.loc["random", "dishonesty"])
err = np.sqrt(se.loc["v_pirate", "dishonesty"] ** 2 + se.loc["random", "dishonesty"] ** 2)
working = m.loc["v_pirate", "pirate-ness"] > m.loc["random", "pirate-ness"] + 20   # "pirate-ness clearly up"
leak = working & (gap > 10) & (gap > 2 * err)
log.info("pirate − random dishonesty by strength: %s · leak (pirate-ness up, gap > 10 and > 2 SE): %s",
         " · ".join(f"{f:g}: {g:+.0f}" for f, g in gap.items()), "YES at " + ", ".join(f"{f:g}" for f in leak[leak].index) if leak.any() else "no")

judge_panels(P, SCORES4, {"v_pirate": savoy.color(0), "random": None}, "figures/pirate_leak.png",
             f"push (× stream length, block {LEAK_BLOCK})")
for _, r in P[(P.push == "v_pirate") & P.frac.isin([0.3, 0.5])].sample(4, random_state=0).sort_values("frac").iterrows():
    print(f"--- pirate · {r.frac:g} | dishonesty {r.dishonesty} · false {r['false content']} · pirate {r['pirate-ness']} · "
          f"coherence {r.coherence}\nQ: {r.question}\nA: {' '.join(r.response.split())[:400]}\n")

# %% Leak follow-up: pirate without dishonest, random matched on coherence
pir, dsh = (F.normalize(pv.load_persona(t)["vectors"][LEAK_BLOCK].float(), dim=0) for t in ("pirate", "dishonest"))
V5 = {"v_pirate − dishonest": torch.zeros(len(h_lens), M.shape[1]), "random": V4["random"]}
V5["v_pirate − dishonest"][LEAK_BLOCK] = F.normalize(pir - (pir @ dsh) * dsh, dim=0)   # the plan's projection
log.info("cos(v_pirate, v_dishonest) at block %d: %.2f, removed", LEAK_BLOCK, float(pir @ dsh))
RAND_FRACS = [0.8, 1.0, 1.2]   # random breaks coherence far later than pirate (78 vs 43 at 0.5)
parts = []
for n, V, fr in [("v_pirate − dishonest", V5["v_pirate − dishonest"], STEP3_FRACS), ("random", V5["random"], RAND_FRACS)]:
    g = pv.cached_df(f"theta_step4b_{pv.key(V[LEAK_BLOCK], LEAK_BLOCK, fr, float(h_lens[LEAK_BLOCK]), GEN)}",
                     lambda: pv.steering_sweep(model, tok, qs, V, [LEAK_BLOCK], fr, scale=h_lens, n_samples=3, include_baseline=False,
                                               positions="all", device=device, **GEN))
    g = judge.score_df(g, JUDGES4["dishonesty"]).rename(columns={"trait_score": "dishonesty"})
    for name in ("false content", "pirate-ness"):
        g[name] = judge.score_df(g, JUDGES4[name], coherence=False)["trait_score"]
    parts.append(g.assign(push=n))
P2 = pd.concat([P, *parts], ignore_index=True)
T5 = P2.groupby(["push", "frac"])[SCORES4].agg(["mean", "sem"]).round(1)
display(T5)
by = P2[P2.frac > 0].groupby(["push", "frac"])[SCORES4].mean()
near = lambda push, coh: by.loc[push].iloc[(by.loc[push].coherence - coh).abs().argsort()[:1]]
for coh in (58, 43):
    log.info("at coherence ≈ %d, dishonesty / pirate-ness: %s", coh, " · ".join(
        f"{p} {float(r.dishonesty):.0f}/{float(r['pirate-ness']):.0f} (at {f:g}, coh {float(r.coherence):.0f})"
        for p in ("v_pirate", "v_pirate − dishonest", "random") for f, r in near(p, coh).iterrows()))

# first three panels: x = coherence (judge), falling to the right; last: x = push (× stream length, block LEAK_BLOCK)
COL5 = {"v_pirate": savoy.color(0), "v_pirate − dishonest": savoy.color(1), "random": None}
fig, axes = savoy.figure(2, 2, height=170, ytitle="mean judge score", sharey=True,
                         titles=[f"{c}, by coherence" for c in ("dishonesty", "false content", "pirate-ness")] + ["coherence, by push"])
for n, g in P2[P2.frac > 0].groupby("push", sort=False):
    mm = g.groupby("frac")[SCORES4].mean()
    for ax, col in zip(axes, ["dishonesty", "false content", "pirate-ness"]):
        errline(ax, mm.coherence, mm[col], c=COL5[n])
    errline(axes[3], mm.index, mm.coherence, c=COL5[n])
for ax in axes[:3]:
    savoy.xticks(ax, 0, 100); ax.invert_xaxis()
savoy.xticks(axes[3], 0, P2.frac.max())
savoy.yticks(axes[0], 0, 100)
for n, c in COL5.items():
    savoy.key(fig, n, c or savoy.T["sn"], "line" if c else "base")
savoy.save(fig, "figures/pirate_leak_matched.png")
