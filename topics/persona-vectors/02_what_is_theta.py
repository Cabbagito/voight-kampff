# Exp 2: what is θ_mm? Compares linear-probes' truth direction with Exp 1's persona vectors. Flow doc: 02_what_is_theta.md.
# %% Load the directions
import logging

import numpy as np
import pandas as pd
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

import matplotlib.pyplot as plt

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
pushes = list(dict.fromkeys(S.push))
fig, axes = plt.subplots(1, len(pushes), figsize=(3.2 * len(pushes), 3.6), sharey=True)
for ax, n in zip(axes, pushes):
    for b, ls in zip(BLOCKS, ["-", "--"]):
        for k, c in [("true", "C2"), ("false", "C3")]:
            g = S[(S.push == n) & (S.block == b) & (S.statements == k)]
            if len(g):
                ax.plot(g.frac, g.delta, ls, marker="o", color=c, label=f"{k} statements, block {b}")
    ax.axhline(0, color="k", lw=.6)
    ax.set(title=n, xlabel="push (× stream length)")
axes[0].set_ylabel("verdict Δ = log P(True) − log P(False)")
axes[0].legend(fontsize=7)
fig.suptitle("What each push does to the model's own true/false verdict (cross = liar, both down = 'looks false', both → 0 = broken)")
plt.tight_layout()
