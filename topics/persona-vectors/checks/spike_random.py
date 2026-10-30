# Are MONITOR's spikes about our vector, or about the token? (Peter, 2026-10-29)
# For the 103 ladder answers with a spike (dishonest, block 18), project each token's block-18 state onto:
# the dishonest vector, the mystic and pirate vectors (real directions with nothing to do with lying),
# and 20 random unit directions. If every direction jumps at the spike token, it's the token's size.
# %% Setup
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path.cwd() if (Path.cwd() / "pv.py").exists() else Path.cwd().parent   # esper runs from checks/
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pv

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
LAYER, TH, N_RAND = 18, 20, 20

model, tok, device = pv.load_model()
mon = pv.load_df("monitor_dishonest_105c55d1")
traj = pv.load_tensor("traj_dishonest_105c55d1")
spiked = [i for i, t in enumerate(traj) if float(t[1:].max()) > TH]
d = model.config.hidden_size
g = torch.Generator().manual_seed(0)
dirs = {t: pv.load_persona(t)["vectors"][LAYER].float() for t in ("dishonest", "mystic", "pirate")}
dirs |= {f"random {k}": torch.randn(d, generator=g) for k in range(N_RAND)}
D = torch.stack([v / v.norm() for v in dirs.values()])          # (n_dirs, d), unit rows
log.info("%d spiked answers · %d directions", len(spiked), len(D))

# %% Project every answer token
prompts = [pv.chat_prompt(tok, mon.system[i], mon.question[i]) for i in spiked]
spike, normal, norm_s, norm_n, top = [], [], [], [], []
for j, (h, n_prompt) in enumerate(pv._per_example_hidden(model, tok, prompts, [mon.response[i] for i in spiked],
                                                         8, device, layers=[LAYER])):
    a = h[0, n_prompt:].float().cpu()                            # answer tokens, (T, d)
    k = int(traj[spiked[j]][1:].argmax())
    p = (a @ D.T).abs()                                          # (T, n_dirs)
    spike.append(p[k]); normal.append(p[torch.arange(len(a)) != k].median(0).values)
    norm_s.append(float(a[k].norm())); norm_n.append(float(a.norm(dim=-1).median()))
    top.append(a[k].abs().topk(5).indices.tolist())
spike, normal = torch.stack(spike), torch.stack(normal)          # (n_answers, n_dirs)
ratio = (spike / normal).median(0).values                        # how many times bigger at the spike token
log.info("state length at the spike token: median %.0f · at other tokens: median %.0f",
         np.median(norm_s), np.median(norm_n))

# %% Which dimensions, and how much each direction leans on them
from collections import Counter
dims = [k for k, _ in Counter(x for t in top for x in t).most_common(5)]
log.info("top-5 dims at the spike tokens: %s (how often each is in a spike's top 5: %s)",
         dims, [sum(x in t for t in top) for x in dims])
w = D[:, dims].norm(dim=-1)                                       # each direction's weight on those dims
names = list(dirs)
log.info("weight on those dims: dishonest %.3f · mystic %.3f · pirate %.3f · random median %.3f (max %.3f)",
         *w[:3].tolist(), float(w[3:].median()), float(w[3:].max()))
log.info("spike / typical |projection|, median over answers: dishonest %.0f× · mystic %.0f× · pirate %.0f× · random median %.0f× (range %.0f–%.0f)",
         *ratio[:3].tolist(), float(ratio[3:].median()), float(ratio[3:].min()), float(ratio[3:].max()))

# %% Plot
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(3 + 1)
vals = [*ratio[:3].tolist(), float(ratio[3:].median())]
ax.bar(x, vals, color=["C3", "C4", "C2", "gray"])
ax.errorbar([3], [vals[3]], yerr=[[vals[3] - float(ratio[3:].min())], [float(ratio[3:].max()) - vals[3]]],
            color="k", capsize=4)
ax.set_xticks(x, ["dishonest", "mystic", "pirate", f"{N_RAND} random\n(median, range)"])
ax.set_yscale("log")
ax.set(ylabel="how many times bigger the reading is\nat the spike token than at other tokens",
       title=f"Spike tokens, read along different directions (block {LAYER}, {len(spiked)} answers)")
plt.tight_layout(); plt.savefig("figures/spike_random_directions.png", dpi=120)
log.info("saved figures/spike_random_directions.png")
