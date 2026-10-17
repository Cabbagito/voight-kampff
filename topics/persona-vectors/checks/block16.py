# Why every curve breaks at block 16. Run inside the dishonest esper session after
# 01_extract.py Parts 1-3 (see checks/README.md). Writes three figures into figures/.
#  1. stability and separation over 20 random half-splits (pooled vs matched);
#  2. four measurements per block: signal-to-noise, top-5-dim share, rotation, update size;
#  3. the top-5-dim share on other inputs (pirate/mystic answers, plain statements, random
#     tokens), plus which answers are the loud ones, by condition and question.
# Finding (2026-10-17): a rare, content-triggered spike; see experiments.md.

# ---- dishonest_stability_separation ----------------------------------------
# Two checks on the kept dishonest answers (first-40 activations), pooled vs matched:
#  1. Stability: build each vector from two random halves of the answers, 20 different
#     splits. Is the dip at blocks 15-17 in the half-vs-half line real or one unlucky split?
#  2. Separation: fit each vector on one half, then score the other half. Within each
#     question, how far apart are the lies and the honest answers along the vector
#     (Cohen's d, weighted by the question's number of lies)?
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

log = logging.getLogger(__name__)
F = torch.nn.functional.cosine_similarity
a = acts_all[idx]                                   # kept POS + NEG, (N, 36, d)
cond, qid = lab.condition.to_numpy(), lab.qid.to_numpy()


def split(seed):
    h1 = lab.groupby(["condition", "qid"]).sample(frac=.5, random_state=seed).index.to_numpy()
    m = np.zeros(len(lab), bool)
    m[h1] = True
    return m


def vec(m, match):
    return pv.persona_vectors(a[torch.from_numpy(m)], lab[m].reset_index(drop=True), match_questions=match)


def within_question_d(P, test):
    ds, ws = [], []
    for q in np.unique(qid):
        pos, neg = test & (qid == q) & (cond == "POS"), test & (qid == q) & (cond == "NEG")
        if pos.sum() >= 2 and neg.sum() >= 2:
            pp, nn = P[torch.from_numpy(pos)], P[torch.from_numpy(neg)]
            ds.append((pp.mean(0) - nn.mean(0)) / torch.sqrt((pp.var(0) + nn.var(0)) / 2))
            ws.append(float(pos.sum()))
    w = torch.tensor(ws)
    return (torch.stack(ds) * w[:, None]).sum(0) / w.sum()


stab = {"pooled": [], "matched": []}
sep = {"pooled": [], "matched": []}
for seed in range(20):
    m = split(seed)
    for kind, match in (("pooled", False), ("matched", True)):
        v1, v2 = vec(m, match), vec(~m, match)
        stab[kind].append(F(v1, v2, dim=-1))
        P = torch.einsum("nld,ld->nl", a, v1)          # every answer projected on the half-1 vector
        sep[kind].append(within_question_d(P, ~m))      # scored on half 2 only
stab = {k: torch.stack(v) for k, v in stab.items()}
sep = {k: torch.stack(v) for k, v in sep.items()}

blocks = np.arange(a.shape[1])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for kind, c in (("pooled", "C0"), ("matched", "C1")):
    for ax, D in ((axes[0], stab[kind]), (axes[1], sep[kind])):
        ax.plot(blocks, D.mean(0), color=c, label=kind)
        ax.fill_between(blocks, D.min(0).values, D.max(0).values, color=c, alpha=.2)
axes[0].set(xlabel="block (layer, 0-35)", ylabel="cosine between the two halves' vectors",
            title="Stability: vector from half A vs half B\n(20 random splits; line = mean, band = min-max)")
axes[1].set(xlabel="block (layer, 0-35)", ylabel="separation, Cohen's d (held-out half)",
            title="Lies vs honest answers on unseen answers,\nwithin the same question")
for ax in axes:
    ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("figures/dishonest_stability_separation.png", dpi=110)

for kind in ("pooled", "matched"):
    s = stab[kind].mean(0)
    log.info("%s stability: mean cos at blocks 12·14·15·16·17·18·20 = %s · lowest single split at 16: %.2f",
             kind, " ".join(f"{s[L]:.2f}" for L in (12, 14, 15, 16, 17, 18, 20)), stab[kind][:, 16].min())
    d = sep[kind].mean(0)
    best = int(d.argmax())
    log.info("%s separation: best block %d (d = %.2f) · at 6·12·16·20·24·30: %s", kind, best, d[best],
             " ".join(f"{d[L]:.2f}" for L in (6, 12, 16, 20, 24, 30)))

# ---- dishonest_block16 -----------------------------------------------------
# What happens at block 16? Four per-block measurements on the kept dishonest answers
# (first-40 activations, CPU). Story 1 (noise) predicts: signal-to-noise drops at 16, maybe
# with a few huge dimensions eating the variance. Story 2 (rotation) predicts: the vector
# turns sharply between 15 and 16, and block 16 writes an unusually large update.
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

log = logging.getLogger(__name__)
a = acts_all[idx]                                    # (N kept, 36, d)
pos = torch.from_numpy((lab.condition == "POS").to_numpy())
P, Nn = a[pos], a[~pos]
L = a.shape[1]

# 1. signal-to-noise: gap between class means vs typical within-class spread (per block)
gap = (P.mean(0) - Nn.mean(0)).norm(dim=-1)                                   # (L,)
spread = torch.sqrt(((P - P.mean(0)) ** 2).sum(-1).mean(0) / 2 + ((Nn - Nn.mean(0)) ** 2).sum(-1).mean(0) / 2)
snr = gap / spread

# 1b. a few huge dimensions? share of all variance in the top-5 dims, and of the gap
var = a.var(0)                                                                # (L, d)
top5_var = var.topk(5, dim=-1).values.sum(-1) / var.sum(-1)
dmu2 = (P.mean(0) - Nn.mean(0)) ** 2
top5_gap = dmu2.topk(5, dim=-1).values.sum(-1) / dmu2.sum(-1)

# 2. rotation: cosine between neighbouring blocks' vectors, and each block's own update size
Vu = V / V.norm(dim=-1, keepdim=True)
cos_next = (Vu[:-1] * Vu[1:]).sum(-1)                                         # block L vs L+1
upd = ((a[:, 1:] - a[:, :-1]).norm(dim=-1) / a[:, :-1].norm(dim=-1)).mean(0)  # update of block L+1

b = np.arange(L)
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
axes[0, 0].plot(b, snr, marker="o", ms=3)
axes[0, 0].set(ylabel="gap ÷ within-class spread",
               title="1 · Signal-to-noise: lies-vs-honest gap,\nrelative to how much answers vary anyway")
axes[0, 1].plot(b, top5_var, marker="o", ms=3, label="all variance across answers")
axes[0, 1].plot(b, top5_gap, marker="o", ms=3, label="the lies-vs-honest gap")
axes[0, 1].set(ylabel="share held by the top 5 of 2,560 dims", title="1b · Do a few huge dimensions dominate?")
axes[0, 1].legend()
axes[1, 0].plot(b[1:], cos_next, marker="o", ms=3)
axes[1, 0].set(xlabel="block (layer, 0-35)", ylabel="cos(vector at block L−1, at block L)",
               title="2 · Rotation: does the direction turn between neighbours?")
axes[1, 1].plot(b[1:], upd, marker="o", ms=3)
axes[1, 1].set(xlabel="block (layer, 0-35)", ylabel="‖update‖ ÷ ‖incoming residual‖",
               title="2 · How much does each block rewrite the residual?")
for ax in axes.flat:
    ax.axvline(16, color="C3", lw=1, ls=":")
    ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("figures/dishonest_block16.png", dpi=110)

at = (13, 14, 15, 16, 17, 18, 20)
fmt = lambda x, ks=at: " ".join(f"{float(x[k]):.2f}" for k in ks)
log.info("signal-to-noise at blocks %s: %s", "·".join(map(str, at)), fmt(snr))
log.info("top-5-dim share of variance: %s · of the gap: %s", fmt(top5_var), fmt(top5_gap))
log.info("cos with previous block's vector at %s: %s", "·".join(map(str, at)), fmt(cos_next, [k - 1 for k in at]))
log.info("update size of blocks %s: %s", "·".join(map(str, at)), fmt(upd, [k - 1 for k in at]))

# ---- block16_any_input -----------------------------------------------------
# Is block 16's "few huge dimensions" tied to our traits, or does Qwen3-4B do it on anything?
# Same measurement as before (share of all variance across inputs held by the top 5 of 2,560
# dims, per block) on: our three traits' answers (first 40 answer tokens, all conditions),
# plain true/false statements (no chat template), and sequences of random tokens. For the last
# two, also with the first token left out, in case it's a first-position effect.
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


def top5_share(x):                       # x: (n inputs, L, d) -> (L,)
    var = x.float().var(0)
    return var.topk(5, dim=-1).values.sum(-1) / var.sum(-1)


@torch.no_grad()
def hidden_means(texts=None, ids=None, bs=32):
    """Mean residual per block over each input's tokens, all tokens and without the first."""
    tok.padding_side = "left"
    allm, rest = [], []
    n = len(texts) if texts is not None else len(ids)
    for b in range(0, n, bs):
        if texts is not None:
            enc = tok(texts[b:b + bs], return_tensors="pt", padding=True, add_special_tokens=False)
            input_ids, attn = enc.input_ids, enc.attention_mask
        else:
            input_ids = torch.stack(ids[b:b + bs])
            attn = torch.ones_like(input_ids)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        hs = torch.stack(model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                               position_ids=pos.to(device), output_hidden_states=True).hidden_states[1:])  # (L, B, T, d)
        m = attn.to(device)[None, :, :, None].to(hs.dtype)
        first = (pos == 0).to(device)[None, :, :, None] & (attn.to(device)[None, :, :, None] > 0)
        allm.append(((hs * m).sum(2) / m.sum(2)).transpose(0, 1).float().cpu())
        m2 = m * (~first)
        rest.append(((hs * m2).sum(2) / m2.sum(2)).transpose(0, 1).float().cpu())
    return torch.cat(allm), torch.cat(rest)


statements = pd.read_csv("../linear-probes/data/facts_true_false.csv").statement.tolist()[:300]
g = torch.Generator().manual_seed(0)
rand_ids = [torch.randint(1000, 150000, (40,), generator=g) for _ in range(300)]
st_all, st_rest = hidden_means(texts=statements)
rn_all, rn_rest = hidden_means(ids=rand_ids)

sets = {"dishonest answers": pv.load_tensor("acts_dishonest_a63ee90e_v1_first40"),
        "pirate answers": pv.load_tensor("acts_pirate_08531e1a_v1_first40"),
        "mystic answers": pv.load_tensor("acts_mystic_2bda5421_v1_first40"),
        "true/false statements": st_all, "statements, first token left out": st_rest,
        "random tokens": rn_all, "random tokens, first token left out": rn_rest}
shares = {k: top5_share(v) for k, v in sets.items()}

fig, ax = plt.subplots(figsize=(10, 4.5))
styles = ["-", "-", "-", "--", ":", "--", ":"]
for (k, s), ls in zip(shares.items(), styles):
    ax.plot(np.arange(len(s)), s, ls, marker="o", ms=2.5, label=k)
ax.axvline(16, color="C3", lw=1, ls=":")
ax.set(xlabel="block (layer, 0-35)", ylabel="share of variance held by the top 5 of 2,560 dims",
       title="Does Qwen3-4B 'go wild' at block 16 on any input?")
ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("figures/block16_any_input.png", dpi=110)

for k, s in shares.items():
    log.info("%s: top-5 share at blocks 14·15·16·17·20 = %s", k, " ".join(f"{s[L]:.2f}" for L in (14, 15, 16, 17, 20)))
top = {k: set(v.float().var(0)[16].topk(5).indices.tolist()) for k, v in sets.items()}
ref = top["dishonest answers"]
log.info("block 16's top-5 dims shared with dishonest's: %s", " · ".join(f"{k} {len(ref & t)}/5" for k, t in top.items() if k != "dishonest answers"))

# ---- which answers are loud at block 16 ------------------------------------------------
import numpy as np  # noqa: E402
for name, key in (("dishonest", "dishonest_a63ee90e"), ("mystic", "mystic_2bda5421"), ("pirate", "pirate_08531e1a")):
    acts = pv.load_tensor(f"acts_{key}_v1_first40")
    roll_ = pv.load_df(f"rollouts_{key}")
    top = acts[:, 16].var(0).topk(5).indices
    loud = acts[:, 16, top].abs().sum(-1).numpy()
    roll_ = roll_.assign(outlier=loud > 1.5 * np.median(loud))
    by_c = roll_.groupby("condition").outlier.sum().to_dict()
    by_q = roll_[roll_.outlier].groupby("qid").size().sort_values(ascending=False).head(4).to_dict()
    log.info("%s: %d loud answers (> 1.5x median) of %d · by condition %s · by question %s",
             name, roll_.outlier.sum(), len(roll_), by_c, by_q)
