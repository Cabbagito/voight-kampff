# Exp 3: the Assistant Axis on Qwen3-4B. Flow doc: 03_assistant_axis.md. Draft: nothing run yet.
# Needs the paper's repo: git clone --depth 1 https://github.com/safety-research/assistant-axis papers/assistant-axis
# %% Build the axis: role answers
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display

import judge as J
import pv
import savoy

logging.basicConfig(level=logging.INFO, format="· %(message)s")   # no-op inside esper
log = logging.getLogger(__name__)

AA = pv.TOPIC_DIR / "papers" / "assistant-axis"
FIG = pv.TOPIC_DIR / "figures"
BLOCK = 18                      # middle of 36 blocks: the paper reads the middle layer
N_PROMPTS, N_QUESTIONS = 2, 10  # per role (paper: 5 × 240)
N_DEFAULT_SAMPLES = 4           # per (neutral prompt, question); 5 neutral prompts
MIN_FULL = 5                    # role kept if ≥ this many score-3 answers (paper: 10 of 1,200)
GEN = dict(max_new_tokens=300, batch_size=64, seed=0)

roles = {p.stem: json.loads(p.read_text()) for p in sorted((AA / "data/roles/instructions").glob("*.json"))}
default = roles.pop("default")
all_q = [json.loads(line)["question"] for line in (AA / "data/extraction_questions.jsonl").read_text().splitlines()]
qids = sorted(np.random.default_rng(0).choice(len(all_q), N_QUESTIONS, replace=False).tolist())

rows = [dict(role=r, wording=w, qid=q, question=all_q[q], system=spec["instruction"][w]["pos"], sample=0)
        for r, spec in roles.items() for w in range(N_PROMPTS) for q in qids]
rows += [dict(role="default", wording=w, qid=q, question=all_q[q], sample=s,
              system=d["pos"].replace("{model_name}", "Qwen"))
         for w, d in enumerate(default["instruction"]) for q in qids for s in range(N_DEFAULT_SAMPLES)]
table = pd.DataFrame(rows).assign(history=[[] for _ in rows])

model, tok, device = pv.load_model()
df = pv.cached_df(f"axis_rollouts_{pv.key(table[['role', 'system', 'question', 'sample']].to_json(), GEN)}",
                  lambda: pv.generate_rollouts(model, tok, table, **GEN))
log.info("%d answers: %d roles × %d, default %d · truncated %.0f%%", len(df), len(roles), N_PROMPTS * N_QUESTIONS,
         (df.role == "default").sum(), 100 * df.truncated.mean())

# %% Build the axis: judge role adherence
judge = J.Judge()
is_role = df.role != "default"
raw = judge.raw([J.fill(roles[r]["eval_prompt"], q, a)
                 for r, q, a in zip(df.role[is_role], df.question[is_role], df.response[is_role])])
df.loc[is_role, "role_score"] = [J.parse_score(x) for x in raw]   # 0 refuse · 1 plain AI · 2 AI with the role's flavour · 3 fully the role
full = df[is_role & (df.role_score == 3)].groupby("role").size()
kept = sorted(full[full >= MIN_FULL].index)
log.info("score 3 on %.0f%% of role answers · %d of %d roles kept (≥%d at score 3) · judge $%.2f",
         100 * (df.role_score[is_role] == 3).mean(), len(kept), len(roles), MIN_FULL, judge.cost())

# %% Build the axis: vectors and PCA
name = f"axis_acts_{pv.key(df.response.to_json())}"
if (pv.CACHE_DIR / f"{name}.pt").exists():
    acts = pv.load_tensor(name)
else:
    acts = pv.collect_reduced(model, tok, df.prompt.tolist(), df.response.tolist(), pv.reduce_tokens).half()   # ~1 GB
    pv.save_tensor(acts, name)
acts = acts.float()   # (N answers, 36 blocks, d): each answer's mean over its tokens
SPIKE_DIMS = [4, 100]   # the Exp 1 spike-token dims: zeroed, not answers dropped (Peter, 2026-10-31)
odd = torch.stack([pv.extremes(acts[:, b]).any(1) for b in range(acts.shape[1])]).any(0).numpy()
acts[:, :, SPIKE_DIMS] = 0
left = torch.stack([pv.extremes(acts[:, b]).any(1) for b in range(acts.shape[1])]).any(0).numpy()
log.info("extremes check: %d answers flagged; %d still flagged with dims %s zeroed (all answers kept)", odd.sum(), left.sum(), SPIKE_DIMS)

def role_vectors(score):
    """Mean answer activation per role, over its answers at `score`; roles with < MIN_FULL such answers skipped."""
    out = {}
    for r in roles:
        m = (df.role == r).to_numpy() & (df.role_score == score).to_numpy()
        if m.sum() >= MIN_FULL:
            out[r] = acts[m].mean(0)
    return out

default_vec = acts[(df.role == "default").to_numpy()].mean(0)                            # (36, d)
fully, somewhat = role_vectors(3), role_vectors(2)
kept = list(fully)
role_vecs = torch.stack(list(fully.values()))                                             # (R, 36, d)
axis = default_vec - role_vecs.mean(0)                                                    # (36, d), + = toward the Assistant
pca_vecs = torch.cat([role_vecs, torch.stack(list(somewhat.values()))])                   # the paper's PCA: fully + somewhat
pca_names = [f"fully {r}" for r in fully] + [f"somewhat {r}" for r in somewhat]

def pc1(V, b):
    """First principal component of the vectors V[:, b], signed toward the Assistant end of the axis."""
    _, S, Vt = torch.linalg.svd(V[:, b] - V[:, b].mean(0), full_matrices=False)
    return Vt[0] * torch.sign(Vt[0] @ axis[b]), float(S[0] ** 2 / (S**2).sum())

cos_pc1 = {k: [float(F.cosine_similarity(pc1(V, b)[0], axis[b], dim=0)) for b in range(acts.shape[1])]
           for k, V in (("fully", role_vecs), ("both", pca_vecs))}
p, var1 = pc1(pca_vecs, BLOCK)
on_pc1 = pd.Series((pca_vecs[:, BLOCK] @ p).numpy(), pca_names)
d0 = float(default_vec[BLOCK] @ p)
where = (d0 - on_pc1.min()) / (on_pc1.max() - on_pc1.min())   # paper: 0.03 from the Assistant edge
cos_role = pd.Series(F.cosine_similarity(role_vecs[:, BLOCK] - default_vec[BLOCK], -axis[BLOCK][None], dim=1).numpy(), kept)
proj_all = torch.einsum("nbd,bd->nb", acts, F.normalize(axis, dim=1))   # every answer's projection, per block
pv.save_tensor(dict(axis=axis, default=default_vec, role_vecs=role_vecs, roles=kept, proj_all=proj_all,
                    spike_dims=SPIKE_DIMS), "assistant_axis")

blocks = list(range(acts.shape[1]))
fig = savoy.chart("line", {
    "x": {"title": "block (layer 0–35)", "values": blocks},
    "y": {"title": "cos(PC1 of persona space, axis)", "include": [0, 1]},
    "series": [{"id": "fully", "label": f"PCA on {len(fully)} fully-in-role vectors", "short": "fully", "values": cos_pc1["fully"]},
               {"id": "both", "label": f"+ {len(somewhat)} somewhat-in-role (paper)", "short": "+ somewhat", "values": cos_pc1["both"]}],
}, FIG / "axis_pc1_cos.png")
plt.show()

bins = np.linspace(on_pc1.min(), on_pc1.max(), 31)
fig = savoy.chart("histogram", {
    "x": {"title": f"position on PC1 at block {BLOCK} (+ = Assistant end)"}, "y": {"title": "role vectors"}, "bins": bins,
    "series": [{"id": s, "label": f"{s} in role", "counts": np.histogram(on_pc1[on_pc1.index.str.startswith(s)], bins)[0]}
               for s in ("fully", "somewhat")],
})
savoy.event(fig.axes[0], d0, "default Assistant")
savoy.save(fig, FIG / "axis_pc1_roles.png")
plt.show()
log.info("block %d: cos(PC1, axis) %.2f fully-only → %.2f with somewhat · PC1 explains %.0f%% · default at %.2f of PC1's range "
         "(1 = Assistant end), %d vectors past it · furthest roles from the Assistant: %s · closest: %s",
         BLOCK, cos_pc1["fully"][BLOCK], cos_pc1["both"][BLOCK], 100 * var1, where, (on_pc1 > d0).sum(),
         ", ".join(cos_role.nlargest(5).index), ", ".join(cos_role.nsmallest(5).index))

# %% Place what we have
LP = pv.CACHE_DIR.parent.parent / "linear-probes" / "cache"
P = {t: pv.load_persona(t)["vectors"].float() for t in ("dishonest", "pirate", "mystic")}   # (36, d) each, + = toward the trait
P["shared"] = torch.stack([F.normalize(v, dim=1) for v in P.values()]).mean(0)            # Exp 2's "character" part
for r in ("pirate", "mystic"):
    P[f"{r} role"] = fully[r] - default_vec                                               # the paper's role, not our trait
cos = pd.DataFrame({k: F.cosine_similarity(v, axis, dim=1).numpy() for k, v in P.items()})   # (36 blocks, 6)
theta = torch.as_tensor(np.asarray(torch.load(LP / "truth_probe_L20.pt", weights_only=False)["theta_mm"]), dtype=torch.float32)
cos_theta = float(F.cosine_similarity(theta, axis[20], dim=0))
null = float((F.normalize(torch.randn(2000, axis.shape[1], generator=torch.Generator().manual_seed(0)), dim=1)
              @ F.normalize(axis[BLOCK], dim=0)).abs().quantile(.95))
display(cos.loc[[BLOCK, 20, 24]].round(2))

fig = savoy.chart("line", {
    "x": {"title": "block (layer 0–35)", "values": blocks},
    "y": {"title": "cosine with the Assistant Axis (+ = Assistant)", "include": [0]},
    "refs": [{"y": 0}],
    "events": [{"x": BLOCK, "label": f"block {BLOCK}"}],
    "series": [{"id": k, "label": f"v_{k}" if k in ("dishonest", "pirate", "mystic") else k, "values": cos[k].tolist()}
               for k in ("dishonest", "pirate", "mystic", "shared")],
}, FIG / "axis_vs_persona_vectors.png")
plt.show()
log.info("cos with the axis at block %d (random |cos| 95th pct %.3f): %s · roles: pirate %+.2f, mystic %+.2f · θ_mm (block 20) %+.2f",
         BLOCK, null, " · ".join(f"{k} {cos.loc[BLOCK, k]:+.2f}" for k in ("dishonest", "pirate", "mystic", "shared")),
         cos.loc[BLOCK, "pirate role"], cos.loc[BLOCK, "mystic role"], cos_theta)

# %% Drift: replay
T = AA / "transcripts"
FILES = {"coding": "persona_drift/coding.json", "writing": "persona_drift/writing.json",
         "therapy": "persona_drift/therapy.json", "philosophy": "persona_drift/philosophy.json",
         "delusion": "case_studies/qwen-3-32b/delusion_unsteered.json",
         "selfharm": "case_studies/qwen-3-32b/selfharm_unsteered.json.json"}
convs = {k: [m["content"] for m in json.loads((T / f).read_text())["conversation"] if m["role"] == "user"]
         for k, f in FILES.items()}
DRIFT_GEN = dict(max_new_tokens=400, batch_size=8, seed=0)
CAP_BLOCKS = [26, 27, 28, 29]   # ~72-83% depth, the paper's Qwen3-32B setting scaled
v_dis = pv.load_persona("dishonest")["vectors"][BLOCK].float().clone()
v_dis[SPIKE_DIMS] = 0
a_hat = F.normalize(axis[BLOCK], dim=0)
v_dis_perp = v_dis - (v_dis @ a_hat) * a_hat   # dishonest without its axis slice (Peter: is the drift dishonesty, or just "not the Assistant"?)

def run_replay(names, tag, cap=None):
    """Replay `names` (uncapped, or under pv.cap with `cap` = (blocks, taus)); project each answer on the axis."""
    def go():
        if cap is None:
            return pv.replay(model, tok, [convs[n] for n in names], **DRIFT_GEN)
        with pv.cap(model, cap[0], axis, cap[1]) as stats:
            out = pv.replay(model, tok, [convs[n] for n in names], **DRIFT_GEN)
        log.info("cap fired on %s of positions", " · ".join(f"block {b} {100 * s['capped'] / max(s['seen'], 1):.1f}%" for b, s in stats.items()))
        return out
    d = pv.cached_df(f"axis_drift_{tag}_{pv.key(names, DRIFT_GEN, axis if cap else 0, cap[1] if cap else 0)}", go)
    d["domain"] = [names[i] for i in d.conv]
    a = pv.collect_reduced(model, tok, d.prompt.tolist(), d.response.tolist(), pv.reduce_tokens, batch_size=1)
    a[:, :, SPIKE_DIMS] = 0   # as in step 1
    for b in sorted({BLOCK, *CAP_BLOCKS}):
        d[f"proj_{b}"] = pv.project(a[:, b], axis[b])
    d["dis"], d["dis_perp"] = pv.project(a[:, BLOCK], v_dis), pv.project(a[:, BLOCK], v_dis_perp)
    return d

drift = run_replay(list(convs), "free")
first, last = drift[drift.turn < 3].groupby("domain")[f"proj_{BLOCK}"].mean(), drift.groupby("domain")[f"proj_{BLOCK}"].apply(lambda s: s.iloc[-3:].mean())
log.info("block %d projection, first 3 turns → last 3: %s", BLOCK, " · ".join(f"{k} {first[k]:.1f} → {last[k]:.1f}" for k in convs))
r = drift[[f"proj_{BLOCK}", "dis", "dis_perp"]].corr()
log.info("across all turns, r with the axis projection: v_dishonest %.2f · v_dishonest ⊥ axis %.2f (cos(v_dishonest, axis) %.2f)",
         r.iloc[0, 1], r.iloc[0, 2], float(F.cosine_similarity(v_dis, axis[BLOCK], dim=0)))

# %% Drift: judge and plot
JUDGES = json.loads((pv.DATA_DIR / "exp3_judges.json").read_text())
TAU_PCT = 75   # the paper's "25th percentile" is on its flipped vector (role-play +); on our axis (+ = Assistant) that's the 75th (checked on its Qwen-32B config, 2026-10-31)
tau = np.percentile(proj_all.numpy(), TAU_PCT, axis=0)   # (36,), per block
default_med = proj_all[(df.role == "default").to_numpy()].median(0).values.numpy()
log.info("τ (%dth pct) vs the default Assistant's median: %s", TAU_PCT,
         " · ".join(f"block {b} {tau[b]:+.1f} vs {default_med[b]:+.1f}" for b in [BLOCK, *CAP_BLOCKS]))

def context(d):
    """The judge's view: the last two exchanges and the current user message."""
    out = []
    for _, g in d.groupby("conv"):
        g = g.sort_values("turn")
        pairs = list(zip(g.user, g.response))
        out += ["\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in pairs[max(0, t - 2):t]) + f"\n\nUser: {pairs[t][0]}"
                for t in range(len(pairs))]
    return pd.Series(out, index=d.sort_values(["conv", "turn"]).index)

def judged(d):
    d = d.drop(columns=["harmful_drift", "persona_break", "coherence", "refusal", "trait_raw"], errors="ignore")   # safe to re-run
    d = d.assign(context=context(d))
    d = judge.score_df(d, JUDGES["harmful_drift"], q_col="context").rename(columns={"trait_score": "harmful_drift"})
    d["persona_break"] = judge.score_df(d, JUDGES["persona_break"], coherence=False, q_col="context")["trait_score"]
    return d

drift = judged(drift)
DOMAINS = list(convs)
for col, title, name in ((f"proj_{BLOCK}", f"projection on the axis, block {BLOCK} (τ: {TAU_PCT}th pct)", "axis_drift_projection"),
                         ("harmful_drift", "harmful-drift judge (0-100)", "axis_drift_harm")):
    fig, axs = savoy.figure(2, 3, height=190, ytitle=title, xtitle="turn", titles=DOMAINS)
    lo, hi = drift[col].min(), drift[col].max()
    for ax, k in zip(axs, DOMAINS):
        g = drift[drift.domain == k].sort_values("turn")
        savoy.line(ax, g.turn, g[col], savoy.color(0), points=True)
        if col.startswith("proj"):
            savoy.ref(ax, tau[BLOCK], "τ", threshold=True)
        savoy.yticks(ax, *((lo, hi) if col.startswith("proj") else (0, 100)), include=[tau[BLOCK]] if col.startswith("proj") else [], n=4)
        savoy.xticks(ax, 0, drift.turn.max(), n=4)
    savoy.save(fig, FIG / f"{name}.png")
    plt.show()
fig, axs = savoy.figure(2, 3, height=190, ytitle=f"projection, block {BLOCK}, change since turn 1", xtitle="turn", titles=DOMAINS)
lo = min((drift[c] - drift.groupby("domain")[c].transform("first")).min() for c in ("dis", "dis_perp"))
hi = max((drift[c] - drift.groupby("domain")[c].transform("first")).max() for c in ("dis", "dis_perp"))
for ax, k in zip(axs, DOMAINS):
    g = drift[drift.domain == k].sort_values("turn")
    for i, c in enumerate(("dis", "dis_perp")):
        savoy.line(ax, g.turn, g[c] - g[c].iloc[0], savoy.color(i))
    savoy.ref(ax, 0)
    savoy.yticks(ax, lo, hi, include=[0], n=4)
    savoy.xticks(ax, 0, drift.turn.max(), n=4)
savoy.key(fig, "v_dishonest", savoy.color(0))
savoy.key(fig, "v_dishonest with its axis slice removed", savoy.color(1))
savoy.save(fig, FIG / "axis_drift_dishonest.png")
plt.show()
risky = drift[~drift.domain.isin(["coding", "writing"])].groupby("domain").harmful_drift.mean()
WORST = "selfharm"   # Peter, 2026-10-31: longest, and its harm builds over turns (by mean harm it would be risky.idxmax(): delusion)
log.info("harmful drift (mean over turns): %s · persona break: %s · worst: %s · r(projection, harm) %.2f",
         " · ".join(f"{k} {v:.0f}" for k, v in drift.groupby("domain").harmful_drift.mean().items()),
         " · ".join(f"{k} {v:.0f}" for k, v in drift.groupby("domain").persona_break.mean().items()),
         WORST, drift[[f"proj_{BLOCK}", "harmful_drift"]].corr().iloc[0, 1])

# %% Cap: replay capped
capped = judged(run_replay([WORST, "coding"], "capped", cap=(CAP_BLOCKS, tau)))
log.info("capped vs free, mean harmful drift: %s",
         " · ".join(f"{k} {drift[drift.domain == k].harmful_drift.mean():.0f} → {capped[capped.domain == k].harmful_drift.mean():.0f}"
                    for k in (WORST, "coding")))

# %% Cap: compare
g0, g1 = (d[d.domain == WORST].sort_values("turn") for d in (drift, capped))
fig, axs = savoy.figure(1, 3, height=220, xtitle="turn", titles=[f"projection, block {BLOCK}", "harmful drift", "coherence"])
for ax, col in zip(axs, (f"proj_{BLOCK}", "harmful_drift", "coherence")):
    savoy.line(ax, g0.turn, g0[col], savoy.color(0))
    savoy.line(ax, g1.turn, g1[col], savoy.color(1))
    lo, hi = min(g0[col].min(), g1[col].min()), max(g0[col].max(), g1[col].max())
    savoy.yticks(ax, *((lo, hi) if col.startswith("proj") else (0, 100)), include=[tau[BLOCK]] if col.startswith("proj") else [], n=4)
    savoy.xticks(ax, 0, g0.turn.max(), n=4)
savoy.ref(axs[0], tau[BLOCK], "τ", threshold=True)
savoy.key(fig, f"{WORST}, free", savoy.color(0))
savoy.key(fig, f"{WORST}, capped at blocks {CAP_BLOCKS[0]}–{CAP_BLOCKS[-1]}", savoy.color(1))
savoy.save(fig, FIG / "axis_cap_compare.png")
plt.show()
log.info("coherence free → capped: %s · judge total $%.2f",
         " · ".join(f"{k} {drift[drift.domain == k].coherence.mean():.0f} → {capped[capped.domain == k].coherence.mean():.0f}"
                    for k in (WORST, "coding")), judge.cost())
