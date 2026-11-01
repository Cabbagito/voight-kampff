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

# %% Cap: one-turn bench
JUDGES = json.loads((pv.DATA_DIR / "exp3_judges.json").read_text())   # re-read: crisis_care was added for this cell
BENCH_TURN = 29   # self-harm: "I want to walk into the ocean and disappear…"
N_BENCH = 10
BENCH_GEN = dict(max_new_tokens=400, batch_size=5, seed=0)
h_len = float(np.mean([pv.load_persona(t)["h_lens"][BLOCK] for t in ("dishonest", "pirate", "mystic")]))   # stream length at BLOCK
sh = drift[drift.domain == "selfharm"].sort_values("turn")
hist = list(zip(sh.user, sh.response))[:BENCH_TURN]   # the free (uncapped) history: the hardest case
user = sh.user.iloc[BENCH_TURN]
tau90 = np.percentile(proj_all.numpy(), 90, axis=0)
a_unit = F.normalize(axis, dim=1)
CONDS = {   # name: (history, system prompt, intervention)
    "no history": ([], None, None),
    "free": (hist, None, None),
    "cap 26–29, τ 75th": (hist, None, ("cap", [26, 27, 28, 29], tau)),
    "cap 22–29, τ 75th": (hist, None, ("cap", list(range(22, 30)), tau)),
    "cap 26–29, τ 90th": (hist, None, ("cap", [26, 27, 28, 29], tau90)),
    "cap 14–21, τ 75th": (hist, None, ("cap", list(range(14, 22)), tau)),
    **{f"push +{f} at block {BLOCK}": (hist, None, ("push", BLOCK, f)) for f in (0.1, 0.2, 0.3)},
    "system prompt": (hist, "You are a helpful assistant.", None),
}

def bench(name, h, system, iv):
    prompts = [pv.chat_prompt(tok, system, user, h)] * N_BENCH
    if iv is None:
        gens = pv.sample(model, tok, prompts, **BENCH_GEN)
    elif iv[0] == "cap":
        with pv.cap(model, iv[1], iv[3] if len(iv) > 3 else axis, iv[2]) as stats:   # iv[3]: another direction to cap along (controls)
            gens = pv.sample(model, tok, prompts, **BENCH_GEN)
        log.info("%s: cap fired on %s (mean lift per capped position in brackets)", name,
                 " · ".join(f"{100 * s['capped'] / max(s['seen'], 1):.0f}% ({s.get('lift', 0) / max(s['capped'], 1):.0f})" for s in stats.values()))
    else:   # h ← h + f · |h| · â at every position: a push toward the Assistant, not a floor
        with pv.steer(model, iv[1], a_unit[iv[1]], iv[2] * h_len, positions="all"):
            gens = pv.sample(model, tok, prompts, **BENCH_GEN)
    ctx = "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in h[-2:]) + f"\n\nUser: {user}"
    return pd.DataFrame(gens).assign(cond=name, sample=range(N_BENCH), question=ctx)

frames = []
for name, (h, system, iv) in CONDS.items():
    k = pv.key(name, len(h), system or "", iv[1] if iv else 0, iv[2] if iv and iv[0] == "cap" else (iv[2] if iv else 0), BENCH_GEN, BENCH_TURN, axis[BLOCK])
    frames.append(pv.cached_df(f"axis_bench_{k}", lambda: bench(name, h, system, iv)))
bench_df = pd.concat(frames, ignore_index=True)
bench_df = judge.score_df(bench_df, JUDGES["crisis_care"]).rename(columns={"trait_score": "crisis_care"})
bench_df["harmful_drift"] = judge.score_df(bench_df, JUDGES["harmful_drift"], coherence=False)["trait_score"]
s = bench_df.groupby("cond", sort=False)[["crisis_care", "harmful_drift", "coherence"]].mean()
log.info("one-turn bench, mean of %d replies (recognises risk / harmful drift / coherence): %s · judge $%.2f", N_BENCH,
         " · ".join(f"{c} {r.crisis_care:.0f}/{r.harmful_drift:.0f}/{r.coherence:.0f}" for c, r in s.iterrows()), judge.cost())

# %% Cap: bench plot
g = bench_df.groupby("cond", sort=False)
mean, sem = g[["crisis_care", "harmful_drift"]].mean(), g[["crisis_care", "harmful_drift"]].sem()
SHORT = {"no history": "no\nhistory", "free": "free", "cap 26–29, τ 75th": "cap\n26–29", "cap 22–29, τ 75th": "cap\n22–29",
         "cap 26–29, τ 90th": "cap 26–29\nτ 90th", "cap 14–21, τ 75th": "cap\n14–21", "push +0.1 at block 18": "push\n+0.1",
         "push +0.2 at block 18": "push\n+0.2", "push +0.3 at block 18": "push\n+0.3", "system prompt": "system\nprompt"}
fig, axs = savoy.figure(2, 1, height=200, titles=["recognises the risk, points to help (higher is better)", "harmful drift (lower is better)"],
                        ytitle=f"judge score, mean ± 1 s.e. over {N_BENCH} replies", xtitle="condition (grey: no intervention)")
for ax, k in zip(axs, ("crisis_care", "harmful_drift")):
    savoy.bars(ax, [SHORT[c] for c in mean.index], mean[k].tolist(), yerr=sem[k].fillna(0).tolist(),
               baseline=[list(mean.index).index(c) for c in ("no history", "free")])
    savoy.yticks(ax, 0, 100, n=5)
savoy.save(fig, FIG / "axis_bench.png")
plt.show()
best = mean.crisis_care.drop(["no history"]).idxmax()
log.info("best rescue with the drifted history: %s (recognises risk %.0f, vs free %.0f and no history %.0f)",
         best, mean.crisis_care[best], mean.crisis_care["free"], mean.crisis_care["no history"])

# %% Cap: where in the model?
tau50 = np.percentile(proj_all.numpy(), 50, axis=0)
BANDS = {   # name: (blocks, τ); same 8-block width as the paper, slid through the model, then split and re-thresholded at 14–21
    "free": (None, None),
    "cap 6–13, τ 75th": (list(range(6, 14)), tau),
    "cap 10–17, τ 75th": (list(range(10, 18)), tau),
    "cap 14–21, τ 75th": (list(range(14, 22)), tau),
    "cap 18–25, τ 75th": (list(range(18, 26)), tau),
    "cap 22–29, τ 75th": (list(range(22, 30)), tau),
    "cap 14–17, τ 75th": (list(range(14, 18)), tau),
    "cap 18–21, τ 75th": (list(range(18, 22)), tau),
    "cap 14–21, τ 50th": (list(range(14, 22)), tau50),
    "cap 14–21, τ 90th": (list(range(14, 22)), tau90),
    "cap 14–21, τ 95th": (list(range(14, 22)), np.percentile(proj_all.numpy(), 95, axis=0)),   # Peter: push harder
    "cap 14–21, τ 99th": (list(range(14, 22)), np.percentile(proj_all.numpy(), 99, axis=0)),
}
frames = []
for name, (blocks, t) in BANDS.items():
    iv = None if blocks is None else ("cap", blocks, t)
    k = pv.key(name, len(hist), "", iv[1] if iv else 0, iv[2] if iv else 0, BENCH_GEN, BENCH_TURN, axis[BLOCK])
    frames.append(pv.cached_df(f"axis_bench_{k}", lambda: bench(name, hist, None, iv)))
bands_df = pd.concat(frames, ignore_index=True)
bands_df = judge.score_df(bands_df, JUDGES["crisis_care"]).rename(columns={"trait_score": "crisis_care"})
bands_df["harmful_drift"] = judge.score_df(bands_df, JUDGES["harmful_drift"], coherence=False)["trait_score"]
g = bands_df.groupby("cond", sort=False)
mean, sem = g[["crisis_care", "harmful_drift", "coherence"]].mean(), g[["crisis_care", "harmful_drift"]].sem()
starts = [6, 10, 14, 18, 22]
by_band = mean.loc[[f"cap {b}–{b + 7}, τ 75th" for b in starts]]
TAU_QS = (50, 75, 90, 95, 99)
by_tau = mean.loc[[f"cap 14–21, τ {q}th" for q in TAU_QS]]
fig, axs = savoy.figure(1, 2, height=240, ytitle="judge score, mean of 10 replies", xtitle=["first block of the 8 capped (τ 75th)", "τ percentile (blocks 14–21)"],
                        titles=["which blocks?", "how hard?"])
for ax, x, d in ((axs[0], starts, by_band), (axs[1], list(TAU_QS), by_tau)):
    for i, k in enumerate(("crisis_care", "harmful_drift")):
        savoy.line(ax, x, d[k].tolist(), savoy.color(i), points=True)
    savoy.ref(ax, float(mean.loc["free", "harmful_drift"]), "harmful drift without a cap", threshold=True)
    savoy.yticks(ax, 0, 100, n=5)
    savoy.xticks(ax, x[0], x[-1], ticks=x)
savoy.key(fig, "recognises the risk (higher is better)", savoy.color(0))
savoy.key(fig, "harmful drift (lower is better)", savoy.color(1))
savoy.save(fig, FIG / "axis_bench_bands.png")
plt.show()
log.info("where in the model (recognises risk / harmful drift / coherence): %s · judge $%.2f",
         " · ".join(f"{c} {r.crisis_care:.0f}/{r.harmful_drift:.0f}/{r.coherence:.0f}" for c, r in mean.iterrows()), judge.cost())

# %% Cap: is it the axis?
MASSIVE = [0, 4, 7, 100, 396]   # Exp 1's spike-token dims; 4 and 100 are already zero in the axis
CTRL_BLOCKS, CTRL_Q = list(range(14, 22)), 95

def direction_cap(name, d):
    """Cap along direction d (36, d) at CTRL_BLOCKS, τ = d's own CTRL_Q-th percentile over step 1's answers."""
    t = np.percentile(torch.einsum("nbd,bd->nb", acts, F.normalize(d, dim=1)).numpy(), CTRL_Q, axis=0)
    return name, ("cap", CTRL_BLOCKS, t, d)

g0 = torch.Generator().manual_seed(0)
rand = [F.normalize(torch.randn(axis.shape, generator=g0), dim=1) for _ in range(2)]
for r in rand:
    r[:, SPIKE_DIMS] = 0
axis_nm = axis.clone()
axis_nm[:, MASSIVE] = 0
CTRL = dict([("free", None), ("cap 14–21, τ 95th", ("cap", CTRL_BLOCKS, np.percentile(proj_all.numpy(), 95, axis=0))),
             direction_cap("axis, massive dims zeroed", axis_nm),
             direction_cap("random direction 1", rand[0]), direction_cap("random direction 2", rand[1])])
# The random floors above push ~3× less than the axis's (0.7 vs 2.5 per capped token on the turn-29 context).
# Matched control: per block, raise random 1's τ until its mean lift over the (uncapped) context equals the axis's.
with torch.no_grad():
    H = model(**tok(pv.chat_prompt(tok, None, user, hist), return_tensors="pt").to(model.device),
              output_hidden_states=True).hidden_states[1:]
def lift(proj, t):
    return float(np.clip(t - proj, 0, None).mean())
t_ax, t_r = CTRL["cap 14–21, τ 95th"][2], CTRL["random direction 1"][2].copy()
for b_ in CTRL_BLOCKS:
    pa = (H[b_][0].float() @ F.normalize(axis[b_], dim=0).to(H[b_].device)).cpu().numpy()
    pr = (H[b_][0].float() @ rand[0][b_].to(H[b_].device)).cpu().numpy()
    grid = np.linspace(pr.min(), pr.max() + 3 * lift(pa, t_ax[b_]), 400)
    t_r[b_] = grid[np.argmin([abs(lift(pr, g) - lift(pa, t_ax[b_])) for g in grid])]
del H
CTRL["random 1, push matched"] = ("cap", CTRL_BLOCKS, t_r, rand[0])
SHORT_C_EXTRA = {"random 1, push matched": "random 1,\npush matched"}
frames = []
for name, iv in CTRL.items():
    k = pv.key(name, len(hist), "", iv[1] if iv else 0, iv[2] if iv else 0, BENCH_GEN, BENCH_TURN, axis[BLOCK])
    frames.append(pv.cached_df(f"axis_bench_{k}", lambda: bench(name, hist, None, iv)))
ctrl_df = pd.concat(frames, ignore_index=True)
ctrl_df = judge.score_df(ctrl_df, JUDGES["crisis_care"]).rename(columns={"trait_score": "crisis_care"})
ctrl_df["harmful_drift"] = judge.score_df(ctrl_df, JUDGES["harmful_drift"], coherence=False)["trait_score"]
g = ctrl_df.groupby("cond", sort=False)
mean, sem = g[["crisis_care", "harmful_drift", "coherence"]].mean(), g[["crisis_care", "harmful_drift"]].sem()
SHORT_C = {"free": "free", "cap 14–21, τ 95th": "axis", "axis, massive dims zeroed": "axis without\nmassive dims",
           "random direction 1": "random\ndirection 1", "random direction 2": "random\ndirection 2", **SHORT_C_EXTRA}
fig, axs = savoy.figure(2, 1, height=200, titles=["recognises the risk, points to help (higher is better)", "harmful drift (lower is better)"],
                        ytitle=f"judge score, mean ± 1 s.e. over {N_BENCH} replies", xtitle=f"cap along… (blocks 14–21, τ {CTRL_Q}th; grey: no cap)")
for ax, k in zip(axs, ("crisis_care", "harmful_drift")):
    savoy.bars(ax, [SHORT_C[c] for c in mean.index], mean[k].tolist(), yerr=sem[k].fillna(0).tolist(), baseline=[0])
    savoy.yticks(ax, 0, 100, n=5)
savoy.save(fig, FIG / "axis_bench_controls.png")
plt.show()
log.info("is it the axis? (recognises risk / harmful drift / coherence): %s · judge $%.2f",
         " · ".join(f"{c} {r.crisis_care:.0f}/{r.harmful_drift:.0f}/{r.coherence:.0f}" for c, r in mean.iterrows()), judge.cost())
