"""
Shared code for the persona-vectors experiments.

Layout:
  1. Paths, model loading, reuse of linear-probes/lib.py
  2. Trait artifacts (data/traits/*.json) + prompt construction
  3. Sampled generation (rollouts)
  4. Activation collection (every layer, per-example prompt/response split)
  5. Persona-vector extraction        <- TODO_PETER (the gears)
  6. Steering sweep + monitoring harness
  7. Cache I/O

Layer convention (same as linear-probes): `layer L` = output of transformer
block L, 0-indexed = hidden_states[L + 1]. Qwen3-4B has 36 blocks, so L ∈ [0, 35].
The paper counts from 1 and means the output of the ℓ-th block, so the paper's
"layer 20" is our L = 19. A vector extracted at block L is steered at block L:
`probes.steering(model, layer=L, ...)` hooks that same block's output.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 1. Paths / model / linear-probes reuse
# ---------------------------------------------------------------------------

TOPIC_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOPIC_DIR.parents[1]
DATA_DIR = TOPIC_DIR / "data"
TRAITS_DIR = DATA_DIR / "traits"
CACHE_DIR = TOPIC_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

LOCAL_MODEL = REPO_ROOT / "models" / "Qwen3-4B"   # same weights linear-probes used
HUB_MODEL = "Qwen/Qwen3-4B"

load_dotenv(REPO_ROOT / ".env")  # HF_TOKEN, ANTHROPIC_API_KEY


def _load_probes_lib():
    """linear-probes/lib.py, imported as `probes`. Both topics would otherwise fight
    over the module name `lib`. Reused here: your `steering` hook."""
    path = REPO_ROOT / "topics" / "linear-probes" / "lib.py"
    spec = importlib.util.spec_from_file_location("probes_lib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probes = _load_probes_lib()


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str | None = None, device: str | None = None, dtype=torch.bfloat16):
    """Qwen3-4B by default: the local copy under models/ if present, else the HF hub.
    Pass e.g. "Qwen/Qwen3-0.6B" for a mechanics-only pass on the Mac.

    Returns (model, tokenizer, device). Left padding, as in linear-probes, so the
    last real token of every sequence in a batch sits at position -1.
    """
    device = device or default_device()
    src = model_id or (str(LOCAL_MODEL) if LOCAL_MODEL.exists() else HUB_MODEL)
    tok = AutoTokenizer.from_pretrained(src)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, device


def n_blocks(model) -> int:
    return model.config.num_hidden_layers


# ---------------------------------------------------------------------------
# 2. Trait artifacts + prompts
# ---------------------------------------------------------------------------

def load_trait(name: str) -> dict:
    """{trait, description, instruction: [{pos, neg}] x5, questions_extract (20),
    questions_eval (20), eval_prompt}. Paper App. A.1 format."""
    return json.loads((TRAITS_DIR / f"{name}.json").read_text())


def load_ladder(name: str) -> list[str]:
    """8 system prompts, index 0 = least trait-encouraging, 7 = most (paper §3.3)."""
    return json.loads((DATA_DIR / "monitor_ladders.json").read_text())[name]


def has_ladder(name: str) -> bool:
    """Monitoring runs only for traits with a ladder: dishonest. pirate and mystic
    skip it (scope trim)."""
    return name in json.loads((DATA_DIR / "monitor_ladders.json").read_text())


SYSTEM_PREFIX = "You are a helpful assistant."


def system_prompt(trait: dict, wording: int, condition: str) -> str:
    """System prompt for one wording: SYSTEM_PREFIX + that wording's POS (toward the
    trait) or NEG (away from it) instruction. The trait JSON stores each wording as
    {"pos": ..., "neg": ...}, the paper's format.

    The prefix is the same for POS and NEG, so they differ only in the trait
    instruction. (The paper's evil ladder suggests "You are an evil assistant." on the
    trait side; a prefix on one side only would leak into the vector, e.g. pirate +
    "less helpful".)
    """
    return f"{SYSTEM_PREFIX} {trait['instruction'][wording][condition.lower()]}"


def chat_prompt(tok, system: str | None, user: str) -> str:
    """Qwen3 chat prompt ending at the assistant header, thinking disabled (the
    template appends the empty <think></think> block), so the model answers
    directly. Same thinking-off setting as linear-probes Exp 3 v2 onward."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def rollout_table(
    trait: dict,
    split: str = "extract",
    n_answers: int = 10,
    n_questions: int | None = None,
    n_wordings: int | None = None,
    baseline: bool = True,
) -> pd.DataFrame:
    """One row per answer to generate: question × condition × answer.

    Conditions: POS (system prompt pushing toward the trait), NEG (pushing away from
    it), BASE (no system prompt). Every question gets `n_answers` answers per
    condition. POS and NEG cycle through the wordings (the trait file's 5 POS/NEG
    pairs), so 10 answers over 5 wordings is 2 per wording. The paper says "10
    rollouts each" per question and polarity, without saying how they spread over
    the pairs.

    BASE is ours, not the paper's: the subtracted side of the POS − BASE vector,
    built next to the paper's POS − NEG (see `contrast`). `n_questions` /
    `n_wordings` truncate for smoke runs.
    """
    questions = trait[f"questions_{split}"][:n_questions]
    n_w = len(trait["instruction"][:n_wordings])
    conditions = ("POS", "NEG", "BASE") if baseline else ("POS", "NEG")
    rows = []
    for qid, q in enumerate(questions):
        for cond in conditions:
            for s in range(n_answers):
                w = -1 if cond == "BASE" else s % n_w
                rows.append(dict(
                    trait=trait["trait"], split=split, qid=qid, question=q,
                    condition=cond, wording=w, sample=s,
                    system="" if cond == "BASE" else system_prompt(trait, w, cond),
                ))
    return pd.DataFrame(rows)


def contrast(df: pd.DataFrame, negative: str = "NEG") -> pd.DataFrame:
    """The rows one persona vector is built from: the POS rows plus one subtracted
    side, relabelled "NEG" so `persona_vectors` always sees POS vs NEG. The original
    index is kept, since it picks the matching activations.

      negative="NEG"   POS − NEG   (the paper's vector)
      negative="BASE"  POS − BASE  (no system prompt on the subtracted side)
    """
    out = df[df["condition"].isin(["POS", negative])].copy()
    out["condition"] = out["condition"].replace({negative: "NEG"})
    return out


# ---------------------------------------------------------------------------
# 3. Sampled generation
# ---------------------------------------------------------------------------

def _eos_ids(model, tok) -> set[int]:
    eos = model.generation_config.eos_token_id
    ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    ids.add(tok.eos_token_id)
    ids.add(tok.pad_token_id)
    return {i for i in ids if i is not None}


@torch.no_grad()
def sample(
    model,
    tok,
    prompts: list[str],
    max_new_tokens: int = 384,
    temperature: float = 1.0,
    batch_size: int = 32,
    seed: int = 0,
    device: str | None = None,
) -> list[dict]:
    """Sampled continuations, one per prompt (repeat a prompt to get several).

    Plain temperature sampling (top_p=1, top_k off), overriding whatever sampling
    defaults Qwen3's generation_config ships. Prompts are length-sorted so batches
    pad less; the output order matches the input. Returns
    [{response, n_tokens, truncated}], where truncated = hit max_new_tokens before EOS.
    """
    device = device or model.device
    torch.manual_seed(seed)
    eos = _eos_ids(model, tok)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[dict | None] = [None] * len(prompts)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        enc = tok([prompts[i] for i in idx], padding=True, return_tensors="pt").to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            pad_token_id=tok.pad_token_id,
        )
        new = gen[:, enc["input_ids"].shape[1] :].tolist()
        for j, i in enumerate(idx):
            ids = new[j]
            n = next((k for k, t in enumerate(ids) if t in eos), len(ids))
            out[i] = dict(
                response=tok.decode(ids[:n], skip_special_tokens=True).strip(),
                n_tokens=n,
                truncated=n == len(ids),
            )
    return out


def generate_rollouts(model, tok, df: pd.DataFrame, **sample_kw) -> pd.DataFrame:
    """Fill `prompt`, `response`, `n_tokens`, `truncated` for a rollout_table()."""
    df = df.copy()
    df["prompt"] = [chat_prompt(tok, s, q) for s, q in zip(df["system"], df["question"])]
    gens = sample(model, tok, df["prompt"].tolist(), **sample_kw)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(gens)], axis=1)


# ---------------------------------------------------------------------------
# 4. Activation collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _per_example_hidden(model, tok, prompts, responses, batch_size, device, layers=None):
    """Yield (h, n_prompt) per example, in input order. One forward pass per batch over
    prompt + response.

      h        : (n_layers, seq, d) for THIS example only, padding stripped. All blocks
                 by default (h[L] = output of block L, embeddings dropped), or just
                 `layers`, in that order.
      n_prompt : positions [0, n_prompt) are the chat prompt (system, user, assistant
                 header, empty think block); [n_prompt, seq) is the response.

    Prompt and response are tokenized separately and concatenated, so the boundary is
    exact (no BPE merge across it). Position ids are passed explicitly so left padding
    doesn't shift RoPE positions.
    """
    pad = tok.pad_token_id
    for b in range(0, len(prompts), batch_size):
        p_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts[b : b + batch_size]]
        r_ids = [tok(r, add_special_tokens=False).input_ids for r in responses[b : b + batch_size]]
        seqs = [p + r for p, r in zip(p_ids, r_ids)]
        T = max(map(len, seqs))
        input_ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), T), dtype=torch.long)
        for j, seq in enumerate(seqs):
            input_ids[j, T - len(seq) :] = torch.tensor(seq)
            attn[j, T - len(seq) :] = 1
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        hs = model(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            position_ids=pos.to(device),
            output_hidden_states=True,
        ).hidden_states[1:]  # drop embeddings: hs[L] = block L output, (B, T, d)
        if layers is not None:
            hs = [hs[L] for L in layers]
        for j, seq in enumerate(seqs):
            yield torch.stack([h[j, T - len(seq) :] for h in hs]), len(p_ids[j])


def collect_reduced(
    model,
    tok,
    prompts: list[str],
    responses: list[str],
    reduce_fn,
    batch_size: int = 8,
    device: str | None = None,
) -> torch.Tensor:
    """Every layer's residual stream over each prompt + response, then
    `reduce_fn(h, n_prompt)` collapses the token axis (h, n_prompt as in
    `_per_example_hidden`). Returns (N, n_blocks, d) float32 on CPU."""
    device = device or model.device
    return torch.stack([
        reduce_fn(h, n_prompt).float().cpu()
        for h, n_prompt in _per_example_hidden(model, tok, prompts, responses, batch_size, device)
    ])


def prompt_last_token_acts(model, tok, prompts: list[str], layer: int, batch_size: int = 16,
                           device: str | None = None) -> torch.Tensor:
    """Block-`layer` residual at the last prompt token (the one right before the
    response), (N, d) float32 on CPU."""
    device = device or model.device
    return torch.stack([
        h[0, n_prompt - 1].float().cpu()
        for h, n_prompt in _per_example_hidden(model, tok, prompts, [""] * len(prompts),
                                               batch_size, device, layers=[layer])
    ])


def token_projections(model, tok, prompts: list[str], responses: list[str], layer: int, vector,
                      batch_size: int = 8, device: str | None = None) -> list[torch.Tensor]:
    """MONITOR read during the answer: for each example, the block-`layer` residual
    projected onto v̂ at every position from the last prompt token on. Index 0 is the
    last prompt token (the same reading as `prompt_last_token_acts`, i.e. before the
    answer), index t is the t-th answer token. Returns 1-D float32 CPU tensors, one
    per example, of length 1 + n_answer_tokens."""
    device = device or model.device
    v = torch.as_tensor(vector, dtype=torch.float32)
    v = (v / v.norm()).to(device)
    return [
        (h[0, n_prompt - 1 :].float() @ v).cpu()
        for h, n_prompt in _per_example_hidden(model, tok, prompts, responses, batch_size,
                                               device, layers=[layer])
    ]


def project(acts, v) -> np.ndarray:
    """Scalar projection of each row of `acts` onto the unit vector along `v`."""
    acts = torch.as_tensor(acts, dtype=torch.float32)
    v = torch.as_tensor(v, dtype=torch.float32)
    return (acts @ (v / v.norm())).numpy()


# ---------------------------------------------------------------------------
# 5. Persona-vector extraction  [GEAR — Peter]
# ---------------------------------------------------------------------------

def filter_rollouts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the rollouts that should feed the persona vector."""
    # TODO_PETER: df has one row per answer: condition ∈ {POS, NEG, BASE} (BASE = no system
    #   prompt, the subtracted side of POS − BASE), trait_score and coherence ∈ [0, 100] or NaN,
    #   refusal, truncated. Return the subset you trust (§2.2), BASE rows included.
    raise NotImplementedError


def reduce_tokens(h: torch.Tensor, n_prompt: int) -> torch.Tensor:
    """Collapse one example's residual stream over tokens: (n_blocks, seq, d) -> (n_blocks, d)."""
    # TODO_PETER: rows [0, n_prompt) are the prompt, [n_prompt, seq) the response. Which positions
    #   carry the persona? App. A.3 compares three choices: pick one, or parametrize and compare.
    raise NotImplementedError


def persona_vectors(acts: torch.Tensor, df: pd.DataFrame) -> torch.Tensor:
    """One candidate persona vector per layer from the reduced activations."""
    # TODO_PETER: acts (N, n_blocks, d), row-aligned with df, whose condition is POS or NEG
    #   (`contrast` relabels BASE as NEG for POS − BASE) -> (n_blocks, d).
    #   Decide the normalization: it sets the scale of every steering coefficient downstream.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 6. Steering sweep + monitoring harness
# ---------------------------------------------------------------------------

def steering_sweep(
    model,
    tok,
    questions: list[str],
    vectors: torch.Tensor,
    layers: list[int],
    coeffs: list[float],
    n_samples: int = 2,
    system: str | None = None,
    include_baseline: bool = True,
    **sample_kw,
) -> pd.DataFrame:
    """Generate on `questions` while adding coeff · vectors[L] to block L's output,
    for every (L, coeff). The hook fires on every forward pass: prompt prefill and
    each decoding step (paper §3.2, h_ℓ ← h_ℓ + α·v_ℓ). Plus one unsteered baseline
    (layer = -1, coeff = 0). No system prompt by default: we steer the plain Assistant.

    Returns one row per generation: layer, coeff, qid, question, sample, response, ...
    Judge it afterwards.
    """
    prompts = [chat_prompt(tok, system, q) for q in questions for _ in range(n_samples)]
    meta = [dict(qid=qi, question=q, sample=s)
            for qi, q in enumerate(questions) for s in range(n_samples)]
    frames = []

    def run(layer, coeff, ctx):
        with ctx:
            gens = sample(model, tok, prompts, **sample_kw)
        frames.append(pd.DataFrame([dict(layer=layer, coeff=coeff, **m, **g)
                                    for m, g in zip(meta, gens)]))

    if include_baseline:
        run(-1, 0.0, contextlib.nullcontext())
    device = sample_kw.get("device") or model.device
    for L in layers:
        v = torch.as_tensor(vectors[L], dtype=torch.float32).to(device)
        for c in coeffs:
            run(L, float(c), probes.steering(model, L, v, c, device=device))
    return pd.concat(frames, ignore_index=True)


def monitoring_run(
    model,
    tok,
    ladder: list[str],
    questions: list[str],
    layer: int,
    vector,
    n_samples: int = 4,
    **sample_kw,
) -> pd.DataFrame:
    """Paper §3.3. For each system prompt on the ladder × question: project the
    block-`layer` activation at the LAST prompt token onto v̂ (measured before any
    response exists), then sample `n_samples` responses to judge.

    Returns one row per rollout: level, system, qid, question, projection, sample, response, ...
    """
    cells = [dict(level=lv, system=sp, qid=qi, question=q)
             for lv, sp in enumerate(ladder) for qi, q in enumerate(questions)]
    prompts = [chat_prompt(tok, c["system"], c["question"]) for c in cells]
    proj = project(prompt_last_token_acts(model, tok, prompts, layer), vector)
    gens = sample(model, tok, [p for p in prompts for _ in range(n_samples)], **sample_kw)
    rows = []
    for ci, c in enumerate(cells):
        for s in range(n_samples):
            rows.append(dict(**c, projection=float(proj[ci]), sample=s, **gens[ci * n_samples + s]))
    return pd.DataFrame(rows)


def monitor_correlations(scored: pd.DataFrame, proj_col: str = "projection") -> tuple[pd.DataFrame, dict]:
    """App. C.2. Average the judged rollouts to one point per (level, question), then
    Pearson r between the mean of `proj_col` and the mean trait score (the prompt
    projection is the same for every sample of a cell; a during-the-answer reading
    isn't, hence the mean):
      overall       : across all points
      within-level  : per level, then averaged; levels with score std < 1 skipped
    The gap between the two says how much of the signal is just "which system prompt".
    """
    agg = (scored.groupby(["level", "qid"], as_index=False)
           .agg(projection=(proj_col, "mean"), trait_score=("trait_score", "mean")))
    overall = agg["projection"].corr(agg["trait_score"])
    within = [g["projection"].corr(g["trait_score"])
              for _, g in agg.groupby("level") if g["trait_score"].std() >= 1]
    within = [r for r in within if not np.isnan(r)]
    return agg, dict(overall_r=overall,
                     within_level_r=float(np.mean(within)) if within else np.nan,
                     n_levels_within=len(within))


# ---------------------------------------------------------------------------
# 7. Cache I/O
# ---------------------------------------------------------------------------

def key(*parts) -> str:
    """Short content hash for cache names, so a changed vector, layer list or
    coefficient list never gets served a stale cached generation run."""
    h = hashlib.md5()
    for p in parts:
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().float().numpy()
        h.update(p.tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
    return h.hexdigest()[:8]


def save_df(df: pd.DataFrame, name: str) -> Path:
    p = CACHE_DIR / f"{name}.jsonl"
    df.to_json(p, orient="records", lines=True)
    return p


def load_df(name: str) -> pd.DataFrame:
    return pd.read_json(CACHE_DIR / f"{name}.jsonl", orient="records", lines=True)


def cached_df(name: str, fn) -> pd.DataFrame:
    """Load cache/<name>.jsonl if present, else run fn() and save it."""
    if (CACHE_DIR / f"{name}.jsonl").exists():
        print(f"cache hit: {name}")
        return load_df(name)
    df = fn()
    save_df(df, name)
    return df


def save_tensor(t: torch.Tensor, name: str) -> Path:
    p = CACHE_DIR / f"{name}.pt"
    torch.save(t, p)
    return p


def load_tensor(name: str) -> torch.Tensor:
    return torch.load(CACHE_DIR / f"{name}.pt", map_location="cpu")


def save_persona(trait: str, vectors: torch.Tensor, layer: int, **meta) -> Path:
    """The Exp 1 deliverable, read by Exp 2 and by emergent-misalignment:
    every layer's candidate plus the chosen layer."""
    p = CACHE_DIR / f"persona_{trait}.pt"
    torch.save(dict(trait=trait, vectors=vectors.cpu(), layer=layer, **meta), p)
    return p


def load_persona(trait: str) -> dict:
    return torch.load(CACHE_DIR / f"persona_{trait}.pt", map_location="cpu", weights_only=False)
