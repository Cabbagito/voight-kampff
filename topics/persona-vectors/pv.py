"""
Shared code for the persona-vectors experiments.

Layout:
  1. Paths, model loading
  2. Trait artifacts (data/traits/*.json) + prompt construction
  3. Sampled generation (rollouts)
  4. Activation collection (every layer, per-example prompt/response split)
  5. Persona-vector extraction        <- the method choices: FILTER, TOKENS, SCALE
  6. Steering sweep + monitoring harness
  7. Cache I/O

Layer convention (same as linear-probes): `layer L` = output of transformer
block L, 0-indexed = hidden_states[L + 1]. Qwen3-4B has 36 blocks, so L ∈ [0, 35].
The paper counts from 1 and means the output of the ℓ-th block, so the paper's
"layer 20" is our L = 19. A vector extracted at block L is steered at block L:
`steer(model, layer=L, ...)` hooks that same block's output.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm   # plain text bars (not tqdm.auto widgets): the friday dashboard shows them live
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache, StoppingCriteria, StoppingCriteriaList
from transformers.generation.logits_process import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
                                                    TemperatureLogitsWarper, TopKLogitsWarper)

# ---------------------------------------------------------------------------
# 1. Paths / model
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
    if os.environ.get("ESPER_SOCKET") and device == "cuda" and dtype == torch.bfloat16:
        import esper   # inside an esper session: share the server's copy of the weights (~0.1 s)
        model = esper.model(src)
    else:
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


def chat_prompt(tok, system: str | None, user: str, history=None) -> str:
    """Qwen3 chat prompt ending at the assistant header, thinking disabled (the
    template appends the empty <think></think> block), so the model answers
    directly. Same thinking-off setting as linear-probes Exp 3 v2 onward.

    `history`: earlier (user, assistant) turns placed between the system prompt and
    the real question (see `fewshot_turns`). They count as prompt, so the TOKENS
    average over the answer never includes them."""
    msgs = [{"role": "system", "content": system}] if system else []
    for u, a in history or []:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def fewshot_turns(trait: dict, condition: str) -> list[tuple[str, str]]:
    """Example turns shown before the real question, from the trait file's optional
    `fewshot` list ({question, pos, neg}). POS gets the in-trait answers (dishonest:
    confident lies), NEG and BASE the out-of-trait ones (truthful), so POS and NEG
    differ only in whether the history and the instruction carry the trait.
    [] for traits without a `fewshot` list."""
    side = "pos" if condition == "POS" else "neg"
    return [(ex["question"], ex[side]) for ex in trait.get("fewshot", [])]


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

    `history` holds the trait file's example turns, if it has any (`fewshot_turns`;
    dishonest does).

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
                    history=fewshot_turns(trait, cond),
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
def _token_bar(n_prompts: int, batch_size: int, max_new_tokens: int):
    """A bar that moves per decoding step, not per finished batch: a batch of 40 answers grows
    token by token together, so counting answers made it jump 0 → 40 (Peter, 2026-10-29).
    Total = every batch's token budget; a batch that stops early skips ahead (`_end_batch`)."""
    n_batches = -(-n_prompts // batch_size)
    return tqdm(total=n_batches * max_new_tokens, desc=f"generating {n_prompts} answers", unit="tok")


def _end_batch(bar, steps: int, max_new_tokens: int):
    bar.update(max_new_tokens - steps)


class _TickBar(StoppingCriteria):
    """Never stops generation; ticks the bar once per decoding step."""
    def __init__(self, bar):
        self.bar, self.steps = bar, 0

    def __call__(self, input_ids, scores, **kwargs):
        self.bar.update(1)
        self.steps += 1
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def sample(
    model,
    tok,
    prompts: list[str],
    max_new_tokens: int = 384,
    temperature: float = 1.0,
    batch_size: int = 32,
    seed: int = 0,
    device: str | None = None,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> list[dict]:
    """Sampled continuations, one per prompt (repeat a prompt to get several).

    Plain temperature sampling by default (top_p=1, top_k off, no repetition penalty),
    overriding whatever sampling defaults Qwen3's generation_config ships. Prompts are length-sorted so batches
    pad less; the output order matches the input. Returns
    [{response, n_tokens, truncated}], where truncated = hit max_new_tokens before EOS.
    """
    device = device or model.device
    torch.manual_seed(seed)
    eos = _eos_ids(model, tok)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[dict | None] = [None] * len(prompts)
    bar = _token_bar(len(prompts), batch_size, max_new_tokens)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        enc = tok([prompts[i] for i in idx], padding=True, return_tensors="pt").to(device)
        tick = _TickBar(bar)
        gen = model.generate(
            stopping_criteria=StoppingCriteriaList([tick]),
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
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
        _end_batch(bar, tick.steps, max_new_tokens)
    bar.close()
    return out


@torch.no_grad()
def sample_stepwise(
    model,
    tok,
    prompts: list[str],
    layer: int,
    vector,
    coeff: float,
    cache: str = "clean",
    max_new_tokens: int = 384,
    temperature: float = 1.0,
    batch_size: int = 32,
    seed: int = 0,
    device: str | None = None,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
) -> list[dict]:
    """Like `sample`, with coeff · vector added at block `layer` to the newest position of every
    pass, decoded one token at a time so the cache can be controlled.

    cache="clean" (C): the push only shapes the pass that picks the next token. That position is
      then rerun without the push and stored, so later tokens see an unpushed history. Two passes
      per token.
    cache="steered": the pushed values are stored. That is B, the same as `sample` inside
      `steer(..., "response")`; kept to check this loop against `generate`.
    """
    device = device or model.device
    torch.manual_seed(seed)
    eos = torch.tensor(sorted(_eos_ids(model, tok)), device=device)
    procs = LogitsProcessorList([p for p in (
        RepetitionPenaltyLogitsProcessor(repetition_penalty) if repetition_penalty != 1.0 else None,
        TemperatureLogitsWarper(temperature) if temperature != 1.0 else None,
        TopKLogitsWarper(top_k) if top_k else None) if p is not None])
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[dict | None] = [None] * len(prompts)
    bar = _token_bar(len(prompts), batch_size, max_new_tokens)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        enc = tok([prompts[i] for i in idx], padding=True, return_tensors="pt").to(device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        kv = DynamicCache(config=model.config)
        model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], position_ids=pos[:, :-1],
              past_key_values=kv, use_cache=True)   # the prompt minus its last token, unpushed
        cur, cur_pos, seqs = ids[:, -1:], pos[:, -1:], ids
        done = torch.zeros(len(idx), dtype=torch.bool, device=device)
        new = []
        for _ in range(max_new_tokens):
            step = dict(input_ids=cur, attention_mask=mask, position_ids=cur_pos, past_key_values=kv, use_cache=True)
            with steer(model, layer, vector, coeff, "response"):
                logits = model(**step).logits[:, -1].float()
            if cache == "clean":
                kv.crop(kv.get_seq_length() - 1)   # drop the pushed entry, store the unpushed one
                model(**step)
            nxt = logits.argmax(-1, keepdim=True) if greedy else torch.multinomial(torch.softmax(procs(seqs, logits), -1), 1)
            nxt[done] = tok.pad_token_id
            new.append(nxt)
            bar.update(1)
            seqs = torch.cat([seqs, nxt], 1)
            done |= torch.isin(nxt[:, 0], eos)
            if done.all():
                break
            cur, cur_pos, mask = nxt, cur_pos + 1, torch.cat([mask, torch.ones_like(nxt)], 1)
        new = torch.cat(new, 1).tolist()
        for j, i in enumerate(idx):
            n = next((k for k, t in enumerate(new[j]) if t in set(eos.tolist())), len(new[j]))
            out[i] = dict(response=tok.decode(new[j][:n], skip_special_tokens=True).strip(),
                          n_tokens=n, truncated=n == max_new_tokens)
        _end_batch(bar, len(new[0]), max_new_tokens)
    bar.close()
    return out


def generate_rollouts(model, tok, df: pd.DataFrame, **sample_kw) -> pd.DataFrame:
    """Fill `prompt`, `response`, `n_tokens`, `truncated` for a rollout_table()."""
    df = df.copy()
    df["prompt"] = [chat_prompt(tok, s, q, h)
                    for s, q, h in zip(df["system"], df["question"], df["history"])]
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
        for h, n_prompt in tqdm(_per_example_hidden(model, tok, prompts, responses, batch_size, device),
                                total=len(prompts), desc="activations", unit="ans")
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


def extremes(x, z: float = 10.0) -> torch.Tensor:
    """The "is this number huge compared to everything else?" check (Peter, 2026-10-29).
    `x` is (n_examples, n_features). True where an entry sits more than `z` robust standard
    deviations (1.4826 × MAD) from its own column's median: one example far off in one
    dimension. Some dimensions are large in every example (block 18: dims 0 and 4 sit ~100×
    the typical value); a whole-tensor cut would flag them everywhere, this doesn't. Use it on
    anything averaged: one such entry can drag a mean."""
    x = torch.as_tensor(x, dtype=torch.float32)
    med = x.median(0).values
    sd = (x - med).abs().median(0).values * 1.4826
    return (x - med).abs() > z * sd.clamp(min=1e-6)


def drop_extremes(df: pd.DataFrame, acts: torch.Tensor, z: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a contrast table (rows index `acts`) into (clean, dropped): an answer is dropped if
    any block's average has an `extremes` entry. Per contrast, so POS − NEG and POS − BASE each
    get the rule applied to their own answers."""
    x = acts[torch.tensor(df.index.to_numpy())]
    out = torch.stack([extremes(x[:, b], z).any(1) for b in range(x.shape[1])]).any(0).numpy()
    return df[~out], df[out]


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
    # df has one row per answer: condition ∈ {POS, NEG, BASE} (BASE = no system
    #   prompt, the subtracted side of POS − BASE), trait_score and coherence ∈ [0, 100] or NaN,
    #   refusal, truncated. Return the subset you trust (§2.2), BASE rows included.


    MIN_COHERENCE = 70

    MIN_POS_TRAIT = 60
    MAX_NEG_TRAIT = 40
    # BASE: no trait cut, it's the default model as it is

    df = df[df['trait_score'].notna() & df['coherence'].notna()]

    df = df[(((df['condition'] == 'POS') & (df['trait_score'] >= MIN_POS_TRAIT)) |
             ((df['condition'] == 'NEG') & (df['trait_score'] <= MAX_NEG_TRAIT)) |
             (df['condition'] == 'BASE')) &
            (df['coherence'] >= MIN_COHERENCE)]

    return df


def reduce_tokens(h: torch.Tensor, n_prompt: int, first_k: int | None = None) -> torch.Tensor:
    """Collapse one example's residual stream over tokens: (n_blocks, seq, d) -> (n_blocks, d)."""
    # rows [0, n_prompt) are the prompt, [n_prompt, seq) the response. Which positions
    #   carry the persona? App. A.3 compares three choices: pick one, or parametrize and compare.
    
    if h.shape[1] == n_prompt:   # empty answer: fall back to the last prompt token, not NaN
        return h[:, n_prompt - 1, :].float()
    end = None if first_k is None else n_prompt + first_k   # None: n_prompt + None is a TypeError
    return h[:, n_prompt:end, :].float().mean(dim=1)   # float first: h is bf16


def persona_vectors(acts: torch.Tensor, df: pd.DataFrame, match_questions: bool = False,
                    normalize: bool = True) -> torch.Tensor:
    """One candidate persona vector per layer from the reduced activations."""
    # acts (N, n_blocks, d), row-aligned with df, whose condition is POS or NEG
    #   (`contrast` relabels BASE as NEG for POS − BASE) -> (n_blocks, d).
    #   Decide the normalization: it sets the scale of every steering coefficient downstream.
    # match_questions: topic cancels inside each question. POS mean − NEG mean per question (only
    #   questions with both), averaged with each question weighted by its number of POS answers.

    if not match_questions:
        pos_acts = acts[df['condition'] == 'POS']
        neg_acts = acts[df['condition'] == 'NEG']
        persona_vecs = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    else:
        diffs, weights = [], []
        for q in df['qid'].unique():
            pos = torch.tensor(((df['condition'] == 'POS') & (df['qid'] == q)).to_numpy())
            neg = torch.tensor(((df['condition'] == 'NEG') & (df['qid'] == q)).to_numpy())
            if pos.any() and neg.any():
                diffs.append(acts[pos].mean(dim=0) - acts[neg].mean(dim=0))   # (n_blocks, d)
                weights.append(pos.sum().float())
        w = torch.stack(weights)
        persona_vecs = (torch.stack(diffs) * w[:, None, None]).sum(dim=0) / w.sum()

    if normalize:   # False: the raw gap, whose length is how far apart POS and NEG sit
        persona_vecs = persona_vecs / persona_vecs.norm(dim=1, keepdim=True)

    return persona_vecs


def residual_norms(model, tok, prompts: list[str], responses: list[str],
                   batch_size: int = 8, device: str | None = None) -> torch.Tensor:
    """Median length of each block's residual stream at single response tokens, (n_blocks,)."""
    device = device or model.device
    with torch.no_grad():
        norms = [h[:, n_prompt:].float().norm(dim=-1).cpu()   # (n_blocks, n_response)
                 for h, n_prompt in _per_example_hidden(model, tok, prompts, responses, batch_size, device)]
    return torch.cat(norms, dim=1).median(dim=1).values


# ---------------------------------------------------------------------------
# 6. Steering sweep + monitoring harness
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def steer(model, layer: int, vector, coeff: float, positions: str = "response"):
    """Add coeff · vector to block `layer`'s output on every forward pass while active.

    positions (generation runs one pass over the prompt, then one pass per new token,
    reading earlier positions from the KV cache):
      "all"      — A: every position of every pass: the whole prompt, then each new token.
      "response" — B, the paper's code: only the last position of each pass: the last
                   prompt token, then each new token (paper §3.2, "at each decoding step").
    """
    vec = torch.as_tensor(vector, dtype=torch.float32)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output   # (B, seq, d)
        push = (coeff * vec).to(device=h.device, dtype=h.dtype)
        if positions == "all":
            h += push
        elif positions == "response":
            h[:, -1, :] += push   # left padding: the last position is real for every row
        else:
            raise ValueError(f"positions must be 'all' or 'response', not {positions!r}")

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()


@contextlib.contextmanager
def cap(model, layers: list[int], axis, taus):
    """Activation capping (Assistant Axis paper, eq. 1), at every position of every pass, at each
    block in `layers`:

        h ← h − v̂ · min(⟨h, v̂⟩ − τ_L, 0)

    A floor, not a push: where the projection on the axis is below τ_L, add back exactly the
    missing amount along v̂; at or above it, h is untouched. `axis` (n_blocks, d) points toward
    the Assistant; `taus` (n_blocks,) are in `project` units (unit v̂). Yields a dict counting
    positions seen and capped per block, so a run can say how often the cap fired."""
    stats = {L: dict(seen=0, capped=0) for L in layers}

    def make_hook(L):
        v = torch.as_tensor(axis[L], dtype=torch.float32)
        v = v / v.norm()
        tau = float(taus[L])

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output   # (B, seq, d)
            vd = v.to(h.device)
            short = (h.float() @ vd - tau).clamp(max=0.0)             # min(⟨h, v̂⟩ − τ, 0), (B, seq)
            h -= (short.unsqueeze(-1) * vd).to(h.dtype)
            stats[L]["seen"] += short.numel()
            stats[L]["capped"] += int((short < 0).sum())
        return hook

    handles = [model.model.layers[L].register_forward_hook(make_hook(L)) for L in layers]
    try:
        yield stats
    finally:
        for h in handles:
            h.remove()


def replay(model, tok, convs: list[list[str]], system: str | None = None, seed: int = 0,
           **sample_kw) -> pd.DataFrame:
    """Multi-turn conversations with fixed user messages: `convs[i]` is conversation i's user
    turns in order, and the model writes its own answer every turn (the paper replays its case
    studies this way, §6). One `sample` batch per turn over the conversations still running.
    Returns one row per (conv, turn): user, prompt (the full chat so far, ending at the
    assistant header), response, n_tokens, truncated."""
    hist: list[list[tuple[str, str]]] = [[] for _ in convs]
    rows = []
    for t in range(max(map(len, convs))):
        live = [i for i, c in enumerate(convs) if t < len(c)]
        prompts = [chat_prompt(tok, system, convs[i][t], hist[i]) for i in live]
        gens = sample(model, tok, prompts, seed=seed + t, **sample_kw)
        for i, p, g in zip(live, prompts, gens):
            hist[i].append((convs[i][t], g["response"]))
            rows.append(dict(conv=i, turn=t, user=convs[i][t], prompt=p, **g))
    return pd.DataFrame(rows)


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
    positions: str = "response",
    scale=None,
    **sample_kw,
) -> pd.DataFrame:
    """Generate on `questions` while adding α · vectors[L] to block L's output,
    for every (L, coeff), at `positions`: "all" (A) or "response" (B), see `steer`; or
    "clean" (C), see `sample_stepwise` (h_ℓ ← h_ℓ + α·v_ℓ). Plus one
    unsteered baseline (layer = -1, coeff = 0). No system prompt by default: we steer
    the plain Assistant.

    scale: per-block lengths (e.g. `residual_norms`). Then each coeff is a fraction,
    α = coeff · scale[L], so every block gets the same push relative to its stream.
    Without it, α = coeff.

    Returns one row per generation: layer, coeff (= α), frac (coeff as given, with
    scale), positions, qid, question, sample, response, ... Judge it afterwards.
    """
    prompts = [chat_prompt(tok, system, q) for q in questions for _ in range(n_samples)]
    meta = [dict(qid=qi, question=q, sample=s)
            for qi, q in enumerate(questions) for s in range(n_samples)]
    frames = []

    def run(layer, alpha, frac):
        if layer < 0:
            gens = sample(model, tok, prompts, **sample_kw)
        elif positions == "clean":
            gens = sample_stepwise(model, tok, prompts, layer, vectors[layer], alpha, cache="clean", **sample_kw)
        else:
            with steer(model, layer, vectors[layer], alpha, positions):
                gens = sample(model, tok, prompts, **sample_kw)
        frames.append(pd.DataFrame([dict(layer=layer, coeff=alpha, frac=frac, positions=positions, **m, **g)
                                    for m, g in zip(meta, gens)]))

    if include_baseline:
        run(-1, 0.0, 0.0)
    for L in layers:
        for c in coeffs:
            alpha = float(c) * (1.0 if scale is None else float(scale[L]))
            run(L, alpha, float(c) if scale is not None else None)
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
    return pd.read_json(CACHE_DIR / f"{name}.jsonl", orient="records", lines=True, precise_float=True)   # 0.3, not 0.30000000000000004


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
