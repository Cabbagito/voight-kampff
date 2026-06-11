"""
Shared utilities for the linear-probes experiments.

Layout:
  1. Paths / constants
  2. Model loading
  3. RepE dataset construction (honest/untruthful contrastive pairs)
  4.  Activation collection (last-token + mean-pool)
  4b. On-policy generation (Exp 3+)
  5.  Probe training: LR (sklearn)  +  MM (diff-of-means)
  6.  Eval helpers
  7.  Cache I/O

The substantive gears (residual-stream extraction + the MM feature direction)
were the original TODO_PETER blocks; now implemented. Everything else is glue.
"""

from __future__ import annotations

import contextlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------

TOPIC_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOPIC_DIR.parents[1]
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"
DATA_DIR = TOPIC_DIR / "data"
CACHE_DIR = TOPIC_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Qwen3 chat-format tokens. We hand-build the prompt rather than using
# tokenizer.apply_chat_template() so that (a) we know exactly where the
# assistant content begins, (b) we can place the activation-extraction
# token at the true last token of the statement with no trailing special-
# token surprises.
QWEN_USER = "<|im_start|>user\n"
QWEN_USER_END = "<|im_end|>\n"
QWEN_ASST = "<|im_start|>assistant\n"

PERSONA_TEMPLATE = "Pretend you're {persona} person making statements about the world."


# ---------------------------------------------------------------------------
# 2. Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(dtype=torch.bfloat16, device: str = "cuda"):
    """Load local Qwen3-4B. Eval mode, no grad — we never train the LLM here.

    Returns (model, tokenizer). Tokenizer has padding_side='left' set, which
    makes "last token per sequence" trivially equal to position -1.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=dtype, device_map=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ---------------------------------------------------------------------------
# 3. RepE dataset construction
# ---------------------------------------------------------------------------

def _make_prompt(persona_word: str, statement_prefix: str) -> str:
    """Hand-build a Qwen3 chat-format prompt that ends with statement_prefix.

    Structure:
        <|im_start|>user
        Pretend you're an {persona} person making statements about the world.
        <|im_end|>
        <|im_start|>assistant
        {statement_prefix}        <-- last token of this is our extraction site

    No <|im_end|> after the assistant content, no add_generation_prompt
    trailing newline — we want the last real content token to be the very
    last token of the sequence.
    """
    user_msg = PERSONA_TEMPLATE.format(persona=persona_word)
    return (
        f"{QWEN_USER}{user_msg}{QWEN_USER_END}"
        f"{QWEN_ASST}{statement_prefix}"
    )


def _make_full_prompt(persona_word: str, user_message: str, assistant_response: str) -> str:
    """Switcheroo-style prompt: persona instruction + UM in user role,
    full AR in assistant role. Persona placement matches _make_prompt so a
    probe trained on Exp 1 sees the same SP structure.

    Structure:
        <|im_start|>user
        Pretend you're an {persona} person making statements about the world.

        {user_message}
        <|im_end|>
        <|im_start|>assistant
        {assistant_response}      <-- last token of this is our extraction site
    """
    user_msg = f"{PERSONA_TEMPLATE.format(persona=persona_word)}\n\n{user_message}"
    return (
        f"{QWEN_USER}{user_msg}{QWEN_USER_END}"
        f"{QWEN_ASST}{assistant_response}"
    )


def _make_gen_prompt(persona_word: str, user_message: str, think: bool = True) -> str:
    """Generation prompt: persona instruction + UM in user role, then the
    assistant header with NO content — the model continues from here.

    Same SP structure / placement as _make_full_prompt, just truncated right
    after `<|im_start|>assistant\\n` so the model generates the AR on-policy.

    `think=True` matches Qwen3's default (apply_chat_template enable_thinking=True)
    and the model emits a `<think>...</think>` reasoning trace before its answer.
    `think=False` appends the empty `<think>\\n\\n</think>\\n\\n` block that the
    Qwen3 template uses for enable_thinking=False, so the model skips reasoning
    and generates the answer directly — no truncated-trace artifact, and the
    last-token extraction site is the end of a real answer. (Verified against
    tokenizer.apply_chat_template on Qwen3-4B.)

    To extract activations afterwards, append the generated text to this prompt
    and you get exactly the _make_full_prompt structure (so the Exp 1 probe sees
    the same surface form it was trained under).
    """
    user_msg = f"{PERSONA_TEMPLATE.format(persona=persona_word)}\n\n{user_message}"
    prompt = f"{QWEN_USER}{user_msg}{QWEN_USER_END}{QWEN_ASST}"
    if not think:
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def build_repe_pairs(
    csv_path: Path,
    tokenizer,
    seed: int = 0,
    n_train_pairs: int = 256,
    n_test_pairs: int = 64,
):
    """Build train + test sets of (honest_prompt, untruthful_prompt) pairs.

    Reproduces the Zou et al. construction from
    representation-engineering/examples/honesty/utils.py, adapted to Qwen3
    chat format and trimmed for clarity.

    Why prefix-truncation? Zou et al. use EVERY token-prefix of each true
    statement (positions 1 .. len-5) as a separate training example. This
    inflates ~600 statements into ~30k examples and exposes the probe to
    activations at many different positions inside the statement, not just
    the end. M&T didn't do this — they probed only the final-token activation.
    We keep Zou's approach for v1 because the RepE dataset is small.

    NOTE: we only use the TRUE statements from the CSV. The honest/dishonest
    contrast lives entirely in the SYSTEM PROMPT, not in the statement
    content. (This is why Experiment 2's switcheroo is so important: the
    probe trained here has only ever seen identical AR text, so it has no
    way to learn anything except SP-reading.)

    Returns:
        train: list of dicts {prompt: str, label: int, statement_id: int}
        test:  same shape, disjoint statements
        Each statement contributes ~2 * (len-6) rows (one honest, one
        untruthful per prefix length).
    """
    rng = random.Random(seed)

    df = pd.read_csv(csv_path)
    true_statements = df[df["label"] == 1]["statement"].tolist()
    rng.shuffle(true_statements)

    # Split statements (not rows) into train / test so no statement leaks.
    n_train_stmts = n_train_pairs
    n_test_stmts = n_test_pairs
    train_stmts = true_statements[:n_train_stmts]
    test_stmts = true_statements[n_train_stmts : n_train_stmts + n_test_stmts]

    def expand(statements, start_id):
        rows = []
        for sid, stmt in enumerate(statements, start=start_id):
            tokens = tokenizer.tokenize(stmt)
            # Skip the last 5 tokens — too-late prefixes are nearly the
            # whole statement and add little. (Matches Zou et al.)
            for idx in range(1, max(2, len(tokens) - 5)):
                prefix = tokenizer.convert_tokens_to_string(tokens[:idx])
                rows.append({
                    "prompt": _make_prompt("an honest", prefix),
                    "label": 1,  # honest
                    "statement_id": sid,
                })
                rows.append({
                    "prompt": _make_prompt("an untruthful", prefix),
                    "label": 0,  # untruthful
                    "statement_id": sid,
                })
        return rows

    train = expand(train_stmts, start_id=0)
    test = expand(test_stmts, start_id=10_000)
    rng.shuffle(train)
    return train, test


def build_truth_dataset(
    csv_path: Path,
    seed: int = 0,
    n_train: int = 400,
    n_test: int = 150,
    persona: str = "an honest",
):
    """Exp 4 truth probe: hold the SP CONSTANT, label by is_true of the statement.

    The mirror image of build_repe_pairs. There, the label was the persona and
    the statement was held constant — so the probe could only read the SP. Here
    the persona is held constant (honest by default) and the label is the
    statement's truth value, so the SP carries ZERO label information: the probe
    must find a feature in the statement itself that distinguishes true from
    false under identical surface conditions. That feature — if it exists
    linearly — is the M&T "knowledge-vs-assertion" truth direction.

    Each statement is presented IN FULL as the assistant content (no Zou-style
    prefix expansion), so the last-token extraction site is the END of the
    statement — M&T §3's "summarization onto the final token" position, the
    truth-relevant one.

    Split by statement (no leakage). Returns (train, test), each a list of
    {prompt, label, statement}; label 1 = true, 0 = false.
    """
    rng = random.Random(seed)
    df = pd.read_csv(csv_path)

    # facts_true_false.csv has 27 duplicate-statement rows and one statement
    # labelled BOTH true and false ("The Nile River is the longest...") — drop
    # contradictory statements entirely, then collapse exact duplicates, so no
    # statement leaks across the train/test split and no label is self-contradictory.
    contradictory = df.groupby("statement")["label"].nunique()
    df = df[~df["statement"].isin(contradictory[contradictory > 1].index)]
    df = df.drop_duplicates(subset="statement").reset_index(drop=True)

    rows = list(df[["statement", "label"]].itertuples(index=False, name=None))
    rng.shuffle(rows)
    train_rows = rows[:n_train]
    test_rows = rows[n_train : n_train + n_test]

    def build(rs):
        return [
            {"prompt": _make_prompt(persona, stmt), "label": int(lab), "statement": stmt}
            for stmt, lab in rs
        ]

    return build(train_rows), build(test_rows)


# ---------------------------------------------------------------------------
# 4. Activation collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_activations(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    """Residual-stream activation at the LAST token of each sequence, after
    transformer block `layer`. Returns (N, hidden) on CPU.

    `hidden_states[layer + 1]` is the output of block `layer` (index 0 is the
    embeddings). padding_side='left' guarantees the real last token sits at
    position -1 for every sequence in the batch.

    (nnsight steering version lives in Exp 5 when we actually need write-access
    to the residual stream — not carried here as dead code.)
    """
    activations = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer + 1]  # (B, seq, hidden)
        activations.append(hidden_states[:, -1, :].cpu())  # (B, hidden)
    return torch.cat(activations, dim=0)  # (N, hidden)


# ---------------------------------------------------------------------------
# 4b. On-policy generation (Exp 3+)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 200,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[str]:
    """Greedy-decode a continuation for each prompt; return ONLY the new text.

    Glue, not a gear: standard HF `.generate()` with greedy decoding (temp=0,
    do_sample=False) for reproducibility. Pass `_make_gen_prompt(...)` strings
    (they end at the assistant header) and you get the model's on-policy AR.

    padding_side='left' (set in load_model_and_tokenizer) means every sequence
    in a batch is padded on the LEFT to a common length, so the newly generated
    tokens are exactly `output[:, input_len:]` for the whole batch.
    """
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = gen[:, inputs["input_ids"].shape[1] :]
        outs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outs


def count_response_tokens(
    tokenizer, gen_prompts: list[str], full_texts: list[str]
) -> list[int]:
    """How many tokens the AR adds on top of its generation prompt.

    Used to mean-pool over the response span only. Because we pad on the LEFT,
    the response occupies the FINAL n tokens of the full sequence, so the count
    returned here is exactly "how many trailing positions are the response".
    """
    return [
        len(tokenizer(full).input_ids) - len(tokenizer(prompt).input_ids)
        for prompt, full in zip(gen_prompts, full_texts)
    ]


@torch.no_grad()
def collect_activations_layers(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """Last-token residual activations for MULTIPLE layers in ONE forward pass.

    For layer sweeps: `output_hidden_states=True` already computes every layer,
    so re-running collect_activations per layer wastes N-1 forward passes. Returns
    {layer: (N, hidden) tensor on CPU}.
    """
    out = {L: [] for L in layers}
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        hidden = model(**inputs, output_hidden_states=True).hidden_states
        for L in layers:
            out[L].append(hidden[L + 1][:, -1, :].cpu())  # hidden[L+1] = block L output
    return {L: torch.cat(v, dim=0) for L, v in out.items()}


@torch.no_grad()
def collect_activations_mean(
    model,
    tokenizer,
    texts: list[str],
    n_resp_tokens: list[int],
    layer: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    """Like collect_activations, but MEAN-pool over the last `n_resp_tokens[i]`
    positions of each sequence (the response span) instead of taking the last
    token only. This is Apollo's aggregation; compare against last-token.

    With padding_side='left' the response is the trailing block of real tokens,
    so "last n_resp positions" == "the response" with no masking gymnastics.
    """
    activations = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_n = n_resp_tokens[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, return_tensors="pt").to(device)
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states[layer + 1]
        for j, n in enumerate(batch_n):
            n = max(1, int(n))
            activations.append(hidden_states[j, -n:, :].mean(dim=0).cpu())
    return torch.stack(activations, dim=0)  # (N, hidden)


# ---------------------------------------------------------------------------
# 5. Probes
# ---------------------------------------------------------------------------

def fit_lr_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    """L2-regularized logistic regression. sklearn does the lifting.

    Note on normalization: M&T zero-mean/unit-var each dimension before
    fitting. We don't here; sklearn's LBFGS handles raw activations fine
    and C=1.0 is a sane starting regularization. Easy to add later via
    StandardScaler if you want to compare.
    """
    X = np.asarray(X, dtype=np.float32)
    return LogisticRegression(C=C, max_iter=2000).fit(X, y)


def mm_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mass-mean (diff-of-means) probe direction: μ_pos - μ_neg.

    Unlike LR's max-margin boundary, this doesn't tilt to cancel interference
    from correlated features, so it's a better *feature-direction* estimate
    (M&T §5–6) — intervening along it shifts behavior more (higher NIE). Not
    L2-normalized: callers choose (Exp 5 wants both raw and norm-matched).
    LDA upgrade (Σ⁻¹(μ⁺−μ⁻)) deferred — equivalent to LR in the infinite-data,
    homoscedastic limit; only matters for IID eval.
    """
    pos_mean = X[y == 1].mean(axis=0)
    neg_mean = X[y == 0].mean(axis=0)
    return pos_mean - neg_mean


def mm_score(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Score points by projection onto the MM direction. Higher = more "pos"."""
    X = np.asarray(X, dtype=np.float32)
    return X @ direction.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Eval
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# 7. Cache I/O
# ---------------------------------------------------------------------------

def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pt"


def save_cache(name: str, **tensors) -> Path:
    p = cache_path(name)
    torch.save(tensors, p)
    return p


def load_cache(name: str) -> dict:
    return torch.load(cache_path(name), map_location="cpu", weights_only=False)


def cache_exists(name: str) -> bool:
    return cache_path(name).exists()


# ---------------------------------------------------------------------------
# 8. Steering / causal intervention (Exp 5)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def steering(model, layer: int, vector, coeff: float, pos_mask=None, device: str = "cuda"):
    """[GEAR — Peter] Write-side residual intervention, the mirror of collect_activations.

    For the duration of the `with` block, add `coeff * vector` to the residual stream
    coming OUT of transformer block `layer`, on every forward pass — so the single-pass
    logit-NIE readout and a full `.generate()` (behavioral demo) are both steered.
    Injecting along a probe direction and watching the model's OWN output move is what
    makes this a causal test rather than a correlational one.

    In transformers 5.x a Qwen3 block's forward hook receives the residual stream as a
    bare tensor (B, seq, hidden) — we modify it in place. `pos_mask` (B, seq) selects
    which positions to steer (None → all, e.g. during generation); the NIE readout
    passes a mask that zeroes the final/readout token so we steer the context and read
    a clean prediction.
    """
    def hook(module, inputs, output):
        vec = torch.as_tensor(vector, dtype=output.dtype, device=output.device)
        if pos_mask is None:
            output += coeff * vec                                      # (hidden,) broadcasts over (B, seq, hidden)
        else:
            mask = pos_mask.to(device=output.device, dtype=output.dtype).unsqueeze(-1)  # (B, seq, 1)
            output += mask * coeff * vec                               # (B, seq, 1) * (hidden,)

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()


def matched_steering_vector(direction, mu_pos, mu_neg):
    """M&T matched-strength normalization. Scale `direction` so adding it shifts
    that direction's OWN readout by exactly the true−false mean gap:

        v = [ d·(μ⁺−μ⁻) / (d·d) ] · d      (parallel to d)

    Then `coeff=1` means "move one true−false gap along this probe's readout",
    so θ_lr and θ_mm are compared at matched effect-size and any NIE difference is
    about *causality*, not raw magnitude. `d` = the probe direction (LR coef_ or MM).
    """
    d = np.asarray(direction, dtype=np.float64)
    gap = float(d @ (np.asarray(mu_pos, np.float64) - np.asarray(mu_neg, np.float64)))
    v = (gap / float(d @ d)) * d
    return v.astype(np.float32)


def _make_eval_prompt(statement: str) -> str:
    """True/False judgment prompt: model answers with one word. Thinking disabled
    (empty <think></think>) so the very next token is the verdict — that's where we
    read logP(True) − logP(False)."""
    user = (
        'Is the following statement true or false? '
        'Reply with exactly one word: "True" or "False".\n\n'
        f"{statement}"
    )
    return f"{QWEN_USER}{user}{QWEN_USER_END}{QWEN_ASST}<think>\n\n</think>\n\n"


def truth_token_ids(tokenizer):
    """Single-token ids for True/False answer variants (with and without a leading
    space, capitalized and lower). Returned as (true_ids, false_ids) for logsumexp."""
    def singletons(words):
        ids = []
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        return sorted(set(ids))
    return (singletons([" True", "True", " true", "true"]),
            singletons([" False", "False", " false", "false"]))


@torch.no_grad()
def truth_logit_diff(
    model,
    tokenizer,
    prompts: list[str],
    true_ids: list[int],
    false_ids: list[int],
    steer: dict = None,
    batch_size: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    """Δ = logsumexp logP(True-variants) − logsumexp logP(False-variants) at the
    final position of each prompt. The model's OWN truth judgment — independent of
    the probe (so the causal test isn't circular).

    steer: None for baseline, or dict(layer=, vector=, coeff=, mode='all'|'context').
        'context' steers all real tokens EXCEPT the final readout position (the
        upstream-perturb / downstream-read design). Mask is rebuilt per batch from
        the attention mask, so padding is handled correctly.
    """
    diffs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inp = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        ctx = contextlib.nullcontext()
        if steer is not None:
            pm = None
            if steer["mode"] == "context":
                pm = inp["attention_mask"].clone()
                pm[:, -1] = 0  # left-padded => -1 is the real final (readout) token
            ctx = steering(model, steer["layer"], steer["vector"], steer["coeff"],
                           pos_mask=pm, device=device)
        with ctx:
            logits = model(**inp).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        t = torch.logsumexp(lp[:, true_ids], dim=-1)
        f = torch.logsumexp(lp[:, false_ids], dim=-1)
        diffs.append((t - f).cpu())
    return torch.cat(diffs).numpy()
