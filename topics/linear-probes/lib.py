"""
Shared utilities for the linear-probes experiments.

Layout:
  1. Paths / constants
  2. Model loading
  3. RepE dataset construction (honest/untruthful contrastive pairs)
  4. Activation collection            <-- has a TODO_PETER (the hook)
  5. Probe training: LR (sklearn)  +  MM (diff-of-means)  <-- MM is TODO_PETER
  6. Eval helpers
  7. Cache I/O

The TODO_PETER blocks are deliberately the substantive gears — the hook
function (where you intervene in the forward pass) and the MM direction
(the simplest possible "feature direction" estimator, contrasted with LR
in Marks & Tegmark). Everything else is glue.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# 4. Activation collection                              <-- TODO_PETER
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
    """Forward `texts` through `model`, return the residual-stream activation
    at the LAST token of each sequence after transformer block `layer`.

    TODO_PETER -------------------------------------------------------------

    THIS is the central mechanical primitive of the entire probing setup.
    Implementing it well-and-understood is more valuable than the rest of
    lib.py combined. Read the M&T paper §3 (causal localization) before
    you write it — it'll motivate why the choice of (layer, position)
    matters and what you're committing to.

    What "residual stream" means here:
        Each transformer block reads x_in, computes attn+MLP updates, and
        outputs x_out = x_in + attn(...) + mlp(...). The "residual stream
        at layer L" is x_out of block L (equivalently: input to block L+1).
        In HF Qwen3, that's the output of model.model.layers[L].

    Shape you want to return:
        (N, hidden_size), where N = len(texts). One vector per input,
        taken at the LAST non-pad token (we set padding_side='left', so
        last token is always position -1 regardless of input length).
        CPU is fine — keep memory pressure off the GPU between batches.
        bf16 is fine — probe training will float32-cast as needed.

    Two viable implementations:

    (A) HF native — easiest, no extra deps:
        Call `model(**inputs, output_hidden_states=True)`. The returned
        `hidden_states` is a tuple of length n_layers+1; index `layer+1`
        is the output of block `layer` (index 0 is the embeddings). Take
        position -1, append to a list, cat at the end.

    (B) nnsight — more flexible, what you'll need in Experiment 5 for
        STEERING (adding a vector into the residual stream during a forward
        pass). Pattern:
            with model.trace(inputs) as t:
                act = model.model.layers[layer].output[0][:, -1, :].save()
        nnsight is slightly slower for pure read-only collection but the
        API for write-interventions is much cleaner than HF hooks.

        Spend a moment on the nnsight docs before picking (A). Knowing
        the abstraction now pays off later — but doing (A) first to ground
        out what you're abstracting is also legitimate.

    Tokenization detail:
        Use `tokenizer(texts, padding=True, return_tensors='pt').to(device)`.
        With padding_side='left', position -1 is always the real last token.
        Sanity-check this on a batch of mixed-length inputs before you trust
        the activation vectors.

    Memory budget for Qwen3-4B at hidden=2560:
        N=30k * 2560 * 2 bytes (bf16) ≈ 150 MB. Fits in RAM easily.
        Keep activations on CPU as you go; don't accumulate on GPU.
    -----------------------------------------------------------------------
    """

    activations = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer + 1]  # (B, seq, hidden)
        activations.append(hidden_states[:, -1, :].cpu())  # (B, hidden)
    return torch.cat(activations, dim=0)  # (N, hidden)


def collect_activations_nnsight(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    """Same spec as collect_activations, but implemented with nnsight tracing and
    save().

    See the nnsight docs for how to do this. The main difference is that you
    trace the model in batches, and inside the trace context you index into
    the residual stream tensor and call .save() on it. The API is designed
    so that you can also modify the tensor if you want to do interventions
    later (Experiment 5).
    """
    from nnsight import NNsight
    ns = NNsight(model)

    activations = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, return_tensors="pt").to(device)
        with ns.trace(inputs["input_ids"], attention_mask=inputs["attention_mask"]):
            # layer.output is (B, seq, hidden); slicing -> (B, hidden)
            act = ns.model.layers[layer].output[:, -1, :].save()
        activations.append(act.cpu())  # (B, hidden)
    return torch.cat(activations, dim=0)  # (N, hidden)



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

    TODO_PETER -------------------------------------------------------------

    Tiny code, important concept. Read M&T §5 before writing this.

    THE GEAR:
        LR finds the max-margin separating hyperplane. Under superposition
        (multiple features sharing the residual stream non-orthogonally),
        LR's θ TILTS away from the true feature direction θ_t to suppress
        interference from other features that happen to correlate with the
        label. Result: LR is often a better classifier than MM but a worse
        FEATURE-DIRECTION estimator.

        Diff-of-means makes one assumption: the two classes differ only in
        their mean projection onto the feature. Under that assumption it's
        an unbiased estimate of the feature direction. It doesn't tilt to
        cancel interference; it just points at "where the means differ".
        That's why intervening along θ_mm changes downstream behavior more
        than intervening along θ_lr (M&T §6, NIE 0.85 vs 0.33 in one case).

        Aside (LDA-corrected): with homoscedasticity (both classes share
        Σ), the optimal direction is Σ⁻¹ (μ⁺ - μ⁻). That's equivalent to
        LR in the infinite-data limit. Skip Σ⁻¹ for now; it's a small
        upgrade later for the IID-eval setting.

    INPUTS:
        X: (N, D) activations, np.float32
        y: (N,)   labels in {0, 1}

    OUTPUT:
        d: (D,) np.float32, the direction vector. Do NOT L2-normalize
        inside this function — let callers choose (you'll want both raw
        and norm-matched versions during Experiment 5's intervention).

    Implementation should be 3 lines of numpy.
    -----------------------------------------------------------------------
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
