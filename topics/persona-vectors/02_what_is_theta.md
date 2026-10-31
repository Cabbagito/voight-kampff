# Exp 2: What is θ_mm?

**The question.** In `linear-probes` we found θ_mm, a "truth" direction in Qwen3-4B at block 20. It separates true from false statements (AUROC 0.97), and pushing along it flips the model's own true/false verdict. But pushed hard in free text, the model narrated its own honesty ("I'm not honest") instead of stating clean falsehoods [ours, linear-probes Exp 3]. So which is it?
- an **honesty disposition**, close to our dishonest vector;
- **"what I write vs what I know"**: dishonest *and* mystic would both touch it;
- or **something else** (world-truth, plain plausibility) that the persona vectors don't capture.

"θ_truth" is a label, not a finding. All inference, no new vectors: the directions come from Exp 1 (`cache/persona_<trait>.pt`) and linear-probes (`../linear-probes/cache/*_L20.pt`).

## The directions

All at block 20 (0-indexed, the output of block 20, the same convention in both repos), unit length:

| direction | built from | + points to |
|---|---|---|
| θ_mm (truth) | pasted true / false statements, last token, μ_true − μ_false (Marks & Tegmark style) | "this statement is true" |
| θ_lr (truth) | the same data, logistic regression | "true" |
| θ_SP (honest persona) | Zou-style "pretend you're honest / untruthful" prompts, last token, μ_honest − μ_untruthful | honest persona |
| v_dishonest, v_mystic, v_pirate | Exp 1: POS − NEG answers, averaged over answer tokens | the trait |

So if θ_mm is "honesty", cos(θ_mm, v_dishonest) comes out **negative**.

**Transfer caveat.** θ_mm comes from pasted statements read at one token; the persona vectors come from generated answers averaged over tokens. A low cosine can mean the contexts differ, not the concepts. The steering steps don't depend on that.

## Steps

1. **Cosines.** Every pair, plus a null: 2,000 random directions against θ_mm. Pirate is the control: a dishonest cosine counts only if it clearly beats pirate's.
2. **Steer the persona vectors, read the truth verdict** (no judge): push each one and read the true/false logit difference on the probe's facts.
3. **The reverse:** push θ_mm, judge dishonesty on Exp 1's held-out questions.
4. **Pirate leak test:** push pirate, judge dishonesty. Only if it leaks, project dishonest out and check again.

## Predictions

_Peter_ (2026-10-31, before anything ran; assumes θ points to "telling the truth"):
- cos(θ_mm, v_dishonest) between −0.75 and −0.4; |cos(θ_mm, v_pirate)| < 0.1.
- Mystic moves the verdict toward false, but not robustly: it still says untrue things, for a different reason.

_Claude_ (written after Peter's): cos(θ_mm, v_dishonest) about −0.1 (−0.25 to 0): the harm split showed v_dishonest barely touches neutral facts, and θ_mm is all neutral facts. |cos(θ_mm, v_pirate)| < 0.05. cos(θ_SP, v_dishonest) much stronger, about −0.4, since both are persona contrasts. Mystic moves the verdict about as little as dishonest does.

_Peter_ (2026-10-31, after step 1's cosines, before the overlap checks): removing the shared part, the persona vectors' cosines with each other drop. Against the probes, θ_SP's cosines move toward positive and θ_mm's toward negative; no guess at how much. Unsure whether the shared part can be taken out cleanly.

_Claude_ (same): the massive dimensions explain a third of the overlap at most. With the shared part out, θ_mm vs pirate goes to about 0, and dishonest stays near −0.15.

_Peter_ (2026-10-31, step 2, before it ran): true statements go down under every push; leaning (b), a slide toward "False", for most of them; unsure.

_Claude_ (same, written after Peter's), reading Δ on true and on false statements:
- −θ_mm: (b), both slide down (the mirror of linear-probes' NIE).
- v_dishonest: a weak flip at best, the gap shrinking without crossing by 0.4 (45%); or a slide toward "True", yes-saying from its reassuring flavour (35%); (b) 20%.
- v_mystic: the gap stays above 80% of unsteered; it believes in crystals, not that Paris is in Spain.
- v_pirate and a random direction: nothing until the push breaks the model, then both verdicts sink toward 0 (unsure), which is neither (a) nor (b).

## Results

### Step 1: cosines (2026-10-31, CPU)

At block 20, unit vectors. 2,000 random directions against θ_mm stay under |0.04| (95th percentile) [ours]:

| | θ_mm (truth) | θ_SP (honest persona) | v_dishonest | v_mystic | v_pirate |
|---|---|---|---|---|---|
| θ_mm (truth) | 1 | 0.26 | **−0.19** | −0.03 | −0.11 |
| θ_SP (honest persona) | | 1 | **−0.41** | −0.07 | −0.18 |
| v_dishonest | | | 1 | 0.19 | 0.33 |
| v_mystic | | | | 1 | **0.43** |

θ_lr runs weaker everywhere (−0.11 with dishonest; 0.34 with θ_mm).

- dishonest leans away from θ_mm by −0.19: real (5× the random range), with the expected sign, but small (cos² ≈ 4%).
- The odd part: **the three persona vectors overlap** (0.19–0.43), and pirate leans away from θ_mm too (−0.11). So the planned control, "dishonest must beat pirate", is shaky.

**Where the overlap comes from.**
- **Massive dimensions: not here.** Zeroing the 5 giant dims (0, 4, 7, 100, 396) moves no cosine by more than 0.02 [ours].
- **A shared "playing a character" part.** For each persona, the shared axis is the mean of the *other two* (leave-one-out, so its own trait never enters). Then we remove it and recompute. The leftovers' cosines with each other can't be read (they're forced apart), so only the probes count [ours]:

  | persona | cos with the other two's mean | θ_mm before → after | θ_SP before → after |
  |---|---|---|---|
  | dishonest | 0.31 | −0.19 → **−0.18** | −0.41 → **−0.39** |
  | mystic | 0.38 | −0.03 → +0.04 | −0.07 → +0.08 |
  | pirate | 0.49 | −0.11 → **−0.05** | −0.18 → **−0.03** |

  Pirate's link to both probes came through the shared part. Dishonest's is its own. The shared axis built without dishonest (mystic + pirate) leans only slightly against the probes: −0.08 against θ_mm, −0.14 against θ_SP.
- **Reading [synthesis]:** every POS prompt says "play a character" and every NEG prompt "be the plain assistant", so POS − NEG for any trait also carries "character vs default assistant". That's the subject of Exp 3's paper, *The Assistant Axis* [paper: Lu et al. 2026]. The axis here is crude, though: it's estimated from two vectors.
- **Against the predictions:** Peter's −0.75 to −0.4 for dishonest came out at −0.19. Pirate (−0.11) missed both of our bounds. Claude's −0.1 (−0.25 to 0) and θ_SP ≈ −0.4 (−0.41) held. Massive dims as a third of the overlap was wrong (zero). Peter's θ_SP moving toward positive held for pirate and mystic; θ_mm moving more negative didn't happen.

### Step 2: push the vectors, read the verdict (2026-10-31, `theta` session, 80 s)

The readout is linear-probes Exp 3's. Qwen is asked *"Is the following statement true or false? Reply with exactly one word"* about 150 held-out facts (70 true, 80 false). We read Δ = log P(True) − log P(False) at the next token. Unsteered: +17.5 on true statements, −19.1 on false, the same as linear-probes [ours].

Each direction is pushed on every token except the last (the readout stays clean), at 0.1–0.6 × the stream's length. That's at block 18 (where the persona vectors steer) and at block 20 (where θ_mm lives). A random direction is the control.

What each push can do, with two statements:
- **(a) liar:** it keeps the truth and inverts the report. True ↓ below 0, false ↑ above 0: the lines cross.
- **(b) looks false:** the reading itself is corrupted. True ↓, false stays low.
- **(c) yes-man:** everything ↑ toward "True".
- **(d) broken:** both sink toward 0 (unsure).

![The verdict under each push](figures/theta_verdict_push.png)

*Mean Δ per push and strength. Green: true statements; red: false. Solid: block 18; dashed: block 20. 0 = undecided.* [ours]

- **−θ_mm: a clean (b).** True statements go from +17 to −12 at 0.4 (all called false), while false ones stay at −19. At 0.6 both sink to −7, so it breaks.
- **dishonest: a different pattern at each block, and never (a).**
  - **Block 18: (c), a yes-man.** False statements climb from −19 to −1 at 0.4 and +3 at 0.6; at 0.6, 75% are called "True". True ones stay True (100%). This is the same reassuring flavour as Exp 1's harm split, now on plain facts.
  - **Block 20: (b), like θ_mm but weaker.** True statements fall to +8.5 at 0.4 and −2.5 at 0.6 (40% still called true); false ones stay at about −17. θ_mm gets there by 0.3.
- **mystic, pirate: about random.** Both lines drift a little toward 0 (to about ±13–15 at 0.6), as the random push does. Mystic doesn't change whether Paris is in France.
- **Reading [synthesis]:** at θ_mm's block, the dishonest vector carries a small piece of "this looks false", which fits the −0.19 cosine. Two blocks earlier, the same POS − NEG recipe gives "agree with everything". So "dishonest" isn't one thing across depth, and at neither block is it a liar who keeps the truth and inverts it. Mystic and pirate leave the model's factual judgement alone.
- **Against the predictions:** Peter's "(b) for most" held for −θ_mm and for dishonest at block 20; mystic and pirate were nothing, not (b). Claude's weak flip (45%) didn't happen; the yes-man (35%) showed at block 18. Neither of us expected the block to change the pattern.
