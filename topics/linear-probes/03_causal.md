# Exp 3: Is the truth direction a lever?

**The question.** Exp 2's truth directions *correlate* with truth (0.97 AUROC). Does the model *use* them? Push the activations along a direction and see whether the model's own true/false verdict moves. If it does, the direction is a lever; if not, it's a correlate.

And the comparison we kept both probes for: θ_lr and θ_mm classify equally well (0.969 vs 0.968), but they point 70° apart. **Is one a better lever than the other?** Marks & Tegmark found MM directions move behaviour more than LR at the same accuracy (§6) [paper].

This follows *The Geometry of Truth* §6 (the normalised indirect effect, NIE). Background is in `context.md` §5. Labels: [paper] a paper says it; [ours] we measured it; [synthesis] our inference.

Originally run as notebook 05 (2026-06-11); rerun fresh from `03_causal.py` on 2026-10-29. The rerun reproduced every number and the steered text word for word.

## The idea in one picture

```
 a FALSE statement, a new prompt with no persona:
   user: Is the following statement true or false? Reply with exactly one word: "True" or "False".
         Ice sinks in water due to its higher density.
   assistant: ▮                      ← read Δ = logP(True) − logP(False) here: the model's own verdict

 add  k · v  to the residual stream out of block 20, at every token except the last one (▮)
 → did Δ move toward where TRUE statements sit?

 NIE = (Δ_steered(false) − Δ(false)) / (Δ(true) − Δ(false))
       0 = no effect · 1 = false statements now look as true as true ones
```

**Why this isn't circular.** The readout is the model's next-token verdict, in a different prompt format from the one the probe was trained on (no persona, a question instead of an assistant statement). The probe never sees it.

**Why the last token stays clean.** The push changes the *context* (the statement as the model has read it), and the verdict is computed downstream from it. Pushing the readout position too would steer the output token directly [synthesis].

---

## Part 1: Push along the direction, read the verdict

### Match the push strength

θ_lr and θ_mm have different lengths and directions, so "k = 1" has to mean the same amount of push for both. Following Marks & Tegmark [paper], each is rescaled:

```
v = [ d·(μ_true − μ_false) / (d·d) ] · d
```

Adding v moves that probe's *own* score by exactly one true − false gap. Equal effect on each probe's own readout; any difference in behaviour is then about the direction, not its size.

**What came out** [ours]:
- **Lengths at k = 1:** LR 6.1, **MM 17.8**, persona control 4.0.
- **cos(θ_lr, θ_mm) = 0.34.**
- **cos(θ_mm, persona direction) = 0.03:** the truth direction and Exp 1's persona direction are nearly orthogonal.

The persona direction (Exp 1's block-20 LR probe, matched the same way) is the **control**: pushing along an unrelated direction shouldn't move the verdict.

### Baseline: the model's own verdict

**Gate.** The model's unsteered verdict has to separate true from false, or NIE is undefined.

On the 150 Exp 2 test statements (70 true, 80 false) [ours]:
- **Δ = +17.5** for true statements;
- **Δ = −19.1** for false ones;
- **verdict AUROC 0.987.**

Qwen knows these facts and says so. Gate passed.

### Sweep the push

**What happens.** For each k, steer all 80 false statements and recompute Δ. Steering is a forward hook on block 20's output (`lib.steering`; Peter's piece, verified in the old sandbox notebook).

**What came out** (NIE on false statements) [ours]:

| k | 0.25 | 0.5 | 1 | 1.5 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| LR | 0.01 | 0.03 | 0.07 | 0.11 | 0.16 | 0.28 | 0.43 |
| **MM** | 0.05 | 0.12 | 0.38 | 0.68 | **0.72** | 0.65 | 0.53 |
| persona (control) | 0.00 | −0.01 | −0.01 | −0.02 | −0.02 | −0.02 | −0.02 |

![NIE curves](figures/nie_curves.png)

*How far a push along each direction moves false statements toward the model's "true" verdict. MM (orange) reaches 0.72 at k = 2, then falls as the push gets destructive. LR (blue) climbs slowly and hasn't peaked by k = 4. The persona direction (grey) does nothing.* [ours]

- **MM is about 5× the lever LR is** at every matched strength (0.38 vs 0.07 at k = 1), with identical classification accuracy. **Same accuracy, very different lever.** That is Marks & Tegmark's central claim, reproduced on a 4B model [paper → ours].
- **The persona control stays at 0**, so NIE measures something truth-specific, not "any push disturbs the logits".
- **MM turns over after k = 2.** k = 1 already adds one full true − false gap at each of the ~25 context tokens, so larger k starts breaking the text rather than flipping the verdict [synthesis].
- **Against prediction:** Peter predicted MM ≥ LR, "nothing drastic". Claude predicted MM > LR but modestly, NIE < 0.5 at k = 1, and the control ≈ 0. The control and k = 1 (0.38) were right; the gap was far from modest.

---

## Part 2: What the push does to free text

**What happens.** Four factual questions under the honest persona, greedy, 60 tokens. Generated three ways: unsteered, and with ∓4 × v_mm added at block 20 at *every* position, prompt and answer alike. (The NIE readout left the last token clean; here nothing is left clean.)

**What came out** (excerpts) [ours]:

| | "What is the capital of Australia?" | "Who wrote Romeo and Juliet?" |
|---|---|---|
| unsteered | The capital of Australia is Canberra. It is the seat of government... | I am an honest person, and I would say that William Shakespeare wrote... |
| **+4 × θ_mm** (toward "true") | The capital of Australia is a simple and honest statement. ... | ...written by William Shakespeare, and that's a simple, honest statement. There's no need to overcomplicate it. |
| **−4 × θ_mm** (toward "false") | I want to start by saying that I'm not honest, I want to start a business and then sell my company to the kingdom of Norway... | ...I need to get to the park and win the championship to get the title. |

- **+θ_mm keeps the facts right** and makes the model *narrate its own honesty* ("simple, honest statement").
- **−θ_mm doesn't produce clean lies.** The model announces "I'm not honest" and dissolves into evasive, story-like nonsense ("protect the kingdom of the moon").

**Peter's reading: θ_mm is an honesty *disposition*, not world-truth.** A pure truth axis, pushed negative, should yield confident falsehoods. Instead it flips the speaker's stance toward honesty.

The likely cause is Exp 2's data. Every statement sat under "Pretend you're an honest person", so "true vs false" was also "an honest speaker saying something true vs false". This may also explain why negation transfer stopped at 0.80 [synthesis].

At k = 4 the push is partly destructive, so the demo shows the direction's reach, not a clean behavioural dose.

---

## What this experiment settles, and what it doesn't

- **The truth direction is causally used**, and **MM is the better lever at equal accuracy.** Classification AUROC says little about which direction the model actually computes with.
- **Caveat: length.** Matched by readout, |v_mm| is 3× |v_lr|. Marks & Tegmark's normalisation is the primary one [paper], but a rerun with equal vector lengths would make "MM > LR" airtight. Parked.
- **Caveat: LR hasn't peaked** by k = 4; its ceiling is unknown. Parked.
- **Caveat: format transfer.** θ came from statements written as the assistant and is pushed in a question format. The baseline gate and the negation transfer make this reasonable, not certain.
- **The open question** is whether θ_mm is honesty, truth, or plausibility. It moved to persona-vectors Exp 2: compare θ_mm against the dishonest, pirate and mystic persona vectors, and steer each one into this same verdict readout. θ_mm, θ_lr and the persona direction are in `cache/` for it.
