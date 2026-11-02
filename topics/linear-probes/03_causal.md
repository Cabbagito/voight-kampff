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

- **By k, MM is 4–6× the lever LR is** up to k = 2 (0.38 vs 0.07 at k = 1), with identical classification accuracy. That is Marks & Tegmark's central claim, reproduced on a 4B model [paper → ours]. LR hasn't peaked by k = 4 (0.43).
- **The persona control stays at 0** over this range.
- **But k = 1 is not the same push for both.** The matching makes |v_mm| 17.8 and |v_lr| 6.1, so at every k MM moves the activations 3× further. Next step separates direction from size.

### Match by push size instead

**What happens.** Same NIE readout, but each direction is pushed to the same *vector length* |k·v|: 5 to 70, so LR and the control run well past where the k sweep stopped them (LR to k ≈ 11.5). Added 2026-11-02, after the README review flagged the 3× length difference.

**What came out** (NIE on false statements) [ours]:

| push length | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LR | 0.05 | 0.12 | 0.21 | 0.32 | 0.44 | 0.55 | 0.62 | **0.64** | 0.59 | 0.53 | 0.49 |
| MM | 0.06 | 0.13 | 0.26 | 0.48 | 0.64 | 0.71 | **0.72** | 0.71 | 0.67 | 0.60 | 0.54 |
| persona (control) | −0.01 | −0.02 | −0.02 | −0.01 | 0.00 | 0.02 | 0.03 | 0.05 | 0.10 | 0.17 | 0.25 |

![NIE curves](figures/nie_curves.png)

*Left: by k, the paper's matching. Right: the same directions by actual push length, LR and the control extended to 70. Grey: the persona direction (control).* [ours]

- **At equal push size, LR and MM are nearly the same lever.** MM leads by at most 1.5× on the way up (0.48 vs 0.32 at length 20), and the peaks are close: MM 0.72 at 35, LR 0.64 at 40. Most of the 4–6× gap by k was push size, not direction.
- **Both turn over at about the same size** (35–40). So the turnover is about how hard the activations are pushed, not about either direction: past that, the push breaks the text rather than flipping the verdict [synthesis].
- **The control holds until the pushes get very large.** Flat to length 25, then it creeps up (0.10 at 50, 0.25 at 70). At those sizes any push starts moving the verdict, so the right panel's tail is part truth effect, part disruption [synthesis].
- **Against prediction:** Peter predicted MM ≥ LR, "nothing drastic". Claude predicted MM > LR but modestly, NIE < 0.5 at k = 1, and the control ≈ 0. By k, the gap was far from modest. By push size, Peter's "nothing drastic" was right.

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

**Update (2026-10-31, persona-vectors Exp 2):** this reading didn't hold up. Without the honest-persona prompt and at 0.1–0.6 × the stream, −θ_mm gives no honesty talk at all, only confident factual mistakes. In the verdict readout it makes true statements look false rather than inverting them like a liar, which first read as "belief". A later check (2026-11-02) found prompted liars look the same on that readout, so belief vs honesty is open [ours, `../persona-vectors/02_what_is_theta.md` step 5].

The suspected cause was Exp 2's data: every statement sat under "Pretend you're an honest person", so "true vs false" was also "an honest speaker saying something true vs false". Rebuilt without that line, θ_mm is cos 0.90 with the original and pushes the verdict the same way [ours, persona-vectors Exp 2 step 5], so the training prompt is at most a small part. The likelier cause is the honest persona in this Part's *generation* prompt, which the push clashes with [synthesis].

At k = 4 the push is partly destructive, so the demo shows the direction's reach, not a clean behavioural dose.

---

## What this experiment settles, and what it doesn't

- **The truth direction is causally used**: both truth directions move the verdict, and the unrelated control doesn't until the pushes get very large.
- **"MM is the better lever" depends on how push size is measured.** By the paper's k, MM is 4–6× LR. By actual vector length, MM is only slightly ahead (peak 0.72 vs 0.64). So on this model, two probes 70° apart with the same accuracy are nearly the same lever. Neither normalisation is the "true" one: k treats each probe's own score as the yardstick, and length treats every direction in the residual stream as equal, which it isn't [synthesis].
- **Caveat: format transfer.** θ came from statements written as the assistant and is pushed in a question format. The baseline gate and the negation transfer make this reasonable, not certain.
- **The open question** is whether θ_mm is honesty, truth, or plausibility. It moved to persona-vectors Exp 2: compare θ_mm against the dishonest, pirate and mystic persona vectors, and steer each one into this same verdict readout. θ_mm, θ_lr and the persona direction are in `cache/` for it.
