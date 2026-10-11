# M2b — Does a short history let the model identify the surface?

**Date:** 2026-10-11 · **Status:** complete — **hypothesis rejected** · **Milestone:** M2 follow-up

## Question

M2 found that a learned model beats the calibrated analytic baseline by 73% in-distribution and
by ~1% on a surface it never trained on. The proposed explanation was that the model sees only a
single instant, and slip is identifiable from the *mismatch* between commanded and achieved
velocity — which needs more than one tick to observe.

**Hypothesis:** feeding K ticks of `(action, achieved velocity)` history lets the model infer the
surface and close the transfer gap.

Expected: improvement on `mat` growing with K, little change on `hard`.

## Setup

Identical to M2, with one change: the state encoder is a GRU over the last K ticks of
`[action, achieved vel, wheel speeds, gyro, batt]` (14 features/step) rather than an MLP on a
single instant.

The encoder is a GRU for **every** K including K=1, so the ablation varies only how much history
is fed — same architecture, same parameter count. Otherwise it would confound "more history" with
"different model".

Trained on `hard` only, 25 epochs. Reproduce:
`python software/models/state_model.py --epochs 25 --stride 2 --history K`

## Result

Median 1.0 s pose error, residual variant:

| History | ticks | hard (held out) | mat (never seen) | vs calibrated on mat |
|---|---|---:|---:|---:|
| K = 1 | 0.05 s | 1.38 mm | 17.11 mm | 1.1% better |
| K = 5 | 0.25 s | 1.34 mm | 16.92 mm | 2.2% better |
| K = 20 | 1.00 s | 1.36 mm | 16.96 mm | 1.5% better |
| *calibrated analytic* | — | *5.27 mm* | *17.30 mm* | — |

**No effect.** All three are within noise of each other on both splits. The hypothesis is
rejected.

Incidental robustness check: K=1 here uses a GRU encoder where M2 used an MLP, and reproduces
M2's number to within 0.02 mm (1.38 vs 1.40 mm). Two different encoders, same answer — the M2
result is not an artefact of that architecture choice.

## Why — the diagnostic that matters

A null result is only worth anything if you can distinguish "the information was absent" from
"the model failed to use it". So: how identifiable is the surface, in principle, from the
observations the model receives?

Contact efficiency `eta = |body_vel| / |FK(wheel_meas)|` is directly observable and is exactly
what differs between surfaces. Estimating it by least squares over a K-tick window and classifying
with the optimal threshold:

| K | window | d′ | classification accuracy |
|---|---|---:|---:|
| 1 | 0.05 s | 0.25 | 59.0% |
| 5 | 0.25 s | 0.46 | 59.3% |
| 10 | 0.50 s | 0.55 | 62.4% |
| 20 | 1.00 s | 0.64 | 64.9% |
| 40 | 2.00 s | 0.77 | 66.5% |
| 80 | 4.00 s | 0.96 | 69.0% |

**Even four seconds of history gives only 69% accuracy.** The information is present but weak, and
it grows more slowly than the √K that pure averaging would predict — the noise is correlated, not
independent.

The reason: between surfaces, mean `eta` differs by just 0.08 (0.795 hard vs 0.715 mat), while
tick-to-tick `eta` has a standard deviation of 0.31. That variance is dominated by **actuator-lag
transients**, which look identical on both floors. The robot is constantly accelerating under
random actions, and during a transient the achieved velocity lags the wheel speed for reasons that
have nothing to do with friction.

So the model did not fail. There was almost nothing to find.

## The tension worth noticing

The surface has a **large** effect on dynamics — it moves 1.0 s prediction error from 5.3 mm to
17.3 mm, a 3× difference — while being **hard to identify quickly**. Those two facts are not in
conflict, but together they rule out the obvious fix: you cannot adapt online to a property you
cannot measure online, no matter how good the model is.

## What this changes

**My proposed fix was wrong, and it is worth being precise about why.** I reasoned that because
slip is *physically* a mismatch between commanded and achieved motion, the mismatch must be
*statistically* informative. It is, barely. I did not check the effect size before proposing the
experiment, and should have — the diagnostic above took two minutes and would have predicted the
null result in advance.

Three routes remain, in order of how much I now believe in them:

1. **Use the camera.** Carpet and tile are trivially distinguishable by appearance — far easier
   than by dynamics. This is now a measured argument for vision being *necessary* rather than a
   fashionable addition, and it is the strongest justification for M3 so far.
2. **Train across surfaces (domain randomisation).** Give up on identifying the surface; make the
   model robust to the range instead. Cheap to test and does not need new hardware.
3. **Explicit online estimation over a long window.** Run a dedicated `eta` estimator over 4+
   seconds and feed it in. At 69% accuracy this is marginal, and it adapts too slowly to help when
   the robot crosses from tile onto a rug mid-run.

## Simulator limitation found and fixed

Both surfaces used the **same floor texture** — `hard` and `mat` differed only in physics. That
made route 1 untestable by construction, and it is unphysical: a real mat does not just grip
differently, it looks different.

Fixed in `mecanum_base.xml`: two floor materials, selected per surface at env init
(cool grey checker vs warm brown fine-weave; mean frame RGB `[68, 66, 73]` vs `[94, 75, 60]`).

This does not invalidate M2 or M2b — both are proprioception-only and never saw a pixel. It does
mean any dataset collected **with frames** before 2026-10-11 should be discarded.

## Next

Test route 2 first: it is a few lines, needs no new data, and establishes whether robustness is
achievable without vision. Then route 1, which is M3 proper.

Worth adding a third surface before either, so "train on two, test on a third" measures
interpolation rather than memorisation of a binary choice.
