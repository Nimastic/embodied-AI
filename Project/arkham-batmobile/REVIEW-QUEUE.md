# Review queue

Things waiting on Jeriel. Nothing here is urgent; it is a reading list in priority order so the
project does not drift while you are busy elsewhere.

**Everything below is uncommitted**, sitting alongside your earlier `roadmap/` → `stages/`
restructure. Say the word and it gets committed.

---

## Read these three first (~15 min total)

| | What | Why it needs you |
|---|---|---|
| 1 | [experiments/M2-state-predictor/README.md](experiments/M2-state-predictor/README.md) | The headline result, and the baseline bug that nearly faked it. If you only read one thing, this. |
| 1b | [experiments/M2b-history-conditioning/README.md](experiments/M2b-history-conditioning/README.md) | My proposed fix failed. Includes why, and why I should have predicted it. Short. |
| 1c | [experiments/M4-world-model-planning/README.md](experiments/M4-world-model-planning/README.md) | The payoff experiment. Planning works, but most of the benefit comes from four fitted scalars rather than the neural network — and why. Also a cost-function bug that nearly became a false finding. |
| 1d | [experiments/M5-dagger/README.md](experiments/M5-dagger/README.md) | **The most important result so far.** DAgger closes the accuracy gap 4.9× and buys zero task success, and an oracle proves a *perfect* model would too. Overturns M4's explanation. |
| 1e | [experiments/M3a-surface-from-pixels/README.md](experiments/M3a-surface-from-pixels/README.md) | Vision fixes M2's transfer failure (16.78 → 1.98 mm). Read the caveats — a one-line classifier gets 95%. |
| 2 | [hardware/BOM.md](hardware/BOM.md) | **Spends your money.** ~$450 for the learning robot. I chose Pi 5 over Jetson and slow encoder motors over brushless — both are reversible now and expensive later. |
| 3 | [PRD.md](PRD.md) § 3 | The eight success criteria. If you disagree with what "done" means, now is the cheap time to say so. |

## Decisions I made that you might want to overturn

| Decision | Where | Reversible? |
|---|---|---|
| Pi 5 + Pico, not Jetson Orin Nano | [hardware/BOM.md](hardware/BOM.md) § compute | Yes — Jetson is a later add-on, nothing wasted |
| Geared DC with encoders, not brushless; ~1:12 scale, not 1:8 | [PRD.md](PRD.md) § 1 | Yes, until you order |
| Foam body v1 before a printed v2 | [hardware/build-notes.md](hardware/build-notes.md) § Body shell | Yes |
| No autonomous targeting, ever; launcher is a manual-trigger prop | [README.md](README.md) § Scope boundaries | This one I will not reverse |
| Batmobile as a parallel track, `ROADMAP.md` untouched | [../../README.md](../../README.md) | Yes — you approved this one already |

## Background, read when curious

- [notes/just-in-time/mecanum-kinematics.md](../../notes/just-in-time/mecanum-kinematics.md) — the derivation, and section 7 on where it stops being true
- [software/sim/README.md](software/sim/README.md) — how to run everything
- [software/README.md](software/README.md) — the dataset schema, including the one alignment that silently ruins world models
- [experiments/P0-sim-baseline/README.md](experiments/P0-sim-baseline/README.md) — simulator characterisation, three bugs caught
- [MILESTONES.md](MILESTONES.md) — M0–M6

## Open questions I could not answer for you

1. **GPU for M3 — now more urgent.** M2b makes vision the leading candidate for surface
   adaptation, which promotes M3 from "later" to "next after one cheap test". This machine has
   `torch 2.8.0+cpu`. Colab, or do you have a CUDA machine?
2. **Where are you buying?** The BOM assumes Singapore sourcing (Cytron / SGBotic / AliExpress) —
   inferred from your NUS modules, so correct me if wrong.
3. ~~**Arena space.**~~ Answered: a 2 × 2 m patch of *floor* (not wall) for the robot to drive in,
   with a camera above it. Not needed until hardware exists, and it does not block a parts order —
   I overstated this. It only affects the camera mount design.

## Status log

| Date | What landed |
|---|---|
| 2026-10-10 | Folder cleanup, PRD rewrite, BOM, milestones. Phase 0 simulator built and characterised. |
| 2026-10-11 | M2 state predictor trained. 1.40 mm vs 5.27 mm calibrated on hard floor; no transfer to unseen surface. |
| 2026-10-11 | M2b: **hypothesis rejected.** History does not help — the surface is only 69% identifiable from proprioception even with 4 s of it. Redirects toward vision. |
| 2026-10-11 | Simulator fix: `hard` and `mat` now look different, not just grip different. Any frame dataset from before today is stale. |
| 2026-10-13 | M4 transfer sweep on `mat`: ordering holds on an unseen surface, ~10 point degradation, nothing collapses. Learned-vs-calibrated gap within noise on both surfaces. |
| 2026-10-14 | M5 DAgger: accuracy gap closed 4.9×, task success unchanged. Oracle shows a perfect model scores the same — no headroom existed. Corrects M4's explanation. |
| 2026-10-14 | M3a vision: 100% surface ID from one frame (but a one-line scalar gets 95.1%); conditioning fixes M2's transfer failure, 16.78 → 1.98 mm. |
| 2026-10-14 | Third divergence-indexing bug fixed in run_m4.py; M4's diagnostic column corrected, success rates unaffected. |
| 2026-10-13 | 3D design spec written, then **parked as iteration 2** — battle mode, decisions settled, starts when M4 passes on hardware. Body stays foam v1 until then. |
| 2026-10-11 | M4 MPPI planning. Replan rate dominates model quality; calibration captures most of the gain; learned model degrades 83% -> 35% advantage under the planner's own action distribution. Plans in 40 ms on CPU, so Pi-feasible. |
