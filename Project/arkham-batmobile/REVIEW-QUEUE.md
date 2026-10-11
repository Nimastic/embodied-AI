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
3. **Arena space.** Everything assumes a 2 × 2 m patch of hard floor you can tape and leave
   undisturbed for weeks. Do you have that?

## Status log

| Date | What landed |
|---|---|
| 2026-10-10 | Folder cleanup, PRD rewrite, BOM, milestones. Phase 0 simulator built and characterised. |
| 2026-10-11 | M2 state predictor trained. 1.40 mm vs 5.27 mm calibrated on hard floor; no transfer to unseen surface. |
| 2026-10-11 | M2b: **hypothesis rejected.** History does not help — the surface is only 69% identifiable from proprioception even with 4 s of it. Redirects toward vision. |
| 2026-10-11 | Simulator fix: `hard` and `mat` now look different, not just grip different. Any frame dataset from before today is stale. |
