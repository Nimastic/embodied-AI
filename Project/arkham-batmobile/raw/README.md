# Raw source material

The original unedited dump this project grew out of. Kept for provenance. Nothing here is
authoritative — the live documents are `../PRD.md`, `../MILESTONES.md` and `../hardware/`.

| File | What it is |
|---|---|
| `prd-original.md` | The first PRD, generated in a chat session. Specs a 1:8 scale, 15–25 km/h brushless RC replica, plus a modular CAD/3D-print plan for a 75 cm body. Superseded by `../PRD.md`; the print plan survives, rescaled, in `../hardware/build-notes.md` § Body shell. |
| `notes-dump.txt` | Mixed chat transcript: edge computing versus fog computing, edge AI, Jetson versus Raspberry Pi, TinyML course links, an AI-market-cycle tangent, and a sketch of a person-detecting projectile robot. |
| `images/concept-front-quarter.png` | Generated concept render. Mood reference only — not dimensionally consistent. |
| `images/concept-rear-quarter.png` | Generated concept render. Same caveat. |

## What was removed in the cleanup

- `PRD 2` — byte-identical duplicate of `PRD.md`.
- Two `... 2.png` files — byte-identical duplicates of the two concept renders.

## What was deliberately not carried forward

- **The projectile-launcher-plus-person-detector design.** The YOLO snippet in `notes-dump.txt`
  detects people so the robot can drive at them and fire. Not built. See
  `../hardware/build-notes.md` § Deliberately not built for the rules if a launcher is ever added
  as a prop.
- **The edge-versus-fog computing material.** Accurate but not load-bearing for this project; the
  compute decision and its reasoning live in `../hardware/BOM.md` § The compute decision.
- **The AI market-cycle and value-migration section.** Unrelated to the build.
- **The TinyML and Jetson course links.** The build-first rule in the root `AGENTS.md` puts
  prerequisite coursework behind the artifact. If a specific blocker makes one of them necessary,
  it goes in `notes/just-in-time/` at the root, attached to the experiment it unblocks.
