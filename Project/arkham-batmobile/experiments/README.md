# Experiments

One folder per experiment, named `M<milestone>-<slug>`, e.g. `M2-residual-vs-direct`.

```text
experiments/
└── M2-residual-vs-direct/
    ├── README.md        what was asked, what was done, what happened, what broke
    ├── metrics.json     every number quoted in the README
    ├── config.json      hyperparameters, dataset episodes used, git commit
    └── assets/          plots, filmstrips, GIFs, clips
```

Mirrors the structure the completed roadmap stages already use
(`stages/00-sim-so101/assets/metrics.json`), so the same tooling and habits carry over.

## Each experiment README answers five questions

1. **Question.** One sentence, falsifiable. "Does residual-on-analytic prediction beat direct
   prediction at a 1 s horizon on held-out surfaces?" — not "try the residual model".
2. **Setup.** Dataset episodes, train/held-out split, model, hyperparameters, hardware.
3. **Result.** The number, with a baseline beside it and a spread. Mean alone hides the failures
   that matter.
4. **Visual.** Plot, filmstrip or clip. A result with no picture does not close a milestone.
5. **Failure analysis.** Where it breaks and the current best guess as to why. This section is the
   reason the folder exists; a clean result with no failure analysis means the evaluation was not
   hard enough.

## Rules

- **Baseline in every table.** Analytic FK is always a column. A learned model with no baseline is
  not a result.
- **Held-out means held-out.** Split by episode, and keep a whole surface unseen for M6. Splitting
  by random tick leaks neighbouring frames across the split and inflates everything.
- **Negative results close milestones.** If the learned model loses to physics homework, that is
  the finding — write up where and why. Per `ROADMAP.md`, "mostly no, and here are six reasons
  why" counts.
- **Numbers come from `metrics.json`.** If a number is in the README but not the JSON, it is not
  reproducible and should not be quoted.
- **Record what you expected first.** Writing the predicted outcome into the README before running
  makes it much harder to retrofit a story onto whatever happened.
