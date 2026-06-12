# Contributing

This is the code accompanying a research paper, so the one firm rule is simple: any change must keep the frozen results in `Models/` reproducible — `reproduce_models.m` must still pass. For anything non-trivial, open an issue first.

**Adding a CV strategy.** Write a `CV_<Name>.m` partitioner, register it in the `switch` in `getExperimentConfig.m`, add a cell in `Run_pipeline.m`, and a row in the README's experiment table. The `Discovery` / `Tuning` / `Production` drivers need no changes — they read everything from `getExperimentConfig.m`.

**Seeds.** If you add an `rng(...)` or `RandStream(...)` call, record it in [`Docs/REPRODUCIBILITY.md`](Docs/REPRODUCIBILITY.md) in the same change, and leave any `% LOCKED seed` line untouched.

**Environment.** Install via `environment.yml` (conda — pins Python 3.9 and the LightGBM/Optuna stack) or `requirements.txt` (pip fallback).

**Problems reproducing the paper?** Open a GitHub issue.
