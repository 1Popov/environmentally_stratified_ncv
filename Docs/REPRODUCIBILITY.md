# REPRODUCIBILITY.md — RNG sites on the pipeline runtime path

This document inventories every site that draws a pseudo-random number on the active runtime path (Discovery → Tuning → Production and the Python wrapper). Each site is documented with the call, the purpose, the seed, and whether the seed is intentionally decoupled from `config.RandomSeed`.

The intent is so that a future maintainer reading the code does not silently "normalize" a hard-coded seed value to `config.RandomSeed` and break the original fold assignments. Every decoupling here is deliberate.

---

## 1. Top-level

| File | Line | Call | Purpose |
|---|---|---|---|
| `Discovery.m` | 70 | `rng(config.RandomSeed, 'twister')` | Seeds the MATLAB global stream at the start of every Discovery run. Consumed by the `randperm` calls below and by the parallel-pool worker streams. |
| `Discovery.m` (test-mode branch) | inside the `if testMode && SubsetSize < 1` block | `randperm(height(devSet), floor(SubsetSize*height(devSet)))` | Subsets the development set to 20% in smoke-test mode (`SubsetSize = 0.20`, in the `isTestMode()` block of `getExperimentConfig.m`). The full-budget branch keeps `SubsetSize = 1.0`, so it does not fire on the paper runs. Consumes the global stream. |
| `Discovery.m` (always) | inside the dev-set shuffle block | `randperm(height(devSet))` | One-pass row shuffle of the cleaned dev set before partitioning. Consumes the global stream. |

`config.RandomSeed` is set to `42` in `getExperimentConfig.m` and is the only top-level knob the user should ever change. All other seeds below are **decoupled from this** and locked to their original values.

---

## 2. CV partitioners

### 2.1 Shared helpers (called by the three environmental partitioners)

| File | Line | Call | Purpose | Decoupled? |
|---|---|---|---|---|
| `buildAtomicBlocks.m` | 33 | `rng(42, 'twister')` | Atomic-block k-means on cosine-weighted `(LAT, LON)`. Produces `M = 750` atomic blocks. | **Yes — locked to seed 42.** Changing this would alter every paper number that depends on environmental fold assignment. |
| `assignBlocksToFolds.m` | 30 | `RandStream('mlfg6331_64', 'Seed', 42)` | Within-stratum shuffled round-robin (deterministic, on a local stream so it does not advance the global generator). | **Yes — locked to seed 42 on the `mlfg6331_64` generator family.** |

### 2.2 Per-partitioner stratum k-means

| File | Line | Call | Purpose | Decoupled? |
|---|---|---|---|---|
| `CV_OceanFingerprint.m` | 65 | `rng(42, 'twister')` | Stratum k-means on the 5-D scalar fingerprint after the shared atomic-block step. | **Yes — locked to seed 42.** |
| `CV_EOF.m` | 122 | `rng(42, 'twister')` | Stratum k-means on the 5-D unscaled PC-score fingerprint. | **Yes — locked to seed 42.** |
| `CV_Hybrid.m` | 143 | `rng(42, 'twister')` | Stratum k-means on the 10-D hybrid fingerprint (PCs + λ-scaled scalar indices). | **Yes — locked to seed 42.** |

### 2.3 Geographic baseline

| File | Line | Call | Purpose | Decoupled? |
|---|---|---|---|---|
| `CV_GeoCluster.m` | 23 | `rng(0, 'mlfg6331_64')` | Sets the global stream before the geographic k-means. Uses seed **`0`** and the **`mlfg6331_64`** generator family — both different from every other site here. This is intentional and matches the substream selected by the kmeans `'Options'` argument below. | **Yes — locked to seed 0 + `mlfg6331_64`.** Changing either would alter every paper number for the GeoCluster experiment. |

### 2.4 Random K-fold partition

| File | Line | Call | Purpose | Decoupled? |
|---|---|---|---|---|
| `CV_RandomKFold.m` | 23 | `cvpartition(h, "KFold", nFolds)` | Random K-fold partitioner for the two Random-CV experiments (outer- and inner-CV). Unlike the stratified k-means sites it carries **no local seed**; the outer partition draws from the global stream seeded at `Discovery.m:70`. | No local `rng`/`RandStream` — inherits the ambient stream. |

---

## 3. Python side

`lgbm_wrapper.py` makes LightGBM and Optuna deterministic via the following:

| Where | Setting | Effect |
|---|---|---|
| `_FIXED_GPU_PARAMS` | `'deterministic': True` | LightGBM disables non-deterministic histogram-bin tie-breaking. |
| `_FIXED_GPU_PARAMS` | `'gpu_use_dp': True` | LightGBM GPU reductions use float64 atomic-adds (eliminates the non-deterministic float32 atomic-add path). |
| `_FIXED_GPU_PARAMS` | `'max_bin': 255` | Fixed histogram bin count. |
| `_FIXED_GPU_PARAMS` | `'feature_pre_filter': False` | Disables LightGBM's pre-trial feature pruning — kept off across both the inner-CV trial path and the final-model path so the bin layouts agree. |
| `run_bayes_opt_in_python` | `TPESampler(seed=int(random_seed))` | Optuna's TPE sampler is deterministic for a fixed seed and trial budget. |
| `run_bayes_opt_in_python` | `params['seed'] = int(random_seed)` | Each LightGBM training run inside `lgb.cv` and `lgb.train` re-seeds the booster. |

`random_seed` here is the same `config.RandomSeed = 42` that seeds the MATLAB side; it crosses the `py.*` bridge as `int32`.

---

## 4. Summary

Two seed pairs are in use across the codebase:

- **Seed `42` + `'twister'` generator** — the canonical "MATLAB default" pair used by the global stream, by the environmental k-means sites, and (transitively, via the `random_seed` argument) by Optuna and LightGBM.
- **Seed `0` + `'mlfg6331_64'` generator** — used only by `CV_GeoCluster.m`. Both the seed and the generator family are different from every other site, and both are paper-locked.

A separate `RandStream('mlfg6331_64', 'Seed', 42)` is created locally in `assignBlocksToFolds.m` so the within-stratum shuffle is decoupled from the global generator entirely.

**Never normalize the locked seeds to `config.RandomSeed`.** Doing so would invalidate fold assignment for the original runs and break Path-B numerical reproducibility.
