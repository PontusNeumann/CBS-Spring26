# Alex workspace

Alex's personal working space under the ML report directory. Everything here is his own experimentation: modelling scripts, outputs, notes. Kept separate from the shared pipeline scripts (`../scripts/`) so we can iterate quickly without stepping on Pontus's work.

## ⚠️ FRAMEWORK CONSTRAINT — TensorFlow / Keras only

**The CBS MLDP exam requires TensorFlow / Keras for all neural-network code. PyTorch is NOT permitted.** Any MLP, autoencoder, or deep-learning model in `alex/scripts/` must use `tensorflow` + `tf.keras`. sklearn models (LogReg, RF, GBM, etc.) remain fine as baselines.

Note: Pontus's `scripts/12_train_mlp.py` currently uses PyTorch — that script cannot be used for the exam submission as-is.

## Current focus (2026-04-22)

Building a clean train → val → test MLP pipeline from scratch on the leakage-free cohort parquets. Goal is a defensible answer to RQ1 (does a model trained on the peak-informed-flow strike window generalise to ceasefire markets?) and, if time permits, RQ2 (do the model's largest `|p_hat − market_implied_prob|` gaps concentrate on features documented in the informed-trading literature?).

Working style in this workspace: **Alex writes the code, Claude sparring-partners.** Claude's role is to challenge assumptions, point out gaps, review diffs, and catch bugs — not to generate scripts. Switch modes explicitly if you want something drafted.

### Open design decisions

- **Sub-markets in one training file vs separate.** Current: one combined `train.parquet` with all 4 strike sub-markets. Rationale: more data, cross-market generalisation is RQ1 itself, per-market inference still works post-training. To falsify: after training, score each of the 4 training markets separately and compare `p_hat` distributions. If wildly different, the model is market-specific and this choice hurts.
- **RQ2 reframing.** Plan §3 keeps informed-trading parallel as Discussion-only. Pitch was to elevate to co-primary (same model, feature-importance + Magamyman sanity check = results, not anecdote). Decide before the Results section is written.

## Layout

```
alex/
├── scripts/    # Alex's own modelling / analysis scripts
├── outputs/    # Results — metrics.json, plots, run artefacts
├── notes/      # Free-form working notes, design-decision drafts
└── README.md   # This file
```

## Shared data (read-only from here)

Training cohorts live under `../data/experiments/` and are maintained by the shared slicer (`../scripts/14_build_experiment_splits.py`). Access them via relative paths from `alex/scripts/`:

| Cohort | Path | Rows | Markets |
|---|---|---:|---|
| Train | `../../data/experiments/train.parquet` | 202,082 | Feb 25, 26, 27, 28 strike |
| Val | `../../data/experiments/val.parquet` | 38,158 | Jan 23 strike (NO) |
| Test | `../../data/experiments/test.parquet` | 692 | Apr 18 ceasefire-extended (NO) |

If the cohorts need regenerating, run `python ../scripts/14_build_experiment_splits.py` from `ML/report/`.

## Working rules

- **Don't modify `../scripts/` from here.** Those are shared pipeline scripts that Pontus and others depend on. Any fix to the shared pipeline goes via a PR on the `CBS-Spring26` repo.
- **Cite `../data-pipeline-issues.md`** when assumptions about the data matter. That file tracks known issues + mitigations.
- **Commit often** — this folder is git-tracked alongside the rest of the repo, so small commits are fine.
