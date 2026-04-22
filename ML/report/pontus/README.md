# Pontus workspace

Pontus's personal working space under the ML report directory. Everything here is his own experimentation: modelling scripts, outputs, notes, handovers. Kept separate from the shared pipeline scripts (`../scripts/`) so iteration is fast without stepping on Alex's work.

## ⚠️ FRAMEWORK CONSTRAINT — TensorFlow / Keras only

**The CBS MLDP exam requires TensorFlow / Keras for all neural-network code. PyTorch is NOT permitted.** Any MLP or autoencoder in `pontus/scripts/` must use `tensorflow` + `tf.keras`. sklearn models (LogReg, RF, GBM, Isolation Forest) remain fine as baselines.

Note: the old `scripts/12_train_mlp.py` was a PyTorch scaffold and is deprecated (raises `SystemExit` on run); treat it as historical.

## Current focus (2026-04-22)

Full §5 pipeline end-to-end on the finalised leakage-free cohort parquets. Deliverables:
- MLP (tf.keras) with isotonic calibration on val.
- sklearn baselines: LogReg (L2), Random Forest (heavily regularised).
- Autoencoder (tf.keras) + Isolation Forest for the unsupervised arm.
- Residual-edge analysis (RQ1b proper test, per `alex/notes/session-learnings-2026-04-22.md`).
- Trading-rule backtest: general +EV (`edge > 0.02`) and home-run (`edge > 0.20` AND `time_to_deadline < 6h` AND `market_implied_prob < 0.30`).
- Magamyman sanity check as Discussion anchor.

See `pontus_adventure.md` §5 for the modelling spec and §5.7 for the four-layer validation strategy.

## Layout

```
pontus/
├── scripts/                 # Pontus's own modelling / analysis scripts
│   └── 21_full_pipeline.py  # MLP + baselines + autoencoder + residual-edge + backtest
├── outputs/                 # metrics.json, plots, run artefacts
├── notes/                   # free-form working notes + Alex's orientation brief
│   └── FYI-from-alex.md     # orientation note Alex wrote for this workspace
├── handovers/               # per-session handovers
│   └── handover_21_apr.md   # evening 21 Apr handover
├── pontus_adventure.md      # modelling plan (read before code)
└── README.md                # this file
```

## Shared data (read-only from here)

Cohort parquets live under `../../data/experiments/` and are maintained by the shared slicer (`../../scripts/14_build_experiment_splits.py`). Access them via relative paths from `pontus/scripts/`:

| Cohort | Path | Rows | Markets |
|---|---|---:|---|
| Train | `../../data/experiments/train.parquet` | 202,082 | Feb 25-28 strike (1 YES + 3 NO) |
| Val | `../../data/experiments/val.parquet` | 13,154 | Mar 15 conflict-end (NO) |
| Test | `../../data/experiments/test.parquet` | 13,414 | 7 ceasefires Apr 8-18 (all NO) |

If the cohorts need regenerating, run `python scripts/14_build_experiment_splits.py` from `ML/report/`.

The consolidated CSV is at `../data/03_consolidated_dataset.csv` (54 cols after the 22-Apr physical-drop finalisation). Regression guard at `../../scripts/_check_causal.py` — run after any rebuild.

## Working rules

- **Don't modify `../scripts/` from here.** Those are shared pipeline scripts; fixes go via PR on the repo.
- **Cite `../data-pipeline-issues.md`** when assumptions about the data matter.
- **Commit often** — this folder is git-tracked alongside the rest of the repo.
- **TF / Keras only** for neural networks. Sklearn for baselines.
