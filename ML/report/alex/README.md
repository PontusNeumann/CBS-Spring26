# Alex's idea1 workspace

Self-contained ML pipeline for the CBS MLDP exam project. Predicts trade outcomes on Polymarket geopolitical event markets and tests cross-regime transfer (Iran-strike countdown markets → Iran-ceasefire countdown markets).

**Current state (v3.5 baseline, locked):** AUC 0.899 (RF), 100% top-1% precision, +23.7% headline ROI under N=1 realism. v4 pipeline (with Pontus's wallet-identity features added) is built and ready to run when his joined parquet lands. See [`notes/alex-approach.md`](notes/alex-approach.md) for the full pipeline summary.

**Research question (locked 2026-04-28, see D-001 + D-034):**
> Can a supervised model predict trade outcomes on Polymarket geopolitical event markets, and do the predictions transfer across event regimes?

The original "asymmetric-information detection" framing was retired after pressure tests showed top picks are 100% consensus-aligned and a 3-line naive consensus rule matches the model's top-1% precision (B1b finding).

---

## Read in this order

For a cold start, read these four canonical docs:

1. **[`notes/alex-approach.md`](notes/alex-approach.md)** — Pipeline summary written for Pontus to read cold (~10 min). Research question, cohort, features, models, results, sensitivity, comparison points.
2. **[`notes/design-decisions.md`](notes/design-decisions.md)** — 37 entries (D-001..D-037), append-only. Every design decision with alternatives + justification + status.
3. **[`notes/pressure-test.md`](notes/pressure-test.md)** — Verdict register for 18 assumptions tested. Headline findings, falsified claims, what we know holds.
4. **[`notes/data-manifest.md`](notes/data-manifest.md)** — Schema + checksum reference for the parquet files distributed via the data release.

For the active ML pipeline (the one we'll run on Pontus's v4 wallet-augmented data), see **[`v4_final_ml_pipeline/README.md`](v4_final_ml_pipeline/README.md)**.

Skim the rest of `notes/` only if you need historical context — see [`notes/README.md`](notes/README.md) for the index.

---

## Where the active pipeline lives

```
v4_final_ml_pipeline/        ← THE ACTIVE PIPELINE (run this)
├── README.md                  process + decision tree + invocation
├── pipeline.md                detailed per-stage runbook
└── scripts/
    ├── _common.py             shared utilities
    ├── 01_validate_schema.py  (Stage 1)
    ├── 02_causality_guard.py  (Stage 2)
    ├── 03_sweep.py            (Stage 4 — RF, HGBM, LightGBM, MLP, …)
    ├── 04_iso_forest.py       (Stage 5)
    ├── 05_optuna_tuning.py    (Stage 6)
    ├── 06_phase2_falsification.py (Stage 7.1)
    ├── 07_rigor_additions.py  (Stage 7.2 — skeleton)
    ├── 08_complexity_benchmark.py (Stage 7.3 — skeleton)
    ├── 09_shap_top_picks.py   (Stage 7.4 — skeleton)
    ├── 10_backtest.py         (Stage 8.1)
    ├── 11_realistic_backtest.py (Stage 8.2)
    └── 12_sensitivity_sweep.py (Stage 8.3)

scripts/                      ← HISTORICAL (v3.5 baseline that produced PR #13)
                                Don't run these for v4 work — see scripts/README.md
                                for the audit trail.
```

---

## Quick start (reproduce v3.5 headline numbers from the locked baseline)

```bash
# 1. Pull the v3.5 baseline data (373 MB extracted, gitignored)
gh release download alex-data-2026-04-28 -p '*.tar.gz' --repo PontusNeumann/CBS-Spring26
tar xzf alex-idea1-data-2026-04-28.tar.gz -C ML/report/alex/

# 2. From ML/report/, with venv at .venv:
.venv/bin/python alex/scripts/10_backtest.py            # economic backtest (cached preds, ~30s)
.venv/bin/python alex/scripts/11_realistic_backtest.py  # capital-aware backtest
.venv/bin/python alex/scripts/12_sensitivity_sweep.py   # cost_floor × copycats grid
.venv/bin/python alex/scripts/pressure_tests/phase2_falsification.py  # B1a/B1b decomposition
```

To force fresh model training (workers are deterministic, `random_state=42`): `RETRAIN=1 .venv/bin/python alex/scripts/10_backtest.py`.

For the v4 pipeline (new data), see [`v4_final_ml_pipeline/README.md`](v4_final_ml_pipeline/README.md).

---

## Layout

```
alex/
├── README.md                # This file (navigation entry point)
├── v4_final_ml_pipeline/    # ← ACTIVE pipeline (12 numbered scripts + _common + workers)
│   └── README.md            # process + decision tree + how-to-invoke
├── scripts/                 # HISTORICAL — v3.5 baseline scripts (PR #13)
│   └── README.md            # archive index
├── notes/                   # Markdown docs (canonical + reference + historical)
│   └── README.md            # status-tagged index
├── outputs/                 # Plots, metrics, JSON summaries (~5.6 MB tracked)
│   └── README.md            # what produced what
├── data/                    # Parquets (~373 MB, gitignored, see release alex-data-2026-04-28)
└── .scratch/                # Caches + cached predictions (~143 MB, gitignored)
```

---

## Hard constraints

- **TF/Keras only** for any neural-network code. PyTorch is forbidden by exam rules.
- **Stay in `alex/`** — don't modify `../scripts/` (Pontus's territory) or top-level repo files.
- **No external API calls** — HuggingFace `SII-WANGZJ/Polymarket_data` is the only data source for our pipeline. Within-HF wallet aggregates (cumcount + shift on `maker`/`taker` columns) ARE allowed. Pontus's Layer-6 on-chain enrichment is HIS contribution to the joined v4 parquet — we consume it, don't reproduce it.
- **Working venv** at `../.venv/` already has TF/Keras, sklearn, pandas, optuna, lightgbm, shap. Don't create a new venv.

---

## Data sync

Raw + feature parquets are too big for git (373 MB). They live in a GitHub release:

- **v3.5 baseline release:** https://github.com/PontusNeumann/CBS-Spring26/releases/tag/alex-data-2026-04-28
- **SHA-256:** `ddc4fc02f9b215615c6b2db7b43cc67c07648d54d165306138156ff29e1fd9ca`
- **Manifest** (schemas + checksums): [`notes/data-manifest.md`](notes/data-manifest.md)

For the v4 wallet-augmented parquets, Pontus publishes a separate release. See [`v4_final_ml_pipeline/README.md`](v4_final_ml_pipeline/README.md) for the download command.

When the cohort or feature engineering changes, publish a new tag like `alex-data-2026-05-XX` and update the manifest.

---

## Maintenance contract

When you change something, update the corresponding doc:

| Change | Update |
|---|---|
| Add / remove / rename a script in v4 pipeline | `v4_final_ml_pipeline/pipeline.md` (script tables) + `v4_final_ml_pipeline/README.md` (stages table) |
| Add a new design decision or revise one | `notes/design-decisions.md` (append D-NNN entry) |
| Run a new pressure test or get a new verdict | `notes/pressure-test.md` (verdict table + Phase section) |
| Change cohort size, feature count, or RQ | `notes/alex-approach.md` + this README's first paragraph |
| Re-publish data | `notes/data-manifest.md` + release on GitHub |
| New output category from v4 pipeline | `outputs/README.md` |
| Significant pivot or new finding | Append to `notes/session-learnings-YYYY-MM-DD.md` |

The four canonical notes (`alex-approach.md`, `design-decisions.md`, `pressure-test.md`, `data-manifest.md`) plus the `v4_final_ml_pipeline/` directory are the source of truth. This README is just a map.
