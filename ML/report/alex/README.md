# Alex's idea1 workspace

Self-contained ML pipeline for the CBS MLDP exam project. Predicts trade outcomes on Polymarket geopolitical event markets and tests cross-regime transfer (Iran-strike countdown markets → Iran-ceasefire countdown markets).

**Current state (v4, cleaned 64-feature schema, 5 models in realistic backtest):** Headline focus is **MLP (sklearn)** and **Random Forest** — the two models that survive the realistic backtest. MLP test AUC 0.803, RF 0.778. See [`notes/alex-approach.md`](notes/alex-approach.md) and [`outputs/backtest/overview.png`](outputs/backtest/overview.png) for the full picture.

**Research question (locked 2026-04-28, see D-001 + D-034):**
> Can a ML / DL model predict trade outcomes on Polymarket geopolitical event markets, and do the predictions transfer across event regimes?

The original "asymmetric-information detection" framing was retired after pressure tests showed top picks are 100% consensus-aligned and a 3-line naive consensus rule matches the model's top-1% precision (B1b finding).

**Cohort split (cross-regime by construction):**
- **Train** — 65 Iran-strike countdown markets, 1.11M trades, 2025-12-22 → 2026-02-28 (67 days). Strike event at 2026-02-28 06:35 UTC is the cutoff.
- **Test** — 10 Iran-ceasefire countdown markets, 257K trades, 2026-02-28 → 2026-03-31 (31 days). Ceasefire announcement 2026-04-07 determines outcomes (markets with deadline before that resolve NO, after resolve YES).

**Headline numbers (realistic backtest — $10K bankroll, 5% max bet, 20% concentration, no copycats):**

| model       | best strategy        | 31-day ROI | beats naive (+17%) by |
|-------------|----------------------|-----------:|----------------------:|
| **MLP**     | `phat_gt_0.9`        |     **+20%** |               +3 pp   |
| RF          | `phat_gt_0.9`        |       +14% |              -3 pp   |
| naive       | `phat_gt_0.9`        |       +17% |                  —   |
| hist_gbm    | `top5pct_edge`       |       +27% |             +10 pp¹  |
| lightgbm    | `top1pct_phat`       |        +4% |             -13 pp   |
| logreg_l2   | `top1pct_phat`       |        -2% |             -19 pp   |

¹ Single 601-trade cell, not robust — lightgbm on the same strategy is −15%.

**Why MLP and RF are the focus:** they are the only supervised models that beat or match the naive consensus baseline at high-confidence thresholds (`phat_gt_0.9` and above). The classical models (logreg_l2, hist_gbm, lightgbm) cannot routinely produce p_hat > 0.9 because their isotonic-calibrated outputs saturate below that — see [`outputs/backtest/overview_classical.png`](outputs/backtest/overview_classical.png) for the empty-cell evidence.

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
    ├── 11_realistic_backtest.py (Stage 8.2 — emits live progress.html)
    ├── 12_sensitivity_sweep.py (Stage 8.3)
    ├── 13_naive_baseline_backtest.py (Stage 8.4 — naive consensus benchmark)
    └── 14_overview_chart.py   (Stage 8.5 — single shareable PNG)

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
