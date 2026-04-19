# Handover — 19 April 2026 (late)

**Session focus:** Hybrid HF + API dataset builder, and folder restructure. Supersedes everything from prior sessions on 18 and 19 April; all durable content from those sessions now lives in `project_plan.md` and `docs/design-decisions.md`.

## What was done this session

### 1. Hybrid dataset builder

New `scripts/build_iran_dataset.py`. Single entry point that replaces the API-only path. Flow:

1. Gamma API → enumerate resolved markets under the four target events (114242, 236884, 355299, 357625). 74 resolved markets found.
2. Load HF `markets.parquet` (116 MB, downloaded once via `huggingface_hub.hf_hub_download`). Route each market to `hf` if its `condition_id` is present in HF, else `api`. Split: 67 HF, 7 API (ceasefire markets created after the HF cutoff 2026-03-31).
3. HF path → `duckdb` with `httpfs` streams the remote 38.7 GB `trades.parquet` over HTTPS and filters server-side on the 67 target `condition_id`s. Writes the filtered subset to `data/trades_iran_subset.parquet` (~hundreds of MB) as a local cache for re-runs. Network transfer is the full 38.7 GB once; local disk footprint is only the subset.
4. API path → `fetch_polymarket.fetch_trades` pulls the 7 ceasefire markets via the Data API with side-split pagination. Each market has fewer than ~5k trades so the ~7k offset ceiling does not bite.
5. Normalise schemas (HF uses `taker_direction` / `nonusdc_side` / `token_amount`; API uses `side` / `asset` / `size`), concatenate, run `fetch_polymarket.enrich_trades` (running market and wallet features + expanded six-layer features + trade-timestamp split), write `data/markets.csv`, `data/trades.csv`, `data/prices.csv`, `data/trades_enriched.csv`.

Flags: `--dry-run` (print routing only), `--skip-download` (fail if parquets missing), `--force-download`.

### 2. Folder restructure

`report/` reorganised so the paper, the plan, and the code are each easy to find:

- `ML_final_exam_paper.docx` stays at the top level.
- `project_plan.md` promoted from `memory/` to the top level.
- `design-decisions.md` moved into `docs/`.
- `mldp-project-overview.md` moved into `docs/archive/` (superseded by the plan).
- `memory/` renamed to `handovers/`; `handover_18_apr.md` moved into `handovers/archive/`.
- `eda_outputs/` renamed to `outputs/eda/`.
- `image/` renamed to `assets/`.
- `final_project_guidelines/` renamed to `guidelines/`.

See `project_plan.md` §12 for the full layout and reproduction workflow.

### 3. Script path fixes

Two path bugs surfaced when `fetch_polymarket.py` and the other scripts were moved into `scripts/` earlier in the day:

- `scripts/fetch_polymarket.py` had `OUT_DIR = SCRIPT_DIR / "data"`, which resolved to `scripts/data/` after the move (intended target is `report/data/`). Fixed to `SCRIPT_DIR.parent / "data"`.
- `scripts/eda.py` default output path updated to `outputs/eda/` (was `data/eda_outputs/`).

## State of the data folder

`report/data/` currently holds only `markets.parquet` (116 MB HF metadata) and `_backup_20260419/` (pre-refetch CSV snapshot). The hybrid builder has not yet been run end-to-end, so `trades_enriched.csv` is not present.

## Blockers before modelling

1. **Run the hybrid build.** First run will stream the 38.7 GB HF file once (`trades_iran_subset.parquet` caches the filtered subset for future runs). Sanity check: resulting `trades_enriched.csv` should cover 74 markets with no market truncated at ~7k trades.
2. **Re-run EDA on the full dataset.** The current `outputs/eda/` was produced on the pre-HF API-only extract and is stale.
3. **Modelling.** Deliverables 12 to 18 in `project_plan.md` §10 are still open.
