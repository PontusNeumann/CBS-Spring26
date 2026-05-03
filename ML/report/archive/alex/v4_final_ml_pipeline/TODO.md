# v4 pipeline — TODO

Pending rewrites / fixes deferred until Pontus delivers `data/train_features_v4.parquet` and `data/test_features_v4.parquet`. Tracked here so the data-arrival checklist is one file.

---

## v4 data-guard rollout — script-by-script

Goal: every script that touches feature parquets must (1) read `_v4` files, not v3.5, and (2) refuse to run unless `feature_cols.json` has been updated to 76 features by `01_validate_schema.py`. Pattern (already applied in three scripts):

```python
TRAIN_PARQUET = "train_features_v4.parquet"
TEST_PARQUET  = "test_features_v4.parquet"
EXPECTED_N_FEATURES = 76  # 70 v3.5 + 6 wallet

# in main():
train_path = DATA / TRAIN_PARQUET
test_path  = DATA / TEST_PARQUET
missing = [str(p) for p in (train_path, test_path) if not p.exists()]
if missing:
    raise SystemExit(f"v4 parquet(s) missing: {missing}. Run 01_validate_schema.py first.")
fcols = json.loads((DATA / "feature_cols.json").read_text())
if len(fcols) != EXPECTED_N_FEATURES:
    raise SystemExit(f"feature_cols.json has {len(fcols)} features, expected {EXPECTED_N_FEATURES}.")
```

### Status

| Script | Stage | Guard added | Stale-path fix | Notes |
|---|---|---|---|---|
| `01_validate_schema.py` | 1 | n/a (Stage 1 IS the validator) | n/a | Already reads v4 parquets by definition |
| `02_causality_guard.py` | 2 | ⚠ — needs rewrite, see below | — | Stale paths in D2 + D4; rewrite once data lands |
| `03_sweep.py` | 4 | ✅ | ✅ | Header updated; LightGBM gated by ImportError |
| `_backtest_worker.py` | 8 (worker) | ✅ | ✅ | LightGBM added to MODELS dict |
| `04_iso_forest.py` | 5 | ✅ | ✅ | Header + v4 data guard |
| `05_optuna_tuning.py` | 6 | n/a (no parquet reads) | ✅ | Header + LightGBM added to `--models` choices and default |
| `_optuna_worker.py` | 6 (worker) | ✅ | ✅ | LightGBM factory + search space + dispatch |
| `06_phase2_falsification.py` | 7.1 | ✅ | ✅ | LightGBM + skip-if-missing + sort-key assertion + headline_model fallback + output renamed `phase2_v4_results.json` |
| `07_rigor_additions.py` | 7.2 | n/a (skeleton) | doc paths fixed | **Skeleton** — `main()` raises NotImplementedError. Implementation needed. |
| `08_complexity_benchmark.py` | 7.3 | n/a (skeleton) | clean | **Skeleton** — `main()` raises NotImplementedError. Implementation needed. |
| `09_shap_top_picks.py` | 7.4 | n/a (skeleton) | doc paths fixed | **Skeleton** — `main()` raises NotImplementedError. Implementation needed. Hardcoded best-model flagged for dynamic lookup when implemented. |
| `10_backtest.py` | 8.1 | ✅ | ✅ | Header + LightGBM + dynamic best-model + sort-key assertion |
| `11_realistic_backtest.py` | 8.2 | ✅ | ✅ | LightGBM + sort-key assertion + bookkeeping bug fixed (losing bets now release concentration slot) |
| `12_sensitivity_sweep.py` | 8.3 | ✅ | ✅ | LightGBM in TARGETS, sort-key assertion, graceful skip on missing preds |

---

## `scripts/02_causality_guard.py` — rewrite once v4 data lands

Current file is the v3.5 `phase1_quick_wins.py` lifted in. Several checks reference stale v3.5 paths and v3.5 feature parquets. Rewrite to anchor against v4.

### Fixes

1. **D4 — stale file list.** The audit walks `["_backtest_worker.py", "07_sweep.py", "10_backtest.py", "06_baseline_idea1.py"]`. In the v4 pipeline the relevant files are `03_sweep.py`, `_backtest_worker.py`, `05_optuna_tuning.py`, `_optuna_worker.py`, `06_phase2_falsification.py`. Stale paths silently skip (line 270 `if not p.exists(): continue`) so the check passes vacuously — real coverage gap.
2. **D2 — re-anchor target.** Currently greps `alex/scripts/06b_engineer_features.py` (v3.5 upstream). Decide whether to keep that audit (v4 inherits all 70 v3.5 cols, so still relevant) AND/OR add a parallel grep over Pontus's wallet-join script once its path is known.
3. **D1 / D3 — point at v4 parquets.** Both currently read `test_features.parquet` (v3.5). Switch to `test_features_v4.parquet` so the recompute checks validate the actual data we'll be modelling on.
4. **D4 heuristic → AST.** The regex `fit_transform(X)$` is brittle (misses `fit_transform(X_full)`, `fit_transform(features)`, etc). Swap for an AST walk that confirms each `StandardScaler` instantiation lives inside a CV loop body.

### v4-specific checks to add

- **Wallet causal-bisection invariant.** Assert each row's wallet feature is computed strictly from data with `timestamp < trade.timestamp` (no peek-ahead). Spot-check on a sample of rows by reconstructing the wallet aggregate from raw on-chain data up to `t-1`.
- **`wallet_funded_by_cex_scoped` non-constancy.** Group by `(taker, market_id)` and assert variance > 0 across rows within group — would catch the static lifetime version slipping through under a renamed column.
- **No forbidden columns at modelling time.** Re-assert `wallet_funded_by_cex`, `n_tokentx`, `wallet_prior_win_rate` are absent (Stage 1 already covers this; cheap to re-check defensively).

### Notes

- Fatal vs non-fatal split should stay the same: leakage / target-corruption checks halt; environmental sanity (F1, F2) note only.
- Rerun against v4 outputs once `01_validate_schema.py` PASSES.

---

## Other v4-data-arrival follow-ups

### Stage 4 sweep cross-checks
- After v4 sweep finishes, eyeball `outputs/sweep_idea1/comparison_table.csv` for the `pca_logreg` row — D-038 elbow K may differ from v3.5's K=20. Confirm `pca_selection.json` records the K and `scree.png` is sensible. If elbow lands at K ≤ 3 the AUC will likely tank vs K=20 — defensible per the L05 framing but flag in the report.
- Verify `_backtest_worker.py` produced an `npz` for **all four** models (`logreg_l2`, `random_forest`, `hist_gbm`, `lightgbm`). LightGBM workers will silently no-op on the import gate; the orchestrator now expects all four.

### Conditional MLP backtest (D-039)
- After Stage 4 results land, check the decision tree: if MLP test AUC > best tree AND DeLong p < 0.05, build `_backtest_worker_keras.py` and add `mlp_keras` to the orchestrator's `model_names`. Otherwise leave MLP out of Stage 8 per D-039.

### Stage 6 Optuna fresh study
- Drop the v3.5 partial study (`study_v3.5_partial.db`) into archive before running Optuna on v4. Document under decisions.

### Stage 7 skeletons — implementation work

The three rigor scripts (`07_rigor_additions.py`, `08_complexity_benchmark.py`, `09_shap_top_picks.py`) are skeletons. Each has its `main()` raising `NotImplementedError` with a TODO body in the comments. The docstrings already describe the algorithms; this is fill-in work, not design work.

**Wall-time estimates:**
- 07 — ~1.5 hr to implement (4 sub-routines: bootstrap CI, DeLong test, permutation importance, learning curves)
- 08 — ~30 min to implement (`measure_one()` + a model factory dict in main)
- 09 — ~1 hr to implement BUT depends on `03_sweep.py` pickling the fitted estimator object (currently it doesn't — only writes metrics + preds). Either pickle the model in 03_sweep.py *and* load it here, or refit best-params in 09 (cheap if Optuna's best_params.json from Stage 6 is on disk).

**Order of implementation when v4 lands:**
1. Add a `dump_fitted_model` step at the end of `03_sweep.py` for at least RF/HGBM/LightGBM (not MLP — keras serialisation is fiddly), so 09 can `pickle.load()` the actual estimator for `shap.TreeExplainer`.
2. Implement 08 first — fastest, smallest scope, no cross-script dependencies.
3. Implement 07 — uses cached preds, no model objects needed (DeLong + bootstrap + perm imp work on (y_true, y_prob) pairs and X_test).
4. Implement 09 last — depends on (1).

When implementing, port the dynamic best-model lookup helper from `10_backtest.py::pick_best_model_from_sweep()` so `BEST_MODEL_DEFAULT` in 09 (and any "best model" reference in 07) auto-resolves to the v4 sweep winner.

### Sort-key alignment safety
- All three Stage 8 scripts (`10_`, `11_`, `12_`) now assert worker pred length matches `n_test` before reordering via `sort_key`.

### `11_realistic_backtest.py` bookkeeping bug — FIXED 2026-04-29

`open_resolutions` entries now carry a 3-tuple `(res_ts, return_amt, entry_bet)`. On release, `open_positions[mid]` is decremented by `entry_bet` (the original commitment) instead of `return_amt` (which is 0 on a loss). Losing bets now correctly free their concentration slot when the market resolves.

`12_sensitivity_sweep.py` shares the same `realistic_backtest` function (imported via `importlib`) so it picks up the fix automatically.

**Implication:** Stage 8.2 ROI numbers re-run against v4 preds will be slightly higher than the previous v3.5 numbers (more bets executed once concentration releases on losses). The +14% headline reported on v3.5 was a conservative undercount; the v4 number will likely be modestly higher even before any wallet-feature signal.

---

## Ordering once data lands

1. Run `01_validate_schema.py` → `feature_cols.json` updated to 76, halts if Pontus's delivery is non-conformant.
2. Rewrite `02_causality_guard.py` per the section above; run it + Pontus's `_check_causal_joined.py`.
3. Walk through and apply the v4 data guard pattern to every ☐ script in the table above.
4. Run Stage 4 (`03_sweep.py`) → comparison_table feeds dynamic best-model lookup in 10_backtest.py.
5. Stage 5–8 in parallel where possible.
6. Inspect cross-checks (sweep + LightGBM + PCA elbow + MLP decision-tree gate).
