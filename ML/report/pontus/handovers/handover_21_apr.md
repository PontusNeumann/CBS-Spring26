# Handover — 21 April 2026

**Session focus:** Polish pre-EDA report content, regenerate four EDA figures that were illegible, and extend the feature set with the three Layer-5 completions that were derivable without new data collection. No modelling work this session.

## What was done this session

### 1. Report structural fixes (`scripts/07_docx_restructure.py`)

The `ML_final_exam_paper.docx` had three structural issues before modelling work resumes:

- **Heading 1 `pageBreakBefore=true` in the style itself** forced every top-level section onto a new page. Removed so sections flow continuously.
- **References** used a non-numbered `Reference Heading` style and was missing from the TOC. Promoted to `Heading 1` so it picks up the `numId=8` auto-outline numbering and appears in the regenerated TOC as §10. Appendix already lives as `Heading 1` and enters the TOC automatically now that `w:updateFields=true` is set in `settings.xml`.
- **Post-EDA sections had prose bodies** that pre-empted methodology decisions not yet made. Cleared 16 section bodies (50 paragraphs total) down to title/subtitle placeholders. Modelling, results, discussion, ethics, and conclusion are now empty shells.
- Added a one-blank-Normal "headroom" before each heading that previously followed content directly (4 insertions), matching the Introduction-section pattern.
- Two inline tables inserted in EDA body: Table 1 (dataset summary statistics) after the opening paragraph, Table 2 (trade count / USD volume / mean bet_correct by time-to-settlement bucket) next to the home-run paragraph. The nine Figure A.N panels + Tables A.1–A.5 remain in Appendix.

Backup at `ML_final_exam_paper.pre07.docx`.

### 2. EDA figure regeneration (`scripts/04_eda.py`)

Four panels in `outputs/eda/` were illegible at A4 print size. Rewrote in place:

- **A.2 `02_class_balance.png`** — 74 per-market bars collided at the old 3.6-inch height. Replaced with a `GridSpec` that sets figure height dynamically (`0.18 × n_markets + 1.2 ≈ 14"`), centres the class-balance bars in the left column, and drops y-tick font to 6 pt.
- **A.4 `04_outlier_boxplots.png`** — all 8 features shared one y-axis, so the low-scale features collapsed to invisible lines. Replaced with a 2×4 subplot grid with independent y-axes; every box is now visible at its own scale.
- **A.5 `05_correlation_heatmap.png`** — default seaborn colorbar was shorter than the heatmap. Used `mpl_toolkits.axes_grid1.make_axes_locatable` to bind the colorbar to the heatmap axis so heights match exactly.
- **A.6 `06_pca_wallets.png`** — clipped x-axis to `-15` and y-axis to `25` so the dense core is readable instead of being compressed into a quarter of the canvas by outlier wallets.

Regenerated all four PNGs (`pre08.png` backups kept). `word/media/image{5,7,8,9}.png` in the docx swapped in-place via a short zipfile script; no structural docx edit needed.

### 3. Pre-EDA content sharpening (`scripts/08_docx_incorporate_alex.py`)

Alex's reference files (`alex_updates_before_incorporation/{design-decisions,mldp-project-overview,SESSION-HANDOVER}.md`) were compared to our pre-EDA report content. His post-EDA sections describe a separate repo's 7-market scope and are out of scope for us. His pre-EDA framing was sharper in several places worth pulling in. Fourteen targeted edits applied:

- **Abstract** — one sentence grounding the efficient-market null in the Mitts & Ofir (2026) documented USD 143M anomaly.
- **Motivation §2.1** — replaced "useful testbed" prose with three concrete anchors: Columbia USD 143M total, Magamyman USD 553K, USD 1.2M realised on Iran-strike markets specifically (Gambling Insider, 2026-03-11), plus the Maduro / Venezuelan-leadership USD 400K case as pattern evidence that documented Polymarket insider trading has been a geopolitical-markets phenomenon.
- **Related Work §2.3** — dated novelty position citing Polymarket's 10 March 2026 Palantir / TWG AI partnership announcement. Vergence is sports-only; no public ML baseline exists for geopolitical sub-markets.
- **Dataset Description §5.1** — compressed four-event table (Event ID, Theme, Sub-markets, Trades, Volume, YES/NO mix). Totals: 74 markets, 1,209,787 trades, USD 251M, 55 YES / 19 NO. Event breakdown: 114242 = 64 markets / 45 YES / 19 NO (the dated-by-when strike series); 236884 = 3 markets / 3 YES (conflict-ends); 355299 = 5 / 5 (ceasefire-announced); 357625 = 2 / 2 (ceasefire-extended).
- **Target label §5.1** — defended simple directional (Wolfers & Zitzewitz, 2004) against Alex's three alternatives: profitable-PnL collapses to the same binary at trade level, informed-threshold requires an arbitrary entry-price cutoff, wallet-level induces retroactive label leakage across a wallet's own earlier trades.
- **Features §5.1** — adopted Alex's behavioural-cluster labels (bet-slicing, spread-builder, whale-exit). They map 1:1 to our existing columns, so no code changes were needed here — only prose.
- **Split (both copies)** — acknowledged the event-leakage tradeoff explicitly. Defence: research question is within-market detection of informed flow given observed event structure, not forecasting of novel geopolitical events; an event-holdout design is infeasible with only four events. Pre-empts the reviewer objection instead of leaving it.
- **References** — added Gambling Insider (2026-03-11) and Polymarket / Palantir / TWG AI announcement (2026-03-10), APA 7 style, alphabetically sorted.

Backup at `ML_final_exam_paper.pre08.docx`.

### 4. Project-plan sync (`project_plan.md`)

Stale realised-dataset numbers in §1 Summary and §4 Data table: the earlier draft said "346,898 trades, 73,839 wallets" from a pre-HF-rebuild count. Updated to the current **1,209,787 trades / 109,080 wallets** figures. This is the only staleness found in the plan; feature list, split strategy, and Vergence context already match the report.

### 5. Layer-5 feature completions (Bucket 1 — `scripts/09_patch_new_features.py`)

Three per-trade features that map Alex's feature taxonomy onto our dataset without new data collection. Canonical implementations added to `01_polymarket_api.py` so the next full rebuild includes them; a one-shot patcher applies the same logic to `data/03_consolidated_dataset.csv` in place to avoid the 20-minute full-pipeline rerun.

| Feature | Cluster | Non-null / 1.21M | Median | Max | Insertion point |
|---|---|---|---|---|---|
| `market_buy_share_running` | market_context | 1,209,713 | 0.377 | 1.00 | `_add_running_market_features` |
| `wallet_median_gap_in_market` | bet_slicing | 776,856 | 0 s | 4.36 Ms | `expand_features` bursting block |
| `size_vs_market_avg` | interactions | 1,209,713 | 0.103 | 4,520 | `expand_features` interactions block |

All three are strictly point-in-time:

- `market_buy_share_running` = (prior BUY cumsum / prior trade count), both strictly-prior existing features.
- `wallet_median_gap_in_market` uses `diff()` on `(wallet, market)` groups, then `expanding().median().shift(1)` so row *k* sees only gaps from pairs fully before *k*.
- `size_vs_market_avg` = `trade_value_usd / (market_volume_so_far_usd / market_trade_count_so_far)`, both denominators already strictly-prior.

The patcher ran in ~8 min and grew `03_consolidated_dataset.csv` from 62 to **65 cols**. Backup at `data/03_consolidated_dataset.pre09.csv`.

Also added a `FEATURE_CLUSTERS` constant in `01_polymarket_api.py` that maps every feature to one of eight groups (`trade_local`, `market_context`, `time`, `wallet_global`, `bet_slicing`, `spread_builder`, `whale_exit`, `interactions`). Future feature-importance plots can import and group by this so the narrative stays readable as count grows.

### 6. Bucket 2 feasibility memo (delivered, no collection)

Two feature extensions that require new data collection were scoped for a go/no-go decision before the 25 April deadline:

- **`wallet_market_category_entropy`** (#4) — entropy over market categories a wallet has traded in across the full Polymarket universe, not just our 74 markets.
- **Layer 6 on-chain identity** — `wallet_polygon_age_at_t_days`, `funded_by_cex`, `cex_label`, `first_usdc_inbound_ts`, plus causal per-trade features via bisect on per-wallet event arrays.

Key facts established:

- **The `SII-WANGZJ/Polymarket_data` HF mirror already covers the full Polymarket history (418M trades / 538K markets)** with `maker`, `taker`, `condition_id`, `event_id`, `event_slug`, `timestamp`. Filtering on our 109K wallets locally gives every wallet's full cross-market history in minutes — dissolves the Polymarket data-api bottleneck for feature #4 entirely.
- **Etherscan V2 free tier** is 5 cps / 100K calls/day per key. Three keys in parallel cover 109K wallets in ~3 hours real wall-time. Alex's 60-minute-for-55K estimate is optimistic; sustained throughput lands closer to ~3 hours for our full set.
- **Dune Spellbook `labels.addresses` table** (free, public) has Binance / Coinbase / Kraken / OKX / Bybit hot wallets on Polygon — one-off CSV export, 200–2,000 addresses. Solves the `funded_by_cex` list-assembly problem.

Time-estimate table:

| Feature | Collection | Integration | PIT join | Total |
|---|---|---|---|---|
| Category entropy (#4) | 1–2 h HF stream-filter | 0.5 h keyword bucketing + expanding entropy | Clean | **2–3 h** |
| Layer 6 (#5) | 3 h Etherscan V2 + 0.5 h Dune CEX export | 1–2 h bisect integration | Clean | **5–6 h** |

Both fit the 4-day window if run sequentially. Recommendation deferred to next session's go-ahead call.

## State of the data folder

`report/data/` (after the Bucket 1 patch):

- `03_consolidated_dataset.csv` — **65 cols**, 1,209,787 rows. Consolidated dataset for modelling.
- `03_consolidated_dataset.pre09.csv` — pre-patch backup (62 cols).
- `00_hf_trades_cache.parquet` — 50 MB Iran-subset cache.
- `00_hf_markets_master.parquet` — 116 MB, shared across clusters.
- Rest unchanged.

## State of the report

- Sections §1 Abstract through §5.1 Dataset Description + §5.2 EDA: ready for submission.
- Sections §5.3 onwards: empty placeholders awaiting modelling work.
- Appendix: 9 figures (A.1–A.9, four regenerated today) + 5 tables (A.1–A.5). Inline Tables 1 + 2 in §5.1 and §5.2 respectively.
- TOC: will auto-refresh on next Word open (`w:updateFields=true` set).

Report backups preserved: `ML_final_exam_paper.pre07.docx` (pre-restructure), `ML_final_exam_paper.pre08.docx` (pre-Alex-incorporation).

## Code changes to track

- `scripts/01_polymarket_api.py`
  - Three new feature computations inside `_add_running_market_features` and `expand_features`.
  - New module-level `FEATURE_CLUSTERS` dict (8 groups).
- `scripts/04_eda.py`
  - `panel_class_balance` rewritten with `GridSpec` + dynamic height.
  - `panel_outliers` rewritten as 2×4 grid with independent y-axes.
  - `panel_correlation` now uses `make_axes_locatable` for matched colorbar height.
  - `panel_pca_wallets` adds `set_xlim(left=-15)` / `set_ylim(top=25)`.
- `scripts/07_docx_restructure.py` — new, one-shot restructure (page-break fix, References promote, section clears, headroom blanks, inline tables, `updateFields=true`).
- `scripts/08_docx_incorporate_alex.py` — new, 14 targeted pre-EDA text edits + dataset table + two references.
- `scripts/09_patch_new_features.py` — new, one-shot CSV patcher for the three Layer-5 features.
- `project_plan.md` — §1 and §4 realised-dataset numbers corrected.

## Open questions for next session

1. **Bucket 2 go/no-go.** Pursue #4 (~2–3 h, low-risk), Layer 6 (~5–6 h, higher-payoff but more moving parts), both, or leave both as documented future work.
2. **Modelling kickoff.** MLP + baselines + calibration + backtest. Alex's `train_mlp_reframed.py` in `alex_updates_before_incorporation/` is scoped to his 7-market dataset and does not transfer directly; our pipeline will be a parallel implementation on the 65-column 74-market frame.
3. **Figure-to-prose cross-ref pass** once modelling results are in. Current EDA prose references every figure correctly; post-modelling we'll need to add references to any new results figures in Results and Discussion.

---

## Evening update (21 Apr, late PM)

### Layer 7 cross-market entropy — DONE (other terminal)

Streamed the 568 M-row HF trades mirror via duckdb hash-partition-write, bucketed slugs into 8 coarse categories, computed expanding Shannon entropy per wallet. +1 column `wallet_market_category_entropy` (nats). Frame now **66 cols, 91.9 % real values**. Script: `scripts/10_wallet_category_entropy.py`.

### Layer 6 on-chain identity — RUNNING

- Three Etherscan V2 free-tier keys (3 cps / 100 K-per-day each) written to `report/.env`.
- `scripts/03_enrich_wallets.py` launched. First run at 9 workers only hit 2.0 rps; **restarted at 18 workers (3 keys × 6 workers/key)** for 2.6 rps aggregate. Rate limiter still caps each key at 2.86 rps, so Etherscan sees no change.
- **Current state at 21 Apr 22:30 local: ~75.8 K / 109.1 K wallets done (70 %), ETA ~3 h → finish ~01:30 22 Apr.**
- **Failure rate 5.8 %** — Etherscan V2 returning `status=0 message='NOTOK'` on specific addresses, non-retryable. Higher than the ~1 % originally estimated; logged in `project_plan.md` §11 and promoted to Methodology → Known Limitations when the run lands. Those wallets get `wallet_enriched=0` per §5.6.
- Diagnosis of observed throughput: Etherscan server-side latency is spiky (~0.6 s baseline, but ~20–30 % of calls spike to 5–14 s). Not local-network bound — confirmed ping 150 ms RTT / 0 % loss. Not rate-limit bound either — daily cap ~20 K / 100 K used per key. Just slow servers.
- Output: `data/wallet_enrichment.parquet` (199 MB, 12 cols: scalars + timestamp arrays). Atomic checkpoint every 500 wallets, fully resumable. Killed and restarted once cleanly during the 9→18 worker transition with zero loss.
- `caffeinate -i -m -s` running (pid 90494) to keep the Mac awake overnight; lid must stay open unless clamshell-mode prerequisites are met.

### Modelling scaffold — READY

- `scripts/12_train_mlp.py` written and `py_compile`-clean.
- Feature selection by exclusion set, so Layer 6 + Layer 7 columns are auto-picked once they land (target 75 cols).
- LogReg + RF (both `class_weight='balanced'`) + PyTorch MLP (2–4 HL, SELU, Glorot, dropout 0.3, BatchNorm, Adam + `ReduceLROnPlateau`, `BCEWithLogitsLoss`, early-stop on val BCE) + isotonic calibration on val + ECE helper (15-bin equal-width) + loss-curve PNG + `metrics.json` + `feature_list.json`.
- Torch 2.2.2 CPU wheel installed in `py312` env.

### Report — docx now covers Layer 6 / 7 / missingness

- `scripts/13_docx_add_layer6_layer7.py` — one-shot edit on `ML_final_exam_paper.docx`:
  - §5.1 pipeline paragraph: replaced the outdated "No Polygonscan or external on-chain enrichment is used" with an Etherscan V2 description.
  - §5.1 features paragraph: rewritten from six to seven layers; Layer 6 (nine on-chain identity features) and Layer 7 (cross-market category entropy) described explicitly.
  - New paragraph inserted after §5.1 features: missingness treatment (5 binary indicators, train-split-only median imputation at the modelling stage, indicator columns always retained).
- Backup: `handovers/ML_final_exam_paper.pre09.docx`.

### Open reframing question (not decided)

Consideration: elevate the insider-flagger angle from §8 (out-of-scope / Discussion) to a co-primary **RQ2: do the model's largest-gap trades concentrate on features documented in the informed-trading literature?** Same model, zero extra training — just feature-importance + the Magamyman sanity check already planned. Gives a non-zero finding even if the trading rule (RQ1) only breaks even. Decide before Results writeup.

### Tomorrow (22 Apr)

1. Confirm enrichment landed clean; run `scripts/11_add_layer6.py` → +9 cols → frame at 75 cols. Verify with `describe()` on the 9 new columns and `wallet_enriched` coverage.
2. Final EDA pass: regenerate any figures that benefit from the two new feature layers; refresh Table 1 / Table 2 numbers only if the post-resolution filter or Layer 6/7 change them materially.
3. Kick off modelling: run `12_train_mlp.py` on the 75-col frame; land first LogReg + RF + MLP val metrics. Backtest and calibration sweep follow.

---

## Early-morning update (22 Apr, 06:00–09:00)

### Layer 6 enrichment finished ahead of schedule

The enrichment run that was ETA'd at 03:00 local finished closer to 05:30. The tail of the wallet list was dominated by low-activity wallets (small / empty Polygon token-transfer histories), which returned small payloads quickly and bumped the effective rate above the cumulative 2.6 rps average.

- Final initial-pass parquet: 109,080 wallets / 101,571 ok (93.12 %) / 7,509 NOTOK failures (6.88 %).

### Layer 6 integration landed (two passes)

**First pass** surfaced a silent correctness bug. The integrator expected trade timestamps as int64 Unix seconds; the CSV stores them as ISO strings (`"2026-02-02 01:38:14+01"`). The naive fix — `pd.to_datetime(..., utc=True).astype("int64") // 10**9` — silently produced timestamps **1000× too small** because **pandas 3+ stores datetimes at microsecond resolution**, not nanoseconds. Every `searchsorted` on the per-wallet timestamp arrays returned 0, so every numeric Layer 6 feature came out identically zero. Caught by `describe()` — flat-zero summaries are a red flag worth checking.

Fix applied in `scripts/11_add_layer6.py`: force `.astype("datetime64[ns, UTC]")` before casting to int64 so the `// 10**9` is always correct. Worth grepping any future script that does the same idiom on datetimes.

**Second pass** produced sensible feature distributions (median Polygon age 39 days, median nonce 193, heavy-tailed CEX USDC cumulative capped at $10.5 M; see §5.1 `describe()` in the report docx).

### Retry of failed wallets — 93.9 % recovered

User asked for a focused retry of the 7,509 NOTOK wallets. Procedure: backup parquet → `wallet_enrichment.pre-retry.parquet`; filter to ok-only (101,571 rows); re-run `scripts/03_enrich_wallets.py` → resume logic treats the 7,509 missing wallets as todo. Took 37 min at 2.8 rps. Recovery: **7,050 of 7,509 (93.9 %)** — far better than the 30–60 % I initially guessed.

Diagnosis confirmed: most "NOTOK" responses were Etherscan server-side transient throttling, not permanent address rejection. Re-running even immediately after the first pass succeeded for almost all of them.

**Final pipeline state:**

| Layer | Count |
|---|---|
| Total wallets | 109,080 |
| Etherscan-enriched | **108,621 (99.58 %)** |
| Permanent failures | 459 (0.42 %) — Methodology → Known Limitations |

**Final CSV:** 1,209,787 rows × 82 cols. Trade-level Layer 6 coverage **99.69 %** (1,206,050 / 1,209,787). Residual 3,737 trades carry `wallet_enriched=0` + NaN Layer 6.

### Column-count clarification

Plan and docx previously said "Layer 6 adds 9 columns". The script actually emits **12 columns**: the nine semantic features (`wallet_enriched`, `polygon_age_at_t_days`, `polygon_nonce_at_t`, `n_inbound_at_t`, `n_cex_deposits_at_t`, `cex_usdc_cumulative_at_t`, `days_from_first_usdc_to_t`, `funded_by_cex`, `funded_by_cex_scoped`) plus three log variants (`log_polygon_nonce_at_t`, `log_n_inbound_at_t`, `log_cex_usdc_cum`). Plan, docx, and `MISSING_DATA.md` all updated to say **"12 columns (9 semantic + 3 log variants)"**.

Total column breakdown of the 82-col final frame: 65 base (layers 1–5) + 1 Layer 7 entropy + 4 missingness indicator flags from `11b_add_missingness_flags.py` + 12 Layer 6 columns.

### Files updated this pass

- `scripts/11_add_layer6.py` — datetime-unit fix.
- `data/03_consolidated_dataset.csv` — regenerated with correct Layer 6 values (82 cols, 99.69 % enriched).
- `data/wallet_enrichment.parquet` — post-retry (108,621 ok).
- `data/wallet_enrichment.pre-retry.parquet` — pre-retry backup (kept for audit).
- `data/03_consolidated_dataset.pre11.csv` — pre-Layer-6 backup (70 cols).
- `project_plan.md` — row 10, row 10b status → done; "+9 cols" → "+12 cols" in 4 places; §5.6 figures updated to final post-retry numbers; §11 resolved with final coverage stats.
- `data/MISSING_DATA.md` — §1 + §2 column count updated; observed-shares table now includes `wallet_enriched = 99.69 %`; §6 change log entry added.
- `ML_final_exam_paper.docx` — already reflects Layer 6 / Layer 7 from last night's pass; no further docx edits needed.

### Tomorrow (22 Apr proper — the day session)

1. Final EDA pass on the 82-col frame: regenerate any figures that benefit from the two new feature layers; refresh Table 1 / Table 2 if post-resolution filtering shifts numbers materially.
2. Kick off modelling: `scripts/12_train_mlp.py` on the 82-col frame — LogReg + RF + MLP val metrics, then isotonic calibration, then backtest and cutoff-date sweep per §5.2.
3. Decide the RQ2 (insider-flagger) reframing question before the Results writeup. Recommendation from last night still stands: elevate it from §8 to co-primary.
