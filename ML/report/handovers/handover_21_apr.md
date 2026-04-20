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

Three per-trade features that map Alex's feature taxonomy onto our dataset without new data collection. Canonical implementations added to `01_polymarket_api.py` so the next full rebuild includes them; a one-shot patcher applies the same logic to `data/03_trades_features.csv` in place to avoid the 20-minute full-pipeline rerun.

| Feature | Cluster | Non-null / 1.21M | Median | Max | Insertion point |
|---|---|---|---|---|---|
| `market_buy_share_running` | market_context | 1,209,713 | 0.377 | 1.00 | `_add_running_market_features` |
| `wallet_median_gap_in_market` | bet_slicing | 776,856 | 0 s | 4.36 Ms | `expand_features` bursting block |
| `size_vs_market_avg` | interactions | 1,209,713 | 0.103 | 4,520 | `expand_features` interactions block |

All three are strictly point-in-time:

- `market_buy_share_running` = (prior BUY cumsum / prior trade count), both strictly-prior existing features.
- `wallet_median_gap_in_market` uses `diff()` on `(wallet, market)` groups, then `expanding().median().shift(1)` so row *k* sees only gaps from pairs fully before *k*.
- `size_vs_market_avg` = `trade_value_usd / (market_volume_so_far_usd / market_trade_count_so_far)`, both denominators already strictly-prior.

The patcher ran in ~8 min and grew `03_trades_features.csv` from 62 to **65 cols**. Backup at `data/03_trades_features.pre09.csv`.

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

- `03_trades_features.csv` — **65 cols**, 1,209,787 rows. Mother dataframe for modelling.
- `03_trades_features.pre09.csv` — pre-patch backup (62 cols).
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
