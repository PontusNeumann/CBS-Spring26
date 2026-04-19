# Design Decisions Log

**Project:** Detecting informed (insider) trading on Polymarket prediction markets using neural networks
**Course:** Machine Learning & Deep Learning (MLDP), CBS
**Started:** 2026-04-19

---

## How to use this doc

Running log of every non-trivial decision made during the project, with:
- **Decision** — what we chose
- **Alternatives considered** — what we rejected and why
- **Justification** — the reason, linked to syllabus/literature where possible
- **Implications** — downstream consequences, limitations, risks
- **Open questions** — things still to resolve

The Methodology and Discussion sections of the final report are drafted from this file. When a decision is made during implementation, add it here *immediately* — reconstructing rationale later is lossy.

---

## 1. Dataset source

### Decision
Use the HuggingFace dataset [`SII-WANGZJ/Polymarket_data`](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data) — specifically `trades.parquet` (28GB, 418M processed trades) and `markets.parquet` (116MB, 538k markets) — as the raw source of on-chain Polymarket trade data.

### Alternatives considered
| Option | Why rejected |
|---|---|
| **Polymarket Data API** (`data-api.polymarket.com/trades`) | Hard cap at ~3,100 trades per market. For the Feb 28 market ($89.65M volume) that's only ~12% of trades. `before`/`after` timestamp filters silently ignored. |
| **CLOB API** (`clob.polymarket.com/trades`) | 401 Unauthorized without signed auth. |
| **Goldsky subgraph** (via `warproxxx/poly_data`) | Stale — indexed only to 2026-01-05. |
| **Polygonscan `getLogs` directly** | Works but ~1 hour per market at free tier rate limits; manual decoding of non-indexed `makerAssetId`; still need market metadata join. HF dataset already did this at scale. |
| **Kaggle "Polymarket Prediction Markets Dataset"** (43k events, 100k markets, December 2025) | Market-level aggregates only. No per-trade rows, no wallet addresses. Useful as supplementary metadata at most. |

### Justification
- HF dataset is sourced directly from the two official Polymarket exchange contracts on Polygon (`0x4bFb41d5...982E` and `0xC5d563A3...f80a`) via RPC — identical underlying data to what we'd pull from Polygonscan ourselves.
- MIT licensed, commercial and research use permitted.
- Trade-level granularity with `maker`, `taker`, `timestamp`, `condition_id`, `price`, `usd_amount`, `token_amount`, `nonusdc_side` — exactly what we need.
- Cross-checked: contract addresses match [Polymarket docs](https://docs.polymarket.com/resources/contract-addresses); claims independently verified via fact-check agent (2026-04-19).

### Implications / limitations
- **Staleness**: data cutoff 2026-03-04, ~6 weeks old at project start. Irrelevant for our resolved markets (all closed before this date), but must be stated in Limitations.
- **NegRisk multi-outcome markets**: for markets routed through the NegRisk contract (`0xC5d5...f80a`), the `taker` field can be the contract address itself rather than an end-user wallet. Must be filtered before wallet-level behavioural modelling, else our features get polluted.
- We still depend on Polygonscan for wallet enrichment (age, nonce) — but only for the 5–15k wallets in our filtered subset, not the full 340M user pool. Reduces that task from "multi-day blocker" to "~1 hour on free tier."

### Open questions
- Do we also want Moralis for historical USDC balance per `(wallet, timestamp)`? Decision deferred — add only if features underperform.

---

## 2. Scope — which markets

### Decision (revised 2026-04-19, after group alignment)

**All 7 resolved sub-markets of Polymarket event 114242 "US strikes Iran by X".** One event, nested-date structure, natural mix of YES and NO outcomes.

| Market | Condition ID (head) | Volume | Resolved |
|---|---|---|---|
| by Jan 11 | `0x843913ab...` | $3M | NO |
| by Jan 14 | `0x64b14a09...` | $14M | NO |
| by Jan 31 | `0xabb86b08...` | $42M | NO |
| by Feb 20 | `0xe1c67f75...` | $19M | NO |
| by Feb 28 | `0x3488f31e...` | $90M | **YES** |
| by Mar 31 | `0x4b02efe5...` | $22M | YES |
| by Jun 30 | `0x797d586a...` | $9M | YES |

Total: 7 markets, ~$199M aggregate volume, 4 NO + 3 YES resolutions (good natural class balance on outcome).

### Why this exact set

- **Same event, same semantic question** ("when does the strike happen before") → features mean the same thing across all markets; pooling is legitimate.
- **Cross-market temporal split is clean**: earliest-settled markets train, middle settled validate, latest settled test. No within-market leakage, no subjective cutoff choice.
- **Mixed outcomes** prevent YES/NO imbalance at the market level; label imbalance at the trade level falls out of per-market base rates.
- **Tight scope**: other Iran-related markets (ceasefire, regime fall, Khamenei, Israel strike) are adjacent but different questions → out of v1. Cross-family pooling = Further Research.

### Magamyman's role in scope (framing note)

Magamyman is not a validation target. He is **the reason we chose this market family**: the $553K documented case on the Feb 28 market is what makes the "informed trading exists on Polymarket Iran markets" premise non-speculative. The project stands on the Columbia paper's aggregate evidence ($143M flagged profit across 210k trade pairs); Magamyman is the canonical illustration. His wallet appears in Discussion as an anecdotal example, not in the Results section.

### Data characteristics (for Feb 28 sub-market, verified 2026-04-19)
- **130,889 trades** in the flagship sub-market alone (42× the Polymarket Data API cap)
- **22,930 unique wallets**
- **$41.98M realised volume** (= $83.95M double-counted, matches Gamma API)
- **Time range**: 2026-01-19 → 2026-02-28
- **YES/NO trade split**: 62,956 / 67,933 (naturally balanced)
- **Heavy-tailed size**: median $16.83, mean $320.71 → log1p essential
- **Surprise outcome**: Feb 28 closed at avg $0.45 yet resolved YES → strong informed-trading signal on final days
- **`end_date` field** in `markets.parquet` is unreliable; use `max(timestamp)` or outcome_prices populated timestamp

Similar extraction required for the other 6 sub-markets. Total dataset at full scope: **~250–400k trades**, **~50k unique wallets** (high overlap expected).

### Alternatives considered
| Option | Why rejected |
|---|---|
| Single market (Feb 28 only) | Previous v1 plan. Cross-market temporal split is much cleaner than within-market; loses nested-date outcome variety. |
| Whole 31-market Iran event (includes ceasefire, Khamenei, regime fall, Israel) | Mixed semantics — "by when does strike happen" differs from "does ceasefire hold" differs from "is regime overthrown." Pooling them would require category features and weakens the shared-feature-meaning argument. |
| Only YES-resolved sub-markets | Kills outcome variety; creates class imbalance on resolution. |
| Iran event + sibling events (Maduro, Biden pardons) | Cross-family pooling = Further Research. v1 stays focused on one event family. |

---

## 3. Target label definition

### Decision (locked 2026-04-19)

**Binary label per trade: ex-post PnL sign.**

For each taker trade, compute per-token final value (1 for winning token, 0 for losing token). Then:
- BUY at price `P`: PnL per token = `final_value − P`
- SELL at price `P`: PnL per token = `P − final_value`

Label `bet_correct = 1 if pnl_per_token > 0 else 0`.

For this YES-winning market this collapses to: **BUY YES → correct, BUY NO → incorrect, SELL YES → incorrect, SELL NO → correct.**

**Training set**: all taker trades (~130k). `side` (BUY/SELL) included as a feature.

**Robustness variant**: retrain on BUY-only subset (~65k). Compare performance and feature importance. This IS the exam rubric's required "model complexity comparison."

**SELLs are always included in feature computation**, regardless of training-set scope — the spread-builder and net-exposure features need them to be computed correctly.

### Options on the table

| Option | Definition | Pros | Cons |
|---|---|---|---|
| **(a) Simple directional** | A BUY on the winning token = 1. BUY on losing token = 0. (SELL flipped accordingly.) | Trivially computable. Matches handover draft. | Ignores price: buying YES at 0.99 = correct, buying YES at 0.50 = also correct. The 0.50 buy is the informed one, but label can't distinguish. |
| **(b) Profitable** | Bet is "correct" if it generated positive PnL at resolution (`price_bought < final_price` for longs). | Incorporates skill dimension (getting in cheap). | Most trades in a resolved-YES market are profitable by construction; may not discriminate. |
| **(c) Informed threshold** | Correct if bet was profitable *and* price at entry < some threshold (e.g. 20% implied probability) while eventual outcome went the other way. | Matches Columbia paper's "informed trading" framing. | Requires choosing threshold; drops most trades from positive class. |
| **(d) Wallet-level label** | Label wallets (not trades) as informed based on aggregate PnL + entry price pattern, then propagate to their trades. | Matches the underlying phenomenon — insiders are people, not individual trades. | Label leaks across time: a wallet labelled "informed" on trade 10 retroactively labels its trade 1. |

### Leaning toward
Option **(b) profitable bet** for the trade-level MLP, with **(c) informed** as a secondary "hard label" for a validation-only sanity check against Magamyman.

### Open questions
- How do we handle partial positions that were closed before resolution? (A trader who bought YES at 0.30 and sold at 0.80 made money regardless of outcome.) → probably need to compute PnL at `min(sell_time, resolution_time)` per position.
- What about SELL-side trades? Sellers are taking the other side of buyers. Correctness for sellers = inverse of correctness for buyers at same price.

---

## 4. Feature engineering philosophy

### Decision
Layer features by scope, with a strict **point-in-time causal constraint** on all aggregations:

1. **Trade-local** (single row, no history): `trade_value_usd`, `log_size`, `side`, `nonusdc_side`. **`price` is dropped from the feature set** so the MLP's probability output `p_hat` is independent of the market's own belief (`market_implied_prob`) — avoids circular reasoning and makes the `p_hat − market_implied_prob` gap a clean signal. `price` is retained only as the benchmark `market_implied_prob` at decision time in the trading rule, not as an MLP input.
2. **Market context** (cumulative/rolling over `condition_id`, strictly before `t`): cumulative volume/trade count, 1h/24h rolling volume and price std, market buy/sell ratio, recent price volatility (computed on the market trajectory — allowed because it's not a per-trade feature of the row itself).
3. **Time features** ⭐: `time_to_settlement_s`, `log_time_to_settlement`, `pct_time_elapsed` — computed against each market's nominal deadline (one `SETTLEMENT_TS` per `condition_id`, parsed from the market question text), not actual resolution, to avoid hindsight leakage.
4. **Wallet history — global** (cumulative/rolling over the taker wallet across all markets, strictly before `t`): `wallet_prior_trades`, `wallet_polymarket_age_days`, `wallet_is_new_to_polymarket`, `wallet_avg_size_so_far`, directional concentration across markets, win rate on prior resolved markets, market-category entropy
4b. **Wallet history — in this market** ⭐ (cumulative/rolling over `(wallet, condition_id)`, strictly before `t`):
   - **Bet-slicing (informed BUYs signature)**: `wallet_prior_trades_in_market`, `wallet_trades_in_market_last_1min`, `wallet_trades_in_market_last_10min`, `wallet_trades_in_market_last_1h`, `wallet_cumvol_same_side_last_10min`, `wallet_median_gap_in_market`, `wallet_is_burst`
   - **Spread builders (non-informed signature)**: `wallet_yes_vol_in_market_so_far`, `wallet_no_vol_in_market_so_far`, `wallet_has_both_sides_in_market`, `wallet_directional_purity_in_market`, `wallet_spread_ratio`
   - **Position-aware SELLs (informed whale-exit signature)**: `wallet_position_size_before_trade`, `trade_size_vs_position_pct`, `is_position_exit`, `is_position_flip`, `wallet_is_whale_in_market`, `size_vs_wallet_trade_avg_in_market`
5. **Interactions** (ratios, cross-terms): `size_vs_wallet_avg`, `size_vs_market_avg`, `size_pct_of_market_cumvol`, `size × time_to_settlement`
6. **External — Polygon on-chain identity** ⭐ (promoted to v1 scope 2026-04-19, design revised for causal correctness).

   **Time-invariant scalars** (safe to use directly on any trade row for the wallet):
   - `polygon_first_tx_ts` — first on-chain ERC-20 event ever (fixed, always ≤ any trade_ts we'd query)
   - `funded_by_cex` — binary, did the wallet's first inbound USDC come from a known CEX hot wallet?
   - `cex_label` — which CEX (`binance`/`coinbase`/`kraken`/`okx`/`bybit`/None)
   - `first_usdc_inbound_ts` — timestamp of first USDC received
   - `first_usdc_inbound_amount_usd` — size of first funding event

   **Per-trade causal features** (derived in `build_dataset.py` from bisect on saved timestamp arrays):
   - `wallet_polygon_age_at_t_days` = `(trade_ts − first_tx_ts) / 86400`
   - `wallet_polygon_nonce_at_t` = `bisect_left(outbound_ts, trade_ts)` — outbound tx count strictly before trade
   - `wallet_n_inbound_at_t` = `bisect_left(inbound_ts, trade_ts)`
   - `wallet_n_cex_deposits_at_t` = `bisect_left(cex_deposit_ts, trade_ts)`
   - `wallet_cex_usdc_cumulative_at_t` = `sum(cex_deposit_amounts[:idx])`
   - `days_from_first_usdc_to_t` = `(trade_ts − first_usdc_inbound_ts) / 86400`

   **Why the array-based storage**: flat per-wallet snapshots of counts/sums (what a first-pass enrichment returns) include activity AFTER the trade being modelled → time-series leakage. Storing sorted event timestamps per wallet and bisecting at trade-time gives strictly causal per-trade features with O(log n) lookups.
7. **External — GDELT news lag**: `minutes_since_last_iran_news`, `news_count_last_1h/24h`. Still deferred to v2 unless trivial to add.

**Deliberately excluded from the feature set: `price` / `market_implied_prob`.** The model predicts `p_hat` from features that do NOT include the market's own price. The gap `p_hat − market_implied_prob` is then a genuine independent signal. Cross-market sibling features (the arbitrage-nested-date structure across the 7 markets) are also out of v1 — we're already using all 7 markets for training, but we don't inject other markets' contemporaneous prices as features of a given market's rows. That would be Layer 8 and is Further Research.

### Featured-in detail: four highest-signal feature groups (locked 2026-04-19)

**`time_to_settlement_s`** — seconds remaining until nominal deadline at trade time.
- `SETTLEMENT_TS = 2026-03-01 00:00:00 UTC` (end of Feb 28, implied by market question text)
- Companion features: `log_time_to_settlement = log1p(clip(time_to_settlement_s, 0))`, `pct_time_elapsed ∈ [0, 1]`
- Deadline chosen over actual resolution time so the feature is point-in-time honest (traders only knew the deadline)
- Log transform because insider trades cluster in final hours; linear seconds scale makes that tail invisible

**`wallet_polymarket_age_days`** — days between the wallet's first-ever Polymarket trade and the current trade.
- Computed from the full 418M-row `trades.parquet` via a DuckDB join on `maker`/`taker`
- Companion flag: `wallet_is_new_to_polymarket = (age < 1 day)`
- Free proxy for the Columbia paper's "brand-new wallet" red flag

**On-chain identity cluster (Layer 6)** — causal per-trade features derived from bisect on per-wallet timestamped event arrays.

- Captures the Columbia paper's "brand-new wallet + CEX-funded + single-purpose" insider signature in a form the MLP can exploit per-trade
- Matches the documented Biden-pardons wallet ("shared Kraken wallet" → funded_by_cex=1, cex_label=kraken)
- A wallet's first CEX deposit within 24h of its first Polymarket trade is a textbook insider pattern — captured as an interaction feature in Layer 5 using `polygon_first_tx_ts` + `wallet_polymarket_age` + trade time.

**Provider choice (locked 2026-04-19)**: batch enrichment via **Etherscan V2 `account/tokentx`** endpoint.
- 3 free-tier keys × 5 rps per key × 6 concurrent workers (2 per key) = ~15 rps ceiling; target ~60 min for 55k wallets
- Alchemy's `alchemy_getAssetTransfers` costs 150 CU per call → only 2 rps on free tier → ~15 hours. Not viable for batch.
- Alchemy remains the right provider for **future live deployment** (WebSocket subscriptions to the CTF Exchange contract are not CU-limited the same way) — scripts are structured so the provider can be swapped without touching feature code.

**Concurrency + causal design (locked 2026-04-19)**: `scripts/enrich_onchain.py` writes per-wallet rows containing scalars + list columns (`outbound_ts`, `inbound_ts`, `cex_deposit_ts`, `cex_deposit_amounts_usd`). A new Layer 6 in `build_dataset.py` joins on wallet and bisects the arrays at each row's `trade_ts` to compute the causal per-trade features listed above.

**Bet-slicing cluster** — detects informed wallets that split one directional position across many small rapid-fire trades to avoid moving the price (classic front-running signature).
- Core features: `wallet_trades_in_market_last_1min/10min/1h`, `wallet_median_gap_in_market`, `wallet_cumvol_same_side_last_10min`, `wallet_is_burst = (last_1min_count > 3)`
- All computed with strict `< t` filtering via `groupby([wallet, condition_id]).rolling(...).shift(1)`
- Motivated by the Magamyman pattern: 6 new wallets, hundreds of small trades, all YES, within an hour pre-news

**Spread-builder cluster** — detects wallets buying BOTH sides of the market. These are volatility traders, not informed. They provide a non-informed negative class the model should learn to discount.
- Core features: `wallet_directional_purity_in_market` ∈ [0, 1] (1 = fully one-sided, 0 = perfectly spread), `wallet_has_both_sides_in_market`, `wallet_spread_ratio`, separate running sums on each token

**Position-aware SELL cluster** — distinguishes informed whale-exits (large full-position SELLs near deadline) from uninformative retail profit-takes. Addresses the limitation that SELLs otherwise look like noise.
- Core features: `wallet_position_size_before_trade` (running signed cumsum per token, shifted), `trade_size_vs_position_pct`, `is_position_exit`, `is_position_flip`, `wallet_is_whale_in_market` (wallet's prior cumvol in this market above 95th percentile), `size_vs_wallet_trade_avg_in_market`
- Why it matters: a whale's 100%-exit SELL carries directional information equivalent to an informed BUY — just inverse. Without these features, SELLs look uniformly noisy.

**Combined behavioural taxonomy the MLP can learn:**

| Pattern | Direction purity | Bursty | Position-relative size | Info content |
|---|---|---|---|---|
| Opening informed BUY | high | often | new position | **high** |
| Whale full-position exit SELL | high (was) | sometimes | 100% | **high** |
| Retail profit-take SELL | moderate | rarely | small fraction | low |
| Spread builder (both sides) | low | either | N/A | none |
| Retail single BUY | high | no | N/A | low |

Expected feature-importance narrative in the report: correctness is concentrated in rows with `{high directional purity, bursty or whale-exit, large position-relative size}` → quantifiable evidence of asymmetric information, with clear attribution to specific behavioural patterns.

### Alternatives considered
- **Raw trade features only, no history** — rejected: loses the entire behavioural signal. Columbia paper's core finding is that wallet-level patterns (age, sizing) predict informed trading.
- **Deep representation learning over sequences** (RNN/transformer on each wallet's history) — rejected for v1: adds architectural complexity without clear benefit on a tabular problem. Could add in Discussion as future work.

### Justification — course-grounded pieces
- **Log transform on heavy-tailed features**: L02 (Preprocessing & EDA) lecture flags skewness check via `df.skew()` and says skewed distributions may need log transform before modelling. Trade sizes are log-normal, spanning 6+ orders of magnitude — directly applicable.
- **StandardScaler for gradient-based models**: L02 + L09 (Neural Networks) both stress scaling is required for NNs. "Feature scaling matters — both exercises use MinMax to [0, 1]. L08 stresses GD requires features on a similar scale."
- **Pipeline + ColumnTransformer**: L02 prescribes this as the canonical approach to prevent leakage in CV.
- **Correlation heatmap for redundancy**: L02 multivariate EDA step; will be in our EDA section.
- **L1/Lasso for feature selection**: L04 (Regression & Regularization) prescription; used in baseline comparison.
- **Class-imbalance handling**: L07 prescribes `class_weight='balanced'`, SMOTE/under-sampling, and PR-AUC over ROC-AUC. Insider trades are minority → direct application.

### Justification — improvements beyond course playbook
The MLDP syllabus was built around **static tabular datasets** (Penguins, Iris, Insurance). Our data is **temporal, grouped, and heavy-tailed**. Three deliberate improvements worth calling out in the Methodology section:

1. **Temporal / causal leakage prevention**. The course pipeline prevents leakage from test → train via `Pipeline` but never discusses **lookahead leakage within a time-indexed dataset**. Every wallet-level and market-level aggregation here uses `.groupby(...).cumsum().shift(1)` (or equivalent expanding/rolling windows) so row `t` sees only data from rows with `timestamp < t`. Verified by time-ordered train/test split (not random).

2. **Log1p transforms on monetary and count features**. L02 mentions log transform for skew but applies it case-by-case. We apply it systematically to every USD-denominated and count feature (`log1p(usd_amount)`, `log1p(wallet_prior_trades)`, etc.) because the top 1% of wallets and markets would otherwise dominate StandardScaler.

3. **Entropy-based concentration features**. Directional concentration `max(%buy, %sell)` and Shannon entropy over the wallet's prior market categories are not in the course FE toolkit but are behaviourally meaningful for the insider-trading hypothesis (Columbia paper's "directional concentration" signal). Documented as an adaptation of general concentration-measure theory to our setting.

### Implications / limitations
- Point-in-time engineering is expensive: `groupby().expanding()` is O(N²) naïve; we'll use `cumsum().shift(1)` plus `rolling('1h', on='timestamp')` patterns that are O(N log N).
- Shannon entropy needs a category mapping — use `event_title` or a coarser taxonomy (politics/sports/crypto from market metadata).

### Open questions
- Do we also need an autoencoder pre-feature (reconstruction error on "normal" trades fed into the MLP)? Course lecture L11 covers this. Appealing exam story ("we stacked a shallow AE on top of tabular features") but adds complexity. Defer decision to after baselines.

---

## 5. Dimensionality reduction

### Decision
**No PCA preprocessing in the primary MLP pipeline.** Use PCA only as (a) a single EDA visualisation and (b) a considered-and-rejected alternative documented in Methodology.

### Alternatives considered
| Option | Why rejected |
|---|---|
| **PCA → StandardScaler → MLP** (keep 95% variance) | Destroys feature interpretability (we'd report "PC3 matters" — meaningless); MLPs learn their own input projections in the first hidden layer, so hand-crafted PCA is an uninformative bad-first-layer anti-pattern; only captures linear correlations, flattening the (bursty × directional purity) interactions that are our central hypothesis. |
| **Autoencoder bottleneck → MLP** | Same loss of interpretability. Also adds architectural complexity without clear benefit on ~30–40 features. Keep as possible v2. |
| **Feature selection via L1/Lasso before MLP** | Kept — runs as part of the logistic-regression baseline and gives a sparse importance ranking used to inform manual feature pruning. |

### Justification
- **Narrative integrity.** The entire asymmetric-information framing depends on the ability to say "feature X drives correctness prediction, therefore X reveals information leakage." PCA components are linear combinations of inputs and break this chain. Permutation-importance / SHAP on the raw-feature MLP preserves it.
- **Dimensionality is not high.** L05 positions PCA primarily for distance-based methods (SVM, k-NN) and high-dim domains (images, gene expression). Our 30–40 tabular features are comfortably low-dim; MLPs and ensembles handle redundancy natively via L2/dropout (L09) and bagging (L05).
- **Non-linear interactions.** Our four-quadrant taxonomy (bursty × directional) is by construction non-linear. PCA is linear. The MLP is specifically the model that can capture these interactions.
- **Standard alternatives already in the pipeline.** Correlation heatmap (L02) + Lasso sparsity (L04) + RF feature importance (L05) + L2/dropout (L09) + permutation/SHAP on final model (L14) together handle everything PCA would, without sacrificing interpretability.

### Where PCA earns its keep
- **EDA visualisation**: project wallet-level aggregates onto PC1–PC2, colour by correctness rate. Expected to show visible separation of the four behavioural quadrants. Single high-impact plot for the Results section.
- **Methodology robustness paragraph**: explicitly discussed as considered-and-rejected with the reasoning above. Demonstrates methodological maturity for the rubric.
- **Optional v2 robustness check** (if time): train one MLP variant with PCA (95% variance) vs one without, report the performance delta. Feeds directly into the model-complexity comparison the rubric requires.

### Implications / limitations
- We take on the responsibility of manual redundancy control (correlation heatmap review + Lasso-informed pruning). More work than "just PCA everything" but preserves the story.
- Feature-importance results become the **central Results artefact**, not a side note. Plan the report around that.

---

## 6. Train/test split strategy

### Decision (revised twice — final version 2026-04-19)

**Split by trade timestamp, not by market settlement date.** All 7 markets contribute trades to all three buckets; each trade retains its own market's outcome as the label.

| Bucket | Trade timestamp range | Roughly contains | Outcome mix |
|---|---|---|---|
| **Train** | < 2026-02-01 00:00 UTC | All trades on all 7 markets that occurred before Feb 1. Includes: trades on the Jan 11 / Jan 14 / Jan 31 markets (all NO labels) AND early trades on Feb 28 / Mar 31 / Jun 30 markets (YES labels) | **Mixed YES + NO** |
| **Validation** | 2026-02-01 → 2026-02-21 | Feb 20 market trades (NO labels); ongoing trades on Feb 28, Mar 31, Jun 30 (YES labels). `tau` tuning happens here. | Mixed |
| **Test** | 2026-02-21 → 2026-02-28 | The insider-pressure window. Dominated by Feb 28 market trades (YES) plus final-days trades on Mar 31 and Jun 30. This is where Magamyman and the documented insider cluster operated. | Predominantly YES |

### Why this form beat the cross-market settlement split

**Previous plan (rejected 2026-04-19)**: split by market settlement date — train on first-settled markets, test on last-settled. Rejected for two reasons:

1. **All-NO training set**. The first three settled markets (Jan 11, Jan 14, Jan 31) all resolved NO. A supervised model trained only on NO-outcome trades learns "BUY YES → wrong, BUY NO → right" and inverts when applied to a YES market. Catastrophic outcome-distribution shift at test time.

2. **Mar 31 and Jun 30 auto-resolve with Feb 28**. Their outcomes are mechanically determined by Feb 28's resolution (a strike by Feb 28 is a strike by Mar 31 by Jun 30). If Feb 28 is in validation, Mar 31/Jun 30 outcomes become known at validation time, not later. They're not genuinely held-out future.

The trade-timestamp split fixes both:
- Pre-Feb-1 trades on the (eventually YES-resolved) Feb 28 / Mar 31 / Jun 30 markets **give training set YES-labelled rows** alongside NO-labelled Jan-market rows → mixed outcomes in training → model learns outcome-generalisable patterns.
- Test window is a strict trade-timestamp slice, genuinely held out from training features and validation tuning.

### Per-bucket row-count estimates (based on extracted per-market trade counts)

Rough shape (precise numbers will come from the build step):
- Train bucket: ~170k trades
- Validation bucket: ~100k trades
- Test bucket: ~180k trades (dominated by the Feb 21-28 Feb-28-market spike and final-days Mar 31/Jun 30 trades)

### Implications / limitations

- A wallet active in training AND test is observed in both buckets — but each of its trades keeps its own market's label, so there's no label leakage. Wallet feature state naturally flows through time.
- Trades in the test bucket on Mar 31 / Jun 30 markets occur AFTER Feb 28 resolution has already auto-resolved them; market prices on Mar 31 / Jun 30 will have collapsed to ~$1 YES by then. These are arbitrage-cleanup trades, not genuine prediction — the Feb 28 market trades dominate the interesting test signal. Document in Limitations.
- Some wallets appear in all three buckets. Cross-validation within the training bucket should use `GroupKFold` on wallet if we want to measure wallet-level generalisation, or `TimeSeriesSplit` on timestamp for the temporal story. Both reported.

---

## 7. Trading-signal evaluation (streaming backtest)

### Decision (locked 2026-04-19)

Turn the trade-correctness classifier into a **deployable BUY-YES trading signal** evaluated via a walk-forward, event-driven streaming backtest.

### Timeline structure for the Feb 28 market

```
Jan 19 ───── ... ───── Feb 14    Feb 18    Feb 21 ────── Feb 28
├────── TRAINING ──────┤
                        ├── CALIB ─┤
                                   ├── STREAM REPLAY ──┤
                                                        ↑
                                                  RESOLUTION
```

- **Training (Jan 19 → Feb 14, ~26 days)**: model fit on ex-post PnL-sign label
- **Calibration (Feb 15 → Feb 18, ~4 days)**: held-out slice for Platt/isotonic calibration
- **Stream replay (Feb 21 → Feb 28, ~7 days)**: "live" trading simulation window

Gap between calibration and replay prevents calibration trades bleeding into replay state.

### Entry rule: market-price-relative threshold

For a candidate BUY YES at market price `P_market`:
- **Expected value per token** = `P_model(correct) − P_market`
- **Enter iff** `edge = P_model − P_market > MARGIN`
- Margin covers transaction costs, calibration drift, slippage, adverse selection. Tuned on calibration split; typical range 0.02–0.05.

This means the threshold is **not a fixed scalar but the current market price** — the model's probability must beat the market's implied probability for the trade to be +EV.

### Cutoff-date robustness sweep

Run the stream replay with `N ∈ {14, 7, 3, 1}` (days before deadline). Report PnL vs N as a figure. Expected narrative:
- **Headline result at N = 7** (balance between sample size and information-richness)
- **Curve shape**: rising PnL as N shrinks → insider signal concentrates near deadline (matches Columbia paper's pre-event timing finding)

### Streaming simulation protocol

Treat the test-period trade file as a replayable event stream:

```
state ← build_state_snapshot(all_trades_before=cutoff_ts)
for event in stream_sorted_by_timestamp:
    # decision uses ONLY state, which reflects events BEFORE event.timestamp
    P_market  = state.last_price()
    feats     = build_features_for_hypothetical_buy_yes(state, P_market, event.timestamp)
    P_model   = calibrated_mlp.predict_proba(feats)[:, 1]
    edge      = P_model - P_market
    if edge > MARGIN:
        simulated_trades.append(decision record)
    state.apply(event)        # updates state AFTER using it for decision
settle simulated_trades at resolution
```

Strict invariant: `state.apply(event)` happens **after** the decision. Streaming equivalent of `shift(1)` — preserves the causal guarantee enforced in training features.

### Cadence: one decision per observed trade event

Alternatives (fixed-interval polling, event-triggered) considered but rejected for v1. Per-event cadence maximises sample size and matches the "real Polymarket WebSocket feed" framing cleanly.

### Sizing rule (v1): fixed stake

$100 per triggered entry. Cleanest for PnL reporting. Alternatives documented in Discussion:
- **Proportional to edge**: `stake ∝ P_model − P_market`
- **Kelly criterion**: `f* = (p − q/b) / b` — optimal long-run growth. Mention as theoretically optimal extension.

### Calibration protocol

1. After training, score the calibration split
2. Plot reliability diagram; compute Brier score and ECE
3. If miscalibrated (ECE > ~0.05), fit **isotonic regression** on (score, outcome) pairs from the calibration split
4. Apply the fitted calibrator to all model outputs in stream replay
5. Report pre- and post-calibration metrics in Methodology

### Baselines in the backtest

| Strategy | Rationale |
|---|---|
| Buy-and-hold YES from cutoff | Passive benchmark |
| Random entry at matched frequency | Null hypothesis |
| Follow-the-whales (enter when trade > $10k) | Non-ML heuristic |
| Logistic regression signal (same pipeline) | Tests whether MLP capacity adds edge |
| **Our calibrated MLP signal** | Primary |

Report per strategy: total return, hit rate, mean edge per entry, Sharpe, max drawdown.

### The price-as-feature trap

`P_market` appears in the feature set. Two defences against the model trivially learning "low price → BUY YES is correct":
1. **Price-ablation variant**: train an MLP with `price` dropped; compare feature importances
2. **Oracle benchmark**: `P_baseline = 1 − P_market` (market's implied probability); require `P_model − P_baseline > MARGIN`, not just `P_model > P_market`

Both approaches documented in Methodology.

### Home-run strategy (primary evaluation)

For geopolitical-event markets, asymmetric information is **bursty, not diffuse** — concentrated in the final hours before resolution (Columbia paper: Magamyman 71 min pre-news, Biden pardons "final hours", etc.). We therefore evaluate **two strategies side by side** using the same calibrated MLP:

**Strategy A — General +EV**: `edge > 0.02`, all stream events, fixed $100 stake. Establishes a baseline that the model has usable edge.

**Strategy B — Home-run (primary)**: stricter stacked filter on the same model's output:

```
enter iff all of:
   P_model - P_market > 0.20                 # big edge, not marginal
   time_to_settlement_s < 6 * 3600           # final 6 hours only
   P_market < 0.30                           # cheap token → asymmetric payoff
   wallet_is_burst == False                  # independent signal, not mimicking insider
```

Expected fire rate: **5–30 trades per backtest**, not hundreds. Larger per-trigger stake (e.g. $500) to reflect higher conviction.

### Why home-run focus is the right primary framing

- **Matches the phenomenon**: informed flow on these markets is concentrated events, not continuous edge. Columbia paper's case-by-case documentation is evidence of this.
- **Precision > recall**: we don't need to catch every informed trade. We need to fire rarely and be right. Primary metric becomes **precision@k** and PnL concentration in top-k entries.
- **Bypasses calibration noise at the tail**: at the extreme-confidence tail, miscalibration matters less than at the margin.
- **Enables ground-truth validation**: directly addresses "how many of the K documented Magamyman-class events did we flag?"

### Primary metrics for home-run evaluation

| Metric | Definition |
|---|---|
| **Precision@k** | Share of our triggered trades that resolved correctly, for k = our total triggers |
| **Magamyman-recall** | Share of documented insider wallets whose trades our model flagged (if Columbia paper addresses are available) |
| **PnL concentration** | % of total PnL coming from top 5 trades — high concentration confirms home-run structure |
| **Dollar-weighted precision** | PnL-weighted accuracy, not trade-weighted |
| **Upside ratio** | `mean(wins × (1 − P_market)) / mean(losses × P_market)` — asymmetric payoff capture |

Plot PnL vs N (cutoff-days-before-deadline) for both strategies on the same axes. Expected result: general strategy curve is flat-ish, home-run strategy curve rises sharply as N shrinks.

### Limitations to flag in the report

The sim is an **upper bound** on achievable PnL because it omits:
- Slippage — simulated entries fill at `P_market` exactly; real entries move price
- Queue / rejection — every hypothetical order fills
- Transaction costs beyond margin — not explicitly modelled
- Latency — zero time between event observation and decision

These don't invalidate the signal but constrain interpretation to "theoretical +EV" rather than realised trading returns.

---

## 8. Model architecture & baselines

### Decision (locked 2026-04-19, merged with group project plan)

**Primary: MLP for probability estimation** (L08, L09)
- Architecture: fully-connected feed-forward, 2–4 hidden layers
- Activations: **SELU** with Glorot initialisation (L09 rule-of-thumb; self-normalising, pairs well with dropout)
- Regularisation: **dropout 0.2–0.4 on dense layers**, **batch normalisation after each hidden layer**
- Loss: binary cross-entropy
- Optimiser: Adam with learning-rate schedule
- Input: standardised behavioural + market-state features. **`price` / `market_implied_prob` is withheld from the feature set** — `p_hat` is an independent probability estimate, directly comparable to the market
- Output: `p_hat` = probability the trade resolves correctly. The gap `p_hat − market_implied_prob` is the trading signal
- Class imbalance: `class_weight='balanced'` first (L07); SMOTE on training fold only if still degenerate

**Baselines required by rubric** (L04, L05, L06, L07)
- Logistic regression (plain + L1/Lasso for sparse feature ranking)
- Random Forest (non-linear baseline + feature importance ranking for Discussion)
- Isolation Forest (unsupervised anomaly baseline against the autoencoder arm)
- **Naive market baseline** (`p_hat = market_implied_prob`) — zero-gap by construction, tests the efficient-market null

**Secondary arm: undercomplete stacked autoencoder** (L11)
- Trained on all trade feature vectors, SELU, MSE loss. Does not use the correctness target
- Per-trade reconstruction error = anomaly score
- Cross-check: overlap between top-decile `|p_hat − market_implied_prob|` gap trades and top-decile reconstruction-error trades, benchmarked against random-overlap null
- Two purposes: (1) syllabus coverage for L11, (2) unsupervised sanity check that the gap-based signal picks up structurally anomalous trades

**Model-complexity comparison** (rubric requirement): MLP vs logistic regression on same features + same trading rule. Report ΔPR-AUC, ΔPnL, ΔSharpe, training wall time, inference latency per 1k trades.

**Metrics**
- *Probability quality*: ROC-AUC, PR-AUC, Brier score, calibration curve of `p_hat`
- *Gap quality*: ROC-AUC of `p_hat − market_implied_prob` for predicting `bet_correct`
- *Trading rule*: cumulative PnL, annualised Sharpe, hit rate, max drawdown, turnover, trade count
- *Unsupervised arm*: gap-top-decile × reconstruction-error-top-decile overlap vs random-overlap null

### Trading rule (merged with group plan)

- **Entry**: take the side of the trade whenever `|p_hat − market_implied_prob| > tau`. `tau` tuned on validation markets, frozen for test
- **Sizing**: flat stake (v1). Kelly-scaled variant as robustness check
- **Holding period**: hold to settlement. Early-exit at fixed time-to-event as robustness check
- Two reported variants on the same model's output:
  - **General rule**: `|gap| > tau` with small `tau` (e.g. 0.02)
  - **Home-run rule**: stricter stacked filter (`|gap| > 0.20` AND `time_to_settlement < 6h` AND `market_implied_prob < 0.30`) — primary for geopolitical-event markets where informed flow is bursty

---

## 9. Further research (scoped out of v1)

Deliberate exclusions from the v1 pipeline — flagged as extensions in the Discussion / Further Research section of the report.

### Cross-market contemporaneous features (sibling-price injection)

**What it would add**: inject contemporaneous sibling-market prices as features of a given market's rows. E.g. for a row in the Feb 28 market, include `price_in_feb20_market_at_t`, `price_in_mar31_market_at_t`, `implied_strike_in_our_window = P(Feb28 YES) − P(Feb20 YES)`. Captures arbitrage-propagated insider signal — a spike in the Feb 20 market that hasn't yet propagated to the Feb 28 price is tradable information.

**Why excluded from v1**: We're already using all 7 markets for training via cross-market temporal split. That's the *first-order* use of the nested structure. Adding sibling-price features on top is a second-order enhancement that adds its own causal-guarantee complexity (sibling markets can also have lookahead leakage).

**Why it matters**: plausibly 10–20% improvement in home-run detection precision by catching insiders who traded on the tightest-dated YES market.

### Cross-event-family pooling

Expand beyond the US-strikes-Iran event to other Iran-related markets (ceasefire, regime fall, Khamenei), then further to other geopolitical events (Maduro removal, Biden pardons). Tests whether the structural features generalise across event types, not just within one event family.

### GDELT news-lag features

`minutes_since_last_iran_news`, `news_count_last_1h/24h` from GDELT free API. Direct test of the "informed flow precedes news by X minutes" hypothesis from the Columbia paper. Borderline v1 — may be added if trivial, otherwise deferred.

### Sequence models on wallet trade histories

Feed each wallet's trade sequence into an RNN/Transformer (L10, L11) rather than aggregating to tabular features. Captures temporal micro-patterns that cumulative statistics flatten. Out of v1 because it requires a different architecture and training loop.

### Autoencoder-derived features

Train a shallow autoencoder on the "normal trades" partition (low-conviction retail BUYs). Use reconstruction error as an additional MLP input feature. Separates informed trades from retail noise via unsupervised anomaly signal (L11).

### Sequence models on wallet trade histories

Feed each wallet's trade sequence into an RNN/Transformer (L10, L11) rather than aggregating to tabular features. Captures temporal micro-patterns that cumulative statistics flatten.

---

## 10. Ethics

### Decision
Dedicated Ethics section in the report covering:
- **Privacy**: wallet addresses are pseudonymous but persistent; linking patterns across markets is de-anonymising. We aggregate at the feature level; we do not publish wallet lists.
- **Dual-use**: a trained model could help unregulated market operators surveil users *or* help regulators detect manipulation. Discuss Polymarket's March 2026 insider-trading rules change as context.
- **Label validity**: "informed trading" vs "skill" vs "luck" is genuinely ambiguous. Our labels are probabilistic, not legal determinations.
- **LLM usage disclosure**: required by exam brief. Will document which parts of the pipeline were co-authored with Claude.

---

## Decisions log (append-only)

| Date | Decision | Status |
|---|---|---|
| 2026-04-19 | Dataset source: HF `SII-WANGZJ/Polymarket_data` | Locked |
| 2026-04-19 | Test market: Feb 28 "US strikes Iran" (`0x3488f31e...`) | Locked |
| 2026-04-19 | Full market extraction via HF → 130,889 trades, 22,930 wallets | Done |
| 2026-04-19 | `markets.parquet` `end_date` field not trustworthy; use trade timestamps | Locked |
| 2026-04-19 | FE philosophy: 6-layer causal features with log1p + StandardScaler | Locked |
| 2026-04-19 | Two flagship features: `time_to_settlement` (log-transformed), `wallet_polymarket_age_days` | Locked |
| 2026-04-19 | Wallet-in-market features: bet-slicing cluster (informed BUYs) + spread-builder cluster (non-informed) + position-aware SELL cluster (informed whale-exits) | Locked |
| 2026-04-19 | ~~Label: ex-post PnL sign per taker BUY trade (taker SELLs dropped for v1)~~ superseded | Revised |
| 2026-04-19 | Label: ex-post PnL sign per taker trade; all taker trades in training set; `side` (BUY/SELL) as feature; BUY-only variant trained as robustness check for model-complexity comparison | Locked |
| 2026-04-19 | SELLs always included in feature computation (needed for spread-builder and net-exposure features regardless of training-set scope) | Locked |
| 2026-04-19 | No PCA in primary pipeline; used only for EDA visualisation + methodology robustness check | Locked |
| 2026-04-19 | Trading-signal evaluation via streaming event-replay; market-price-relative edge threshold (`P_model > P_market + margin`); Platt/isotonic calibration on held-out slice | Locked |
| 2026-04-19 | Cutoff-date robustness sweep over N ∈ {14, 7, 3, 1} days before deadline; headline at N=7 | Locked |
| 2026-04-19 | Dual-strategy evaluation: general +EV (edge > 0.02) AND home-run (edge > 0.20, TTS < 6h, price < 0.30) — home-run is primary framing for geopolitical-event markets | Locked |
| 2026-04-19 | ~~Test market: Feb 28 single market~~ superseded | Revised |
| 2026-04-19 | **Scope: all 7 resolved sub-markets of event 114242 ("US strikes Iran by X")** — Jan 11, Jan 14, Jan 31, Feb 20, Feb 28, Mar 31, Jun 30. Total ~250–400k trades, ~50k wallets, 4 NO + 3 YES outcomes. | Locked |
| 2026-04-19 | ~~Cross-market temporal split by settlement date~~ superseded — rejected because train bucket would have been all-NO outcomes (model can't generalise to YES test), and because Mar 31 / Jun 30 auto-resolve with Feb 28 so they aren't genuinely held-out | Revised |
| 2026-04-19 | **Trade-timestamp split across all 7 markets**: train < Feb 1, validate Feb 1 → Feb 21, test Feb 21 → Feb 28. Each trade retains its own market's label. Mixed YES + NO in every bucket. | Locked |
| 2026-04-19 | **Enrichment made strictly causal**: per-wallet output is (scalars for time-invariant features) + (sorted timestamp arrays for count/sum features). `build_dataset.py` Layer 6 bisects at each trade's timestamp → no time-series leakage from post-trade activity into earlier trades' features | Locked |
| 2026-04-19 | **Batch-enrichment provider**: Etherscan V2 `tokentx`, 3 free keys × 6 concurrent workers. Alchemy considered but rejected for batch (free tier = 2 rps for `getAssetTransfers` → ~15 hours vs ~60 min for Etherscan). Alchemy retained for future live-deployment use (WebSocket subscriptions on CTF Exchange). | Locked |
| 2026-04-19 | **Rich Layer 6 feature set**: `polygon_first_tx_ts`, `funded_by_cex`, `cex_label`, `first_usdc_inbound_ts`, `first_usdc_inbound_amount_usd` (scalars); `wallet_polygon_age_at_t_days`, `wallet_polygon_nonce_at_t`, `wallet_n_inbound_at_t`, `wallet_n_cex_deposits_at_t`, `wallet_cex_usdc_cumulative_at_t`, `days_from_first_usdc_to_t` (per-trade causal via bisect) | Locked |
| 2026-04-19 | **`price` / `market_implied_prob` dropped from MLP feature set** so `p_hat` is independent of market's own belief; gap becomes a clean signal. | Locked |
| 2026-04-19 | **Polygonscan enrichment promoted to v1 scope**: `wallet_polygon_age_days`, `wallet_nonce`, `wallet_funded_by_cex` (CEX-funded wallet flag). New `scripts/enrich_polygonscan.py` step between dataset build and MLP training. | Locked |
| 2026-04-19 | **Autoencoder arm (L11) promoted to v1 scope**: parallel unsupervised reconstruction-error signal, cross-checked against gap-based signal. Strengthens syllabus coverage. | Locked |
| 2026-04-19 | **Architecture locked**: MLP 2–4 layers, SELU + Glorot, batch norm, dropout 0.2–0.4, BCE, Adam + LR schedule. | Locked |
| 2026-04-19 | **Naive market baseline** (`p_hat = market_implied_prob`) added alongside logreg / RF / IF — tests efficient-market null directly. | Locked |
| 2026-04-19 | **Magamyman is framing motivation, not validation target**. Feb 28 is included because his documented case is what makes the "informed trading exists on Polymarket" premise non-speculative. His wallet appears as an anecdote in Discussion, not in the Results section. | Locked |
| 2026-04-19 | Cross-market sibling-price injection, cross-event-family pooling, GDELT, sequence models — deferred to Further Research | Locked |
| 2026-04-19 | Target label definition | Locked (see §3) |
