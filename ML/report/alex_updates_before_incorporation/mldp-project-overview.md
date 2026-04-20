# MLDP Project Overview

**Capturing the structural form of asymmetric information on Polymarket Iran-war markets**

| | |
|---|---|
| Course | Machine Learning & Deep Learning (MLDP), CBS |
| Deliverable | Group project, individual oral on written group product |
| Report | Max 15 pages. Abstract > Intro > Conceptual Framework > Methodology > Results > Ethics > Discussion > Conclusion |
| Last updated | 2026-04-19 |
| Companion doc | `design-decisions.md` — running log of every non-trivial decision, alternatives, justification |

---

## 1. Research question

> Can a neural network capture the **structural form** of asymmetric information in specific Polymarket markets prone to it — using only public on-chain behavioural features, focused on Iran-war markets where informed trading is already **proven** — and can that captured structure be turned into a profitable trading signal?

### Why this framing

Asymmetric information on Iran-war markets is not hypothetical. The Columbia paper (Mitts & Ofir 2026) documented $143M of anomalous profit across 210K trade pairs, with six named insider cases including Magamyman's $553K on the exact market we use. We are not trying to detect *whether* asymmetry exists on these markets — that is already documented. We are characterising **what it looks like**: which behavioural channels carry it, how it concentrates in time, and whether it can be captured from raw on-chain data by an ML model.

## 2. Why this is worth doing

### The novelty is newly defensible

- **Polymarket's own ML detection ("Vergence", built by Palantir + TWG AI) launched 2026-03-10 and is scoped to sports markets only.** Geopolitical markets — our domain — have no publicly disclosed ML detection.
- **Existing academic work is rule-based.** The Columbia paper uses handcrafted statistical signals. Hobbyist GitHub trackers (pselamy, Polywhaler, PolyTrack) are rule-based.
- **No public ML baseline exists** for informed-trading detection on non-sports Polymarket data.

### The enforcement gap is the ethics story

Despite documented cases being public since early 2026, **no wallet has been publicly banned and no profits have been clawed back** — not Magamyman's $553K, not Burdensome-Mix's $485K on Maduro, not the Biden-pardons wallet's $316K. Trump administration dropped two prior federal investigations. Detection without enforcement is the current equilibrium, and that is what makes the Ethics discussion non-trivial.

### The Coplan contradiction

CEO Shayne Coplan on *60 Minutes*, November 2025: insider edge is *"a good thing"* and *"an inevitability."*
Polymarket policy, March 2026: rules explicitly prohibiting it.

That four-month pivot anchors the Ethics section: are permissionless markets and informed-trading protection actually compatible, or can platforms only pick one?

### Syllabus fit

Hits MLP (L09), class imbalance (L07), feature engineering (L02), regularisation (L04), ensembles (L05), explainable AI (L14), responsible AI / calibration (L15). Every section of the rubric.

## 3. What "structural" means here

Asymmetric information is not one thing. It leaks through distinct behavioural channels, each with its own signature:

| Channel | Signature | Feature cluster |
|---|---|---|
| Brand-new wallets front-running news | Low Polymarket age + late entry + cheap price | `wallet_polymarket_age_days`, `time_to_settlement`, `price` |
| Order-slicing to avoid price impact | Bursty same-direction trades in short windows | `wallet_trades_in_market_last_1/10/60min`, `wallet_is_burst` |
| Whale full-position exits | Large trade relative to existing position | `trade_size_vs_position_pct`, `is_position_exit`, `wallet_is_whale` |
| Directional concentration | All bets on one token, no spreading | `wallet_directional_purity_in_market`, `wallet_spread_ratio` |

**Hypothesis**: in a geopolitical market with proven insider flow, these channels are detectable from public on-chain data, and their relative importance differs from what a rule-based detector would assume. The MLP learns which channels dominate. Feature importance reveals the structure.

## 4. Why the US-strikes-Iran event family is the right test bed

1. **Proven asymmetry** — documented insider cases, quantified profits, named wallets (Magamyman $553K, 5 other new wallets). These are the reason we chose this event family. Magamyman's role is framing motivation, not a validation target — he appears as an anecdote in Discussion, not in Results.
2. **Nested-date structure across 7 resolved sub-markets** — all ask the same question ("does the strike happen before X") so features mean the same thing across markets, enabling legitimate pooling and cross-market temporal train/val/test split.
3. **Mix of outcomes** — 4 NO + 3 YES resolutions at the market level; naturally balanced trade-level labels via per-market base rates.
4. **Volume** — aggregate ~$199M across 7 markets, ~250–400k trades expected after full extraction, ~50k unique wallets (substantial overlap across markets).
5. **Known resolution outcome** per market → ex-post PnL-sign label on every trade.
6. **Surprise outcome on Feb 28** — final-day avg price $0.45 yet resolved YES. Informed flow is present and late-concentrated.

## 5. Data

### Source

HuggingFace dataset `SII-WANGZJ/Polymarket_data` — full-history mirror of Polymarket's on-chain trade events, pulled directly from the CTF Exchange contracts on Polygon via RPC. MIT licensed, 418M trades, 538k markets.

**Why this source**: Polymarket's own Data API caps at ~3,100 trades per market. For the Feb 28 market ($89.65M volume) that's ~12% of trades. The HF mirror bypasses this cap entirely.

### Our slice — 7 sub-markets of event 114242 "US strikes Iran by X"

| Market | Condition ID (head) | Volume | Resolved | Role in split |
|---|---|---|---|---|
| by Jan 11 | `0x843913ab...` | $3M | NO | Train |
| by Jan 14 | `0x64b14a09...` | $14M | NO | Train |
| by Jan 31 | `0xabb86b08...` | $42M | NO | Train |
| by Feb 20 | `0xe1c67f75...` | $19M | NO | Validation |
| **by Feb 28** | `0x3488f31e...` | **$90M** | **YES** | **Validation** (our flagship, Magamyman's market) |
| by Mar 31 | `0x4b02efe5...` | $22M | YES | Test (held-out) |
| by Jun 30 | `0x797d586a...` | $9M | YES | Test (held-out) |

**Totals**: ~$199M aggregate volume, ~250–400k trades expected, ~50k unique wallets (high overlap across markets). 4 NO + 3 YES outcome mix.

### Flagship sub-market verified (Feb 28)

130,889 trades, 22,930 unique wallets, $41.98M realised volume, time range 2026-01-19 → 2026-02-28. Naturally balanced YES/NO trade split (62,956 / 67,933). Median trade $16.83, mean $320.71 → heavy-tailed, log1p essential. Closed at avg $0.45, resolved YES → surprise outcome, strong informed-flow signal.

### Pipeline so far

- 28GB `trades.parquet` + 116MB `markets.parquet` downloaded into `data/hf_raw/`
- Feb 28 market filtered → `data/feb28_trades.parquet` (130,889 rows)
- **Next**: filter the other 6 sub-markets → pooled `data/iran_strike_trades.parquet`
- Fact-checked: contract addresses match Polymarket docs, HF dataset verified against primary sources

## 6. Documented insider cases (ground truth)

From the Columbia paper (Mitts & Ofir 2026) and subsequent news coverage:

| Case | Wallet | Amount | Timing | Market |
|---|---|---|---|---|
| **Iran strike** | **"Magamyman" + 5 new wallets** | **$553K (Magamyman)** | **71 min before news, entry at 17% implied prob** | **Our market — US strikes Iran by Feb 28** |
| Iran ceasefire | 50+ new accounts | $550K aggregate | Minutes before Trump ceasefire announcement | Iran ceasefire |
| Maduro removal | Burdensome-Mix | $485K from $38.5K (12.6×) | Hours before capture news | Maduro removal |
| Google YiS | Unnamed | $1M+ from $10K (100×) | Pre-announcement | Google search rankings |
| Taylor Swift | romanticpaul | Undisclosed | Days before public announcement | Swift/Kelce engagement |
| Biden pardons | 2 linked accounts | $316K from $64K | Final hours before pardons | Biden pardons |

Magamyman is our primary validation target. Wallet address to be pulled from the Columbia paper appendix or `pselamy/polymarket-insider-tracker` GitHub repo.

## 7. Platform context: Polymarket's own surveillance posture

| Mechanism | Scope | Launched |
|---|---|---|
| Palantir + TWG "Vergence" AI (ML anomaly detection, pre/post-trade monitoring, trader screening) | **Sports only** per announcement | 2026-03-10 |
| Internal control desk + NFA Regulatory Services Agreement | Platform-wide, US CFTC exchange | 2026-03 |
| UMA optimistic oracle | Resolution only (not detection) | Pre-existing |

### March 2026 rule change (2026-03-23)

Three prohibited trade categories:
1. Trading on stolen confidential information (breach of duty of trust)
2. Trading on illegal tips from someone with a duty of trust
3. Trading by anyone in a position of authority over the event outcome

Applies to both DeFi platform and the CFTC-regulated US exchange. Quoted exec: Chief Legal Officer Neal Kumar.

### Known enforcement actions: **none**

No wallet publicly banned, no profits clawed back, no CFTC or criminal case on the documented cases above.

## 8. Methodology

### Label

Binary per trade: **ex-post PnL sign**.
- BUY at price P: PnL = `final_value − P`
- SELL at price P: PnL = `P − final_value`

where `final_value` = 1 for the winning token, 0 for the loser. Label = 1 if PnL > 0.

For this YES-winning market: BUY YES → correct, BUY NO → wrong, SELL YES → wrong, SELL NO → correct.

**Training set**: all taker trades (~130k), with `side` (BUY/SELL) as a feature. BUY-only variant trained separately as the model-complexity robustness check the rubric requires.

### Features — six layers with a strict causal guarantee

**Invariant**: every feature for row at time `t` uses only data from rows with `timestamp < t`. Enforced via `groupby().cumsum().shift(1)` and `rolling(window, on='timestamp')`. No lookahead.

**Important: `price` / `market_implied_prob` is deliberately excluded from the feature set.** This makes the MLP's output `p_hat` independent of the market's own belief, so the gap `p_hat − market_implied_prob` is a clean signal. `price` is used only as the benchmark in the trading rule, not as a model input.

| Layer | Scope | Example features |
|---|---|---|
| 1. Trade-local | single row | `log_size`, `side`, `nonusdc_side` |
| 2. Market context | cumulative/rolling on market | `market_cumvol`, `market_vol_1h/24h`, `market_price_std_1h`, market-level price trajectory stats |
| 3. Time (⭐) | derived from deadline | `time_to_settlement_s` + log variant, `pct_time_elapsed` |
| 4. Wallet global | rolling on wallet across all markets | `wallet_polymarket_age_days`, `wallet_prior_trades`, `wallet_dir_concentration`, `wallet_win_rate_so_far` |
| 4b. Wallet-in-market (⭐) | rolling on (wallet, market) | Three clusters below |
| 5. Interactions | ratios and cross-terms | `size_vs_wallet_avg`, `size × time_to_settlement` |
| 6. On-chain identity (⭐) | per-wallet Polygonscan enrichment | `wallet_polygon_age_days`, `wallet_nonce`, `wallet_funded_by_cex` (first inbound from known CEX hot wallet — Kraken, Binance, Coinbase) |

### The three wallet-in-market clusters

- **Bet-slicing (informed BUY signature)** — `wallet_trades_in_market_last_1/10/60min`, `wallet_cumvol_same_side_last_10min`, `wallet_is_burst`. Detects Magamyman-style order slicing.
- **Spread-builder (non-informed signature)** — `wallet_directional_purity_in_market`, `wallet_spread_ratio`, `wallet_has_both_sides_in_market`. Negative class: vol traders, not informed.
- **Position-aware SELL (whale-exit signature)** — `wallet_position_size_before_trade`, `trade_size_vs_position_pct`, `is_position_exit`, `is_position_flip`, `wallet_is_whale_in_market`. Distinguishes whale liquidations (informed) from retail profit-taking (noise).

### Combined behavioural taxonomy the MLP can learn

| Pattern | Direction purity | Bursty | Position-relative | Info content |
|---|---|---|---|---|
| Opening informed BUY | high | often | new position | high |
| Whale full-position exit | high (was) | sometimes | 100% | high |
| Retail profit-take | moderate | rarely | small fraction | low |
| Spread builder | low | either | N/A | none |
| Retail single BUY | high | no | N/A | low |

### Improvements beyond the course baseline (for Methodology section)

The MLDP syllabus centres on static tabular data (Penguins, Insurance). Ours is temporal, grouped, and heavy-tailed. Three deliberate improvements worth calling out:

1. **Causal leakage prevention** via strict `< t` constraint on all aggregations (not in lecture)
2. **Systematic `log1p`** on all USD and count features (lecture applies it case-by-case)
3. **Entropy-based concentration features** (directional purity, market diversity) — not in course toolkit

**No PCA**: destroys interpretability, MLPs learn projections natively in the first hidden layer, flattens the non-linear bursty × directional interactions that are our central hypothesis. Correlation heatmap (L02) + Lasso (L04) + RF importance (L05) + L2/dropout (L09) + permutation importance / SHAP on final model (L14) handle redundancy without sacrificing interpretation.

### Models

| Model | Role |
|---|---|
| **MLP** (L08, L09) | Primary. 2–4 hidden layers, **SELU + Glorot**, **batch norm**, dropout 0.2–0.4, BCE, Adam + LR schedule, `class_weight='balanced'`, StandardScaler after log1p |
| Logistic regression (L04) | Baseline + L1/Lasso feature ranking |
| Random Forest (L05) | Non-linear baseline + feature importance ranking for Discussion |
| Isolation Forest (L07) | Unsupervised anomaly baseline |
| **Naive market baseline** (`p_hat = market_implied_prob`) | Tests the efficient-market null — zero gap by construction, zero signal |
| **Undercomplete stacked autoencoder** (L11) | Parallel unsupervised arm. Reconstruction error cross-checked against gap-based signal. Strengthens L11 syllabus coverage. |
| BUY-only MLP variant | Model-complexity robustness check per rubric |

**Metrics**
- *Probability quality*: ROC-AUC, PR-AUC, Brier score, calibration curve of `p_hat`
- *Gap quality*: ROC-AUC of `p_hat − market_implied_prob` for predicting `bet_correct`
- *Trading rule*: cumulative PnL, annualised Sharpe, hit rate, max drawdown
- *Unsupervised arm*: overlap between top-decile gap trades and top-decile reconstruction-error trades vs random-overlap null

### Train/test split — trade-timestamp temporal

| Bucket | Trade-timestamp range | Contents | Outcome mix |
|---|---|---|---|
| **Train** | before 2026-02-01 | All pre-Feb-1 trades across all 7 markets (Jan markets settled NO + early trades on later YES-resolved markets) | Mixed YES + NO |
| **Validation** | 2026-02-01 → 2026-02-21 | Feb 20 market (NO) + ongoing trades on YES markets | Mixed |
| **Test** | 2026-02-21 → 2026-02-28 | The insider-pressure window. Dominated by Feb 28 market trades (Magamyman's window). | Predominantly YES |

**Why trade-timestamp rather than cross-market-settlement split**: settling markets cluster (Mar 31 and Jun 30 auto-resolve the instant Feb 28 resolves YES, because a strike by Feb 28 implies a strike by Mar 31). A settlement-date split would put all-NO outcomes in training (first three markets resolved NO) and the model wouldn't generalise to a YES test market. Trade-timestamp split delivers outcome-mixed training rows because *pre-Feb-1 trades on eventually-YES markets exist* (Mar 31 market was traded from Dec 22).

Each trade retains its own market's outcome as label — no mixing. `GroupKFold` on wallet for cross-validation within training; `TimeSeriesSplit` on timestamp as an alternative.

## 9. Trading-signal evaluation

### Timeline structure

```
Jan 19 ────── Feb 14    Feb 18    Feb 21 ────── Feb 28
├── TRAINING ──┤
                ├── CALIB ─┤
                           ├── STREAM REPLAY ──┤
                                                ↑ RESOLUTION
```

### Entry rule — market-price-relative threshold

For a candidate BUY YES at market price `P_market`:
- **EV per token = P_model − P_market**
- Enter iff `edge > MARGIN` (margin covers fees, calibration drift, slippage; typical 0.02–0.05)

The threshold is **not** a fixed scalar — the model's probability must beat the market's implied probability (= price itself) to be +EV.

### Two strategies evaluated side by side

| Strategy | Gate | Sizing | Primary metric |
|---|---|---|---|
| **General +EV** | `edge > 0.02` | fixed $100 | total PnL, Sharpe |
| **Home-run (primary)** | `edge > 0.20` AND `time_to_settlement < 6h` AND `price < 0.30` | larger per trigger | precision@k, PnL concentration |

**Why home-run is primary for geopolitical markets**: informed flow is bursty, not diffuse — Columbia paper's cases are all rare, large, pre-event events. Strategy shape should match the phenomenon.

### Streaming backtest protocol

Replay trades in timestamp order. At each event, decision uses only state strictly before the event (`state.apply(event)` happens AFTER the decision). Streaming equivalent of the `shift(1)` guarantee enforced in training features.

### Cutoff-date sweep

Run replay with `N ∈ {14, 7, 3, 1}` days before deadline. Plot PnL vs N. Expected shape: home-run curve rises sharply as N shrinks — informed flow concentrates near deadline.

### Calibration

After training, isotonic regression on the held-out calibration slice. Report Brier score and ECE. Calibration matters because EV math only works if `P_model = 0.8` actually means right 80% of the time.

### Baselines to beat

Buy-and-hold YES · Random entry at matched frequency · Follow-the-whales heuristic · Logistic regression signal · Our calibrated MLP.

### Price-as-feature trap

`price` is a feature. To prevent the model trivially learning "low price → BUY YES correct":
- **Price-ablation variant**: train with `price` dropped; compare feature importance
- **Oracle benchmark**: require `P_model − (1 − P_market) > margin`, not just `P_model > P_market`

## 10. Ground-truth validation

Pull Magamyman's wallet address from the Columbia paper appendix or the `pselamy/polymarket-insider-tracker` GitHub repo. Check:
- Does the MLP assign high P(correct) to his known trades?
- Do the home-run triggers fire on them?
- Which feature values does the model find most salient for him?

Single most important sanity check.

## 11. Scope, locked decisions, and further research

### Locked for v1

- **All 7 sub-markets of event 114242** pooled into one dataset with cross-market temporal split (earliest train, middle validate, latest test)
- All taker trades in training, with `side` as feature; BUY-only variant as robustness
- Ex-post PnL-sign label
- Six-layer causal feature set with `price` / `market_implied_prob` deliberately excluded
- **Polygonscan enrichment in-scope**: `wallet_polygon_age_days`, `wallet_nonce`, `wallet_funded_by_cex`
- No PCA, StandardScaler after log1p
- MLP: SELU + Glorot + batch norm + dropout (L09-idiomatic)
- Autoencoder parallel arm (L11 syllabus coverage)
- Naive market baseline (`p_hat = market_implied_prob`) alongside logreg / RF / IF
- Streaming event-replay backtest, general + home-run strategies, cutoff-date sweep

### Deferred to Further Research (per `design-decisions.md` §9)

- Cross-market sibling-price injection (second-order arbitrage feature layer)
- Cross-event-family pooling (Iran ceasefire, Maduro, Biden pardons)
- GDELT news-lag features (may add if trivial)
- Sequence models on wallet trade histories (RNN/Transformer, L10/L11)

## 12. Ethics

Report section to cover:
- **Privacy**: wallet addresses are pseudonymous but persistent; linking patterns is de-anonymising. We aggregate at the feature level and do not publish individual wallet lists.
- **Dual-use**: a trained model could help regulators detect manipulation *or* help platforms surveil users. Polymarket's March 2026 rules vs Coplan's November 2025 comments frame the real policy tension.
- **Label validity**: "informed trading" vs "skill" vs "luck" is genuinely ambiguous. Our labels are probabilistic signals, not legal determinations.
- **Enforcement gap**: documented cases walk away with the money. Detection without consequence is the current equilibrium; our work is relevant only if that equilibrium shifts.
- **LLM-usage disclosure**: required by exam brief. We document which parts of the pipeline were co-authored with Claude.

## 13. Current status and next steps

| Step | Status |
|---|---|
| Data source identified + 28GB downloaded | ✅ |
| Extract all 7 sibling sub-markets of event 114242 (451,933 trades, 57,788 wallets) | ✅ |
| Build labelled feature matrix (63 cols × 451k rows, zero nulls) | ✅ |
| EDA (8 plots + narrative `report.html`) | ✅ |
| Baselines (logreg L1/L2, RF, IF, naive market) — naive best, RF catastrophically inverts on test | ✅ |
| MLP v1 on `bet_correct` — val ROC 0.62, test ROC 0.23 (inverts) | ✅ |
| HPO (L13, 6 configs) — all invert on test, architecture not the fix | ✅ |
| Permutation importance on MLP (L14) — direction features dominate | ✅ |
| Autoencoder arm (L11) + gap-vs-reconstruction-error overlap — agree on val (2.78×), diverge on test (1.19×) | ✅ |
| **Reframed MLP** (target = P(YES), direction features excluded) — val ROC jumped 0.62 → **0.94**, test no longer inverts but weak (0.50) due to single-market structure | ✅ |
| Streaming backtest on reframed MLP — MLP underperforms random (78% vs 93%); home-run 0% hit rate. Honest empirical result. | ✅ |
| Layer 6 Polygonscan enrichment — 55,329 / 55,329 wallets, zero failures | ✅ |
| Layer 6 integrated via bisect → `iran_strike_labeled_v2.parquet` (451k × 75 cols) | ✅ |
| Reframed MLP v2 (Layer 6 included) — test target ROC 0.66, still inverts on derived | ✅ |
| **Memorisation diagnosed** — train loss → 0.005 in 1 epoch, model learning market-identifying fingerprints from absolute-scale features | ✅ locked finding |
| **v3 fix** — dropped market-identifying features. Test target ROC 0.73 (+0.07), **backtest MLP general +EV 116.4% return vs random 93%** — single seed, preliminary | ✅ |
| **Finalise feature set + apples-to-apples baselines + multi-seed MLP + ablations + seed-variance backtest** | ⏭ Next session (proper pipeline) |
| Report drafting | ⏭ |

### Key findings so far

- **Full course syllabus coverage**: L02, L04, L05, L07, L08, L09, L11, L13, L14, L15 all have artifacts
- **Two independent methodological findings worth a Results-section paragraph each**:
  1. **Direction-feature distribution shift** (original framing): training (70% NO markets) → test (99% YES trades) flips the sign of `is_buy × is_token1 → bet_correct`, inverting all supervised baselines on test. Reframing target to `P(market_resolves_YES)` with direction features excluded fixes this.
  2. **Market-fingerprint memorisation** (reframed framing): absolute-scale market-level features (time-to-settlement, cumvol, 1h/24h volume logs) let the model lookup "market → outcome" from 7 samples, collapsing train loss to 0.005 in 1 epoch while val stays at 6. Dropping these features raises train loss to a healthy 0.36 and improves test target ROC.
- **Provisional positive result**: MLP v3 general +EV strategy returns 116.4% vs random 92.9% on the test window — +23.6 pp of alpha. Single seed, ad-hoc feature set. **Requires confirmation via finalised pipeline before being reportable.**
- **Home-run strategy consistently fails** (−100%) — open question whether filter is mis-specified or model's extreme-confidence picks are genuine contrarian signals.

### Key findings so far

- **Full course syllabus coverage**: L02, L04, L05, L07, L08, L09, L11, L13, L14, L15 all have artifacts
- **Inversion is structural, not architectural** — trade-timestamp split's train/test outcome-distribution asymmetry (70% NO train → 99% YES test) makes direction features toxic on test
- **Reframing to P(market resolves YES)** with direction features excluded decouples the model from this trap; val ROC leaps to 0.94
- **Severe overfitting on 7 markets** (train loss 0.005 vs val 5.9) — data-scarcity issue at the market level. Layer 6 features may help by adding trade-specific (not market-identifying) signal
- **Backtest exposes** that val-learned signal doesn't translate to test PnL. Market over the test window was highly inefficient (any strategy makes 90%+), but MLP underperforms random — the model's confidence actively picks wrong trades

## 14. References

**Academic / primary data**
- Mitts & Ofir (2026), *From Iran to Taylor Swift: Informed Trading in Prediction Markets*. SSRN abstract 6426778.
- HuggingFace: [`SII-WANGZJ/Polymarket_data`](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data) (MIT licensed).
- [Polymarket docs](https://docs.polymarket.com) — contract addresses, API endpoints, UMA resolution.

**Platform / regulatory context**
- [Polymarket Market Integrity page](https://polymarketexchange.com/market-integrity.html)
- [Bloomberg, 2026-03-23 — "Polymarket Implements New Insider Trading Rules After Scrutiny"](https://www.bloomberg.com/news/articles/2026-03-23/polymarket-implements-new-insider-trading-rules-after-scrutiny)
- [BusinessWire, 2026-03-10 — Polymarket + Palantir/TWG partnership (sports only)](https://www.businesswire.com/news/home/20260310736467/en/Polymarket-Partners-With-Palantir-and-TWG-AI-to-Build-Next-generation-Sports-Integrity-Platform)
- [CBS News — rule-change summary](https://www.cbsnews.com/news/polymarket-insider-trading-rules-iran-war-venezuela/)

**Documented cases**
- [NPR, 2026-03-01 — Magamyman $553K](https://www.npr.org/2026/03/01/nx-s1-5731568/polymarket-trade-iran-supreme-leader-killing)
- [NPR, 2026-04-16 — Biden pardons $316K](https://www.npr.org/2026/04/16/nx-s1-5786580/a-polymarket-trader-made-300-000-betting-on-bidens-pardons)
- [Harvard Corp Gov Forum — Columbia paper summary](https://corpgov.law.harvard.edu/2026/03/25/from-iran-to-taylor-swift-informed-trading-in-prediction-markets/)

## 15. Open team questions

- Does the v1 scope (single market, home-run focus, single event family) match your expectation, or should we go broader?
- Work split. Suggested: (1) data pipeline + features, (2) MLP + baselines, (3) trading backtest + PnL, (4) report drafting + ethics.
- LLM-usage disclosure norm — agree format now.
- Who pulls Magamyman's wallet address from the Columbia paper appendix / GitHub trackers?

---

## Appendix A — Exam requirements (quick reference)

- Individual oral on written group product (2–4 members)
- Max 15 pages
- Custom dataset (few thousand rows, good number of columns) — our 130k × 35 features clears this comfortably
- Must include: model complexity analysis vs baseline, ethical consideration, LLM usage disclosure
- Report template: Abstract (200 words) > Intro > Conceptual Framework > Methodology > Results > Ethics > Discussion > Conclusion
- Reference strong exemplar: Face Mask Detection (22 pages, 4 models, Grad-CAM interpretability)

## Appendix B — APIs considered and why we rejected them

| API | Why rejected |
|---|---|
| Polymarket Data API (`data-api.polymarket.com/trades`) | Hard cap ~3,100 trades per market; timestamp filters silently ignored |
| CLOB API (`clob.polymarket.com`) | 401 Unauthorized on trades endpoint without signed auth |
| Goldsky subgraph (via `warproxxx/poly_data`) | Stale — indexed only to 2026-01-05 |
| Polygonscan `getLogs` direct | Works but ~1h per market at free tier; manual decoding; HF already did this at scale |
| Kaggle metadata dataset | Market-level aggregates only, no per-trade rows |
| **HuggingFace `SII-WANGZJ/Polymarket_data`** | **Chosen.** Trade-level, MIT licensed, 418M trades, verified against primary sources |

## Appendix C — Sibling Iran-strike markets (context)

From the larger Iran event ($529M volume across 31 sub-markets, all resolved):

| Market | Condition ID (head) | Volume | Resolved |
|---|---|---|---|
| by Jan 11 | `0x843913ab...` | $3M | NO |
| by Jan 14 | `0x64b14a09...` | $14M | NO |
| by Jan 31 | `0xabb86b08...` | $42M | NO |
| by Feb 20 | `0xe1c67f75...` | $19M | NO |
| **by Feb 28** | `0x3488f31e...` | **$90M** | **YES (our target)** |
| by Mar 31 | `0x4b02efe5...` | $22M | YES |
| by Jun 30 | `0x797d586a...` | $9M | YES |

Implied from these outcomes: the actual strike occurred between Feb 21 and Feb 28, 2026. The Feb 20 market was the tightest-dated NO; the Feb 28 market the tightest-dated YES. This nested-date arbitrage structure is captured in the Further Research section of `design-decisions.md`.
