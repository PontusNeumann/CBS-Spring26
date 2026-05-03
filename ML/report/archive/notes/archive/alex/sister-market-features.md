# Sister-market features — design note

*Captured 2026-04-22. Status: **deferred** pending baseline results; revisit if the baseline is weak or if Feb 28 dominates predictions in a way that suggests cross-market arbitrage is the real signal.*

## The idea

For a trade at time `t` on a given sub-market (e.g. `US strikes Iran by February 27, 2026?`), inject features that encode what its **sister markets** are doing at `t`:

- Current implied probability of each still-open sister market.
- Price spread / arbitrage gaps between sisters (e.g. `P(by Feb 28) − P(by Feb 27)`).
- Whether sisters have already settled, and their resolved outcome if so.
- Relative trading volume (this market vs the sibling cohort).

For the 4-market strike training cohort (Feb 25 / 26 / 27 / 28), a Feb 27 trade at `t` would see features like:

| Feature | What it captures | Available when |
|---|---|---|
| `sister_feb25_resolved_no` | 1/0 — has the Feb 25 sister already settled NO by t? | Only after Feb 25 resolution_ts |
| `sister_feb26_resolved_no` | 1/0 — same for Feb 26 | Only after Feb 26 resolution_ts |
| `sister_feb28_implied_prob_at_t` | Current CLOB mid for Feb 28 | As long as Feb 28 is trading |
| `sister_price_gap_this_vs_next` | `P(sister with next deadline) − P(this market)` | Per active pair |
| `sister_cumvol_ratio` | This market's prior volume / sum of sister prior volumes | Always |

## Why this is defensible (not scope creep)

1. **Plan §8 already reserves it.** "Cross-market sibling-price injection. A second-order arbitrage feature layer using prices of related sub-markets (e.g. 'by Feb 14' price as a feature for the 'by Feb 28' market) is deferred." We're not inventing something off-plan; we're considering bringing forward work the plan acknowledges as valuable.
2. **Matches the documented phenomenon.** Mitts & Ofir 2026 describe informed trading as concentrated across the "by-date" strike series (spread-building: buy YES on "by Feb 28", sell YES on "by Feb 27" if you know strikes happen exactly on Feb 28). Sister features surface this arbitrage signal directly.
3. **Aligns with intended test setup.** At test time we score one market (Apr 18 ceasefire-extended). Its sister context is available (Apr 14 already resolved NO, the 5 ceasefire-end markets and their per-t prices). Sister features are computable at both train and test time.

## Why we're not doing it first

1. **Baseline tells us if it matters.** Without a no-sister-features baseline, we can't attribute performance improvement to sister features specifically. Baseline first → sister features second gives us a clean ablation.
2. **No-lookahead implementation is finicky.** Each feature has to respect `t` — sister resolution status only if that sister's `resolution_ts < t`, sister price only from strictly-prior CLOB quotes, etc. Easy to introduce silent leakage.
3. **Feature proliferation.** 4 training markets × 3 sisters each = 12 pairwise combinations, plus relative rankings. Without careful design, we'd 2-3× the feature count for modest gain.
4. **Time-ordered sister resolution** varies per market — Feb 25 has 0 prior sisters but 3 later; Feb 28 has 3 prior sisters but 0 later. Per-market-per-t sister list is non-trivial to compute.

## How we'd implement it (if we revisit)

1. **Define sister relationship per event family:**
   - Strike markets (event 114242): sisters = all other "US strikes Iran by <date>" markets.
   - Conflict-end (236884): sisters = the 3 "conflict ends by <date>" markets.
   - Ceasefire-end (355299) + extended (357625): natural cross-family grouping.
2. **For each trade at (condition_id, t):**
   - List sisters from the same event (or cross-family group).
   - For each sister, determine status at t: (a) already resolved (t ≥ resolution_ts), (b) still trading, (c) not yet open (rare, if sisters have different start dates).
   - Pull sister CLOB price at t via `merge_asof` backward on sister's price series.
3. **Normalise to deadline-ordered positions:**
   - `sister_before_1_resolved_yes`, `sister_before_1_resolved_no`, `sister_before_1_implied_prob_at_t` (nearest earlier-deadline sister).
   - `sister_after_1_implied_prob_at_t` (nearest later-deadline sister).
   - Pad with NaN where no sibling exists at that offset.
4. **Aggregate features:**
   - Count of resolved-YES sisters before t.
   - Count of resolved-NO sisters before t.
   - Mean implied prob of still-open later-deadline sisters (bullish signal?).

## No-lookahead pitfalls to watch for

- **Sister `resolution_ts` precision.** If a sister resolved at t=T, a trade at t=T+1s can see that resolution. Use strict `<` not `≤`.
- **Sister CLOB price fetching.** Use `merge_asof(direction="backward", tolerance=<short>)` to ensure no future quote leaks.
- **Event-family definition.** A trader active in strikes AND ceasefires probably has separate behavioural signatures; mixing their sister definitions could introduce noise.
- **The `market_implied_prob` CLOB contamination issue (P0-4 in the issues log) affects sister prices too** — if the sister is HF-path, its price is the trade-execution price, not CLOB mid. Sister features inherit this bias.

## Trigger conditions for revisiting

We come back to this note and implement if any of:

- [ ] Baseline MLP test ROC-AUC on Apr 18 ceasefire-extended is below ~0.60 (signal is weak; sister features may rescue it).
- [ ] Per-market inference reveals Feb 28 dominates predictions while Feb 25/26/27 look very different (model isn't generalising cross-market — sister features may help bridge).
- [ ] Feature importance on the baseline shows `time_to_settlement_s` or `market_volume_so_far_usd` as top features (these are within-market proxies; sister features would add true cross-market signal).
- [ ] Report needs a differentiating methodological contribution beyond the baseline pipeline.

## Estimated cost if we do it

| Task | Time |
|---|---|
| Define sister relationships + deadline-offset encoding | 1h |
| No-lookahead sister price injection (merge_asof + status flags) | 2-3h |
| Add sister features to `14_build_experiment_splits.py` output | 1h |
| Verify no leakage (unit tests on synthetic markets) | 1-2h |
| Re-train baseline + MLP with new features | 1h |
| Ablation (with vs without sister features, per-feature importance) | 1h |
| **Total** | **7-9h** |

This is ~1 focused day. Feasible within the 3-day deadline if the baseline is weak enough to justify.

## Decision log

- 2026-04-22: Discussed with Claude. Decided to ship baseline first, revisit if baseline is weak or if Feb 28 dominance analysis suggests cross-market signal is the missing piece.
