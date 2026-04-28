# Alex idea1 data snapshot 2026-04-28

Companion data for [PR #13](https://github.com/PontusNeumann/CBS-Spring26/pull/13). Extract to `ML/report/alex/data/` to reproduce all backtest + sensitivity runs.

## `feature_cols.json` (0.002 MB, sha256:62ebdf8403fe79c5)

- 70 items
- First 3: ['log_size', 'side_buy', 'outcome_yes']

## `markets_subset.parquet` (0.0 MB, sha256:d2b62f60e12494ac)

- Rows: 75
- Columns: 21
- Schema:

  | column | dtype |
  |---|---|
  | id | str |
  | question | str |
  | slug | str |
  | condition_id | str |
  | token1 | str |
  | token2 | str |
  | answer1 | str |
  | answer2 | str |
  | closed | uint8 |
  | active | uint8 |
  | archived | uint8 |
  | outcome_prices | str |
  | volume | float64 |
  | event_id | str |
  | event_slug | str |
  | event_title | str |
  | created_at | datetime64[ms, UTC] |
  | end_date | datetime64[ms, UTC] |
  | updated_at | datetime64[ms, UTC] |
  | neg_risk | uint8 |
  | cohort | str |

## `test.parquet` (17.2 MB, sha256:0ff7e2394990e974)

- Rows: 257,177
- Columns: 17
- Schema:

  | column | dtype |
  |---|---|
  | timestamp | uint64 |
  | block_number | uint64 |
  | transaction_hash | str |
  | log_index | uint32 |
  | contract | str |
  | market_id | str |
  | condition_id | str |
  | event_id | str |
  | maker | str |
  | taker | str |
  | price | float64 |
  | usd_amount | float64 |
  | token_amount | float64 |
  | maker_direction | str |
  | taker_direction | str |
  | nonusdc_side | str |
  | asset_id | str |

## `test_features.parquet` (52.7 MB, sha256:97d88273ebd74f86)

- Rows: 257,177
- Columns: 74
- Schema:

  | column | dtype |
  |---|---|
  | log_size | float64 |
  | side_buy | int64 |
  | outcome_yes | int64 |
  | log_time_to_deadline_hours | float64 |
  | pct_time_elapsed | float64 |
  | log_time_since_last_trade | float64 |
  | is_within_24h_of_deadline | int64 |
  | is_within_1h_of_deadline | int64 |
  | is_within_5min_of_deadline | int64 |
  | hour_of_day_sin | float64 |
  | day_of_week_sin | float64 |
  | day_of_week_cos | float64 |
  | log_n_trades_to_date | float64 |
  | market_buy_share_running | float64 |
  | log_recent_volume_5min | float64 |
  | log_recent_volume_1h | float64 |
  | log_recent_volume_24h | float64 |
  | log_trade_count_5min | float64 |
  | log_trade_count_1h | float64 |
  | log_trade_count_24h | float64 |
  | market_price_vol_last_5min | float64 |
  | market_price_vol_last_1h | float64 |
  | market_price_vol_last_24h | float64 |
  | order_flow_imbalance_5min | float64 |
  | order_flow_imbalance_1h | float64 |
  | order_flow_imbalance_24h | float64 |
  | trade_size_to_recent_volume_ratio | float64 |
  | trade_size_vs_recent_avg | float64 |
  | avg_trade_size_recent_1h | float64 |
  | pre_trade_price | float64 |
  | recent_price_mean_5min | float64 |
  | recent_price_mean_1h | float64 |
  | recent_price_mean_24h | float64 |
  | recent_price_high_1h | float64 |
  | recent_price_low_1h | float64 |
  | recent_price_range_1h | float64 |
  | pre_trade_price_change_5min | float64 |
  | pre_trade_price_change_1h | float64 |
  | pre_trade_price_change_24h | float64 |
  | yes_volume_share_recent_5min | float64 |
  | yes_volume_share_recent_1h | float64 |
  | yes_buy_pressure_5min | float64 |
  | token_side_skew_5min | float64 |
  | implied_variance | float64 |
  | distance_from_boundary | float64 |
  | consensus_strength | float64 |
  | contrarian_score | float64 |
  | is_long_shot_buy | int64 |
  | contrarian_strength | float64 |
  | log_payoff_if_correct | float64 |
  | risk_reward_ratio_pre | float64 |
  | kyle_lambda_market_static | float64 |
  | realized_vol_1h | float64 |
  | jump_component_1h | float64 |
  | signed_oi_autocorr_1h | float64 |
  | log_same_block_trade_count | float64 |
  | log_taker_prior_trades_in_market | float64 |
  | taker_first_trade_in_market | int64 |
  | log_taker_cumvol_in_market | float64 |
  | taker_position_size_before_trade | float64 |
  | log_taker_prior_trades_total | float64 |
  | log_taker_prior_volume_total_usd | float64 |
  | log_taker_unique_markets_traded | float64 |
  | taker_yes_share_global | float64 |
  | taker_directional_purity_in_market | float64 |
  | taker_traded_in_event_id_before | int64 |
  | log_taker_burst_5min | float64 |
  | log_taker_first_minutes_ago_in_market | float64 |
  | log_size_vs_taker_avg | float64 |
  | log_maker_prior_trades_in_market | float64 |
  | market_id | str |
  | bet_correct | int64 |
  | ts_dt | datetime64[ms, UTC] |
  | timestamp | uint64 |

## `train.parquet` (76.3 MB, sha256:b49c3fee7decdb32)

- Rows: 1,114,003
- Columns: 17
- Schema:

  | column | dtype |
  |---|---|
  | timestamp | uint64 |
  | block_number | uint64 |
  | transaction_hash | str |
  | log_index | uint32 |
  | contract | str |
  | market_id | str |
  | condition_id | str |
  | event_id | str |
  | maker | str |
  | taker | str |
  | price | float64 |
  | usd_amount | float64 |
  | token_amount | float64 |
  | maker_direction | str |
  | taker_direction | str |
  | nonusdc_side | str |
  | asset_id | str |

## `train_features.parquet` (226.8 MB, sha256:5127a70ce19cfecf)

- Rows: 1,114,003
- Columns: 74
- Schema:

  | column | dtype |
  |---|---|
  | log_size | float64 |
  | side_buy | int64 |
  | outcome_yes | int64 |
  | log_time_to_deadline_hours | float64 |
  | pct_time_elapsed | float64 |
  | log_time_since_last_trade | float64 |
  | is_within_24h_of_deadline | int64 |
  | is_within_1h_of_deadline | int64 |
  | is_within_5min_of_deadline | int64 |
  | hour_of_day_sin | float64 |
  | day_of_week_sin | float64 |
  | day_of_week_cos | float64 |
  | log_n_trades_to_date | float64 |
  | market_buy_share_running | float64 |
  | log_recent_volume_5min | float64 |
  | log_recent_volume_1h | float64 |
  | log_recent_volume_24h | float64 |
  | log_trade_count_5min | float64 |
  | log_trade_count_1h | float64 |
  | log_trade_count_24h | float64 |
  | market_price_vol_last_5min | float64 |
  | market_price_vol_last_1h | float64 |
  | market_price_vol_last_24h | float64 |
  | order_flow_imbalance_5min | float64 |
  | order_flow_imbalance_1h | float64 |
  | order_flow_imbalance_24h | float64 |
  | trade_size_to_recent_volume_ratio | float64 |
  | trade_size_vs_recent_avg | float64 |
  | avg_trade_size_recent_1h | float64 |
  | pre_trade_price | float64 |
  | recent_price_mean_5min | float64 |
  | recent_price_mean_1h | float64 |
  | recent_price_mean_24h | float64 |
  | recent_price_high_1h | float64 |
  | recent_price_low_1h | float64 |
  | recent_price_range_1h | float64 |
  | pre_trade_price_change_5min | float64 |
  | pre_trade_price_change_1h | float64 |
  | pre_trade_price_change_24h | float64 |
  | yes_volume_share_recent_5min | float64 |
  | yes_volume_share_recent_1h | float64 |
  | yes_buy_pressure_5min | float64 |
  | token_side_skew_5min | float64 |
  | implied_variance | float64 |
  | distance_from_boundary | float64 |
  | consensus_strength | float64 |
  | contrarian_score | float64 |
  | is_long_shot_buy | int64 |
  | contrarian_strength | float64 |
  | log_payoff_if_correct | float64 |
  | risk_reward_ratio_pre | float64 |
  | kyle_lambda_market_static | float64 |
  | realized_vol_1h | float64 |
  | jump_component_1h | float64 |
  | signed_oi_autocorr_1h | float64 |
  | log_same_block_trade_count | float64 |
  | log_taker_prior_trades_in_market | float64 |
  | taker_first_trade_in_market | int64 |
  | log_taker_cumvol_in_market | float64 |
  | taker_position_size_before_trade | float64 |
  | log_taker_prior_trades_total | float64 |
  | log_taker_prior_volume_total_usd | float64 |
  | log_taker_unique_markets_traded | float64 |
  | taker_yes_share_global | float64 |
  | taker_directional_purity_in_market | float64 |
  | taker_traded_in_event_id_before | int64 |
  | log_taker_burst_5min | float64 |
  | log_taker_first_minutes_ago_in_market | float64 |
  | log_size_vs_taker_avg | float64 |
  | log_maker_prior_trades_in_market | float64 |
  | market_id | str |
  | bet_correct | int64 |
  | ts_dt | datetime64[ms, UTC] |
  | timestamp | uint64 |
