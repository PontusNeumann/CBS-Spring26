"""
_common.py — shared utilities for the v4 final ML pipeline.

Functions extracted from 10_backtest.py / 11_realistic_backtest.py /
12_sensitivity_sweep.py / 06_phase2_falsification.py to eliminate
3-4 way duplication of the same code.

Imports across the pipeline:

    from _common import (
        ROOT, DATA, SCRATCH, OUT,
        compute_pre_yes_price_corrected,
        compute_cost_and_edge,
        general_ev_rule,
        home_run_rule,
        top_k_mask,
        market_resolution_time,
        parse_deadline,
        STRIKE_EVENT_UTC,
        CEASEFIRE_ANNOUNCEMENT_UTC,
        COST_FLOOR_DEFAULT,
        LIQUIDITY_SCALER_DEFAULT,
    )

Path constants resolve to the canonical alex/ workspace, regardless of
where in v4_final_ml_pipeline/ the importing script lives.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path constants — anchored to alex/ workspace
# ---------------------------------------------------------------------------

# this file is at alex/v4_final_ml_pipeline/scripts/_common.py → parents[2] = alex/
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SCRATCH = ROOT / ".scratch"
OUT = ROOT / "outputs"

# ---------------------------------------------------------------------------
# Realism + cohort constants
# ---------------------------------------------------------------------------

COST_FLOOR_DEFAULT = 0.05  # caps payoff at 19× per win
LIQUIDITY_SCALER_DEFAULT = 1.0  # 1/N copycats sharing fill; default = no copycats

STRIKE_EVENT_UTC = pd.Timestamp("2026-02-28T06:35:00", tz="UTC")
CEASEFIRE_ANNOUNCEMENT_UTC = pd.Timestamp("2026-04-07T23:59:59", tz="UTC")

DEADLINE_RE = re.compile(
    r"by\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d+)(?:,\s*(\d{4}))?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Market metadata helpers
# ---------------------------------------------------------------------------


def parse_deadline(question: str, year: int = 2026):
    """Parse 'by [Month] [Day], [Year]' from a market question string."""
    m = DEADLINE_RE.search(question)
    if not m:
        return None
    month, day, y = m.groups()
    return pd.Timestamp(f"{month} {day} {int(y) if y else year}", tz="UTC")


def market_resolution_time(markets: pd.DataFrame) -> dict:
    """Map market_id → resolution_timestamp (epoch seconds) per cohort.

    Train cohort resolves at the strike event; test cohort resolves at the
    ceasefire announcement (capped at the deadline if the deadline is sooner).
    """
    out: dict[str, int] = {}
    for _, row in markets.iterrows():
        mid = str(row["id"])
        cohort = row["cohort"]
        deadline = parse_deadline(row["question"]) or pd.Timestamp(
            row["end_date"], tz="UTC"
        )
        if cohort == "train":
            res = STRIKE_EVENT_UTC if deadline >= STRIKE_EVENT_UTC else deadline
        elif cohort == "test":
            res = (
                CEASEFIRE_ANNOUNCEMENT_UTC
                if deadline >= CEASEFIRE_ANNOUNCEMENT_UTC
                else deadline
            )
        else:
            res = deadline
        out[mid] = int(res.timestamp())
    return out


# ---------------------------------------------------------------------------
# Per-token-price bug fix (D-029)
# ---------------------------------------------------------------------------


def compute_pre_yes_price_corrected(raw: pd.DataFrame) -> np.ndarray:
    """Correct pre-trade YES-token price per trade.

    The HF `price` field is per-token (each token has its own price), not
    YES-normalised. Naive `pre_trade_price = price.shift(1)` gives the
    previous trade's per-token price (could be either side). This function
    tracks last-seen token1 and token2 prices separately, shifts to exclude
    current trade, and reconstructs YES probability as
    ``last_t1`` (else ``1 - last_t2``, else 0.5).

    Returns array aligned to input dataframe sorted by (market_id, timestamp).

    Background: discovered 2026-04-28 as the cause of a ~400× ROI inflation.
    See alex/notes/design-decisions.md::D-029 for the full audit trail.
    """
    df = raw.copy()
    df["market_id"] = df["market_id"].astype(str)
    df = df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    df["_t1"] = np.where(
        df["nonusdc_side"].astype(str) == "token1", df["price"], np.nan
    )
    df["_t2"] = np.where(
        df["nonusdc_side"].astype(str) == "token2", df["price"], np.nan
    )
    df["_last_t1"] = df.groupby("market_id")["_t1"].ffill().shift(1)
    df["_last_t2"] = df.groupby("market_id")["_t2"].ffill().shift(1)
    first_idx = df.groupby("market_id").head(1).index
    df.loc[first_idx, ["_last_t1", "_last_t2"]] = np.nan
    pre_yes = df["_last_t1"].fillna(1 - df["_last_t2"]).fillna(0.5).clip(0, 1)
    return pre_yes.values


# ---------------------------------------------------------------------------
# Cost + edge derivation
# ---------------------------------------------------------------------------


def compute_cost_and_edge(
    test: pd.DataFrame,
    p_hat: np.ndarray,
    cost_floor: float = COST_FLOOR_DEFAULT,
):
    """Per-trade cost + edge.

    Cost = price the trader paid for their winning side at trade time.
        BUY YES at p_yes:    cost = p_yes
        BUY NO at (1-p_yes): cost = 1 - p_yes
        SELL YES at p_yes:   cost = 1 - p_yes  (equivalent to BUY NO)
        SELL NO at (1-p_yes):cost = p_yes      (equivalent to BUY YES)

    Edge = p_hat - cost (model's view minus market-implied probability).

    Uses ``pre_yes_price_corrected`` if present (post-fix); falls back to raw
    ``pre_trade_price`` otherwise. Returns (cost, edge, trader_side_wins_yes).
    """
    if "pre_yes_price_corrected" in test.columns:
        p_yes = test["pre_yes_price_corrected"].values
    else:
        p_yes = test["pre_trade_price"].values
    side_buy = test["side_buy"].values
    outcome_yes = test["outcome_yes"].values
    trader_side_wins_yes = side_buy * outcome_yes + (1 - side_buy) * (1 - outcome_yes)
    cost = np.where(trader_side_wins_yes == 1, p_yes, 1 - p_yes)
    cost = np.clip(cost, cost_floor, 1.0 - cost_floor)
    edge = p_hat - cost
    return cost, edge, trader_side_wins_yes


# ---------------------------------------------------------------------------
# Strategy masks
# ---------------------------------------------------------------------------


def general_ev_rule(edge: np.ndarray, edge_min: float = 0.02) -> np.ndarray:
    """Bet whenever edge > edge_min."""
    return edge > edge_min


def home_run_rule(
    edge: np.ndarray,
    cost: np.ndarray,
    time_to_deadline_sec: np.ndarray,
    edge_min: float = 0.20,
    cost_max: float = 0.30,
    time_max_sec: float = 6 * 3600,
) -> np.ndarray:
    """High-edge late-hour cheap-side bets (asymmetric-information hypothesis).

    Tests the Magamyman-style insider pattern: large edge AND low cost AND
    close to event deadline. Empirically: zero hits on test cohort under v3.5
    (D-031, B1b finding).
    """
    return (edge > edge_min) & (cost < cost_max) & (time_to_deadline_sec < time_max_sec)


def top_k_mask(scores: np.ndarray, k_pct: float) -> np.ndarray:
    """Top-K% mask by descending score."""
    n = len(scores)
    k = max(1, int(n * k_pct))
    mask = np.zeros(n, dtype=bool)
    mask[np.argsort(scores)[-k:]] = True
    return mask
