"""
24_v2_residualised.py

Runs the v2 stacking pipeline with **per-market expanding-mean feature
residualisation** applied before any imputation / scaling.

Intervention (Option 1 in the split-strategy follow-up):

  For each continuous feature X and each trade at time t in market m,
  replace `X[t]` with:
      X_resid[t] = X[t] − E[X | market = m, timestamp < t]

  The running mean is an `expanding().mean().shift(1)` per
  `condition_id`, so the subtracted quantity is strictly prior to t and
  observable in a live deployment. Binary indicators and the
  already-normalised `pct_time_elapsed` / `wallet_spread_ratio` columns
  are left unchanged.

Purpose: the 22-Apr market-identity audit reported a 74-class top-1
accuracy of 49.96 %, a 37× uplift over random. Residualisation **removes
the market-identifying channel by construction** (the residual has zero
per-market mean), while preserving within-market variation — the piece
of the signal that is relevant to predicting a specific trade's
`bet_correct`. The rerun here answers the direct question: does the
cross-family test ROC of 0.579 hold once market identity is stripped?

Pipeline stages (identical to 22_v2_pipeline.py):
  load + residualise + winsorise + impute + scale
  GroupKFold OOF stacking (LogReg + calibrated RF + MLP → LogReg meta)
  K-fold val ECE calibrator picker (isotonic vs Platt)
  residual-edge, permutation importance, backtests, bootstrap CIs
  autoencoder, isolation forest, overlap, magamyman

Outputs land in `pontus/outputs/v2_residualised/` so the original
`pontus/outputs/v2/` numbers remain intact for comparison.

Framework: tf.keras + sklearn + pandas / numpy (CBS MLDP compliant).

Approximate runtime: 12-15 min (residualisation adds ~1 min up-front).

Usage:
  caffeinate -i python pontus/scripts/24_v2_residualised.py \\
      2>&1 | tee pontus/outputs/v2_residualised/run.log
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import V2 (which pulls V1 transitively). We then monkey-patch the cohort
# loader to residualise features before V2's main() pipeline consumes them.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
V2_PATH = ROOT / "pontus" / "scripts" / "22_v2_pipeline.py"

_spec = importlib.util.spec_from_file_location("pipeline_v2", V2_PATH)
V2 = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_v2"] = V2
_spec.loader.exec_module(V2)
V1 = V2.V1


# Features that should NOT be residualised. Binary or already bounded in
# [0, 1] per-market by construction, so the residualisation would either
# destroy semantic meaning (binary flags) or be redundant (already zero-
# mean per market).
SKIP_RESIDUALISATION = {
    # Already [0, 1] normalised per market
    "pct_time_elapsed",
    # Symmetric min/max ratio in [0, 1]
    "wallet_spread_ratio",
    # Binary indicators
    "wallet_is_burst",
    "wallet_is_whale_in_market",
    "wallet_funded_by_cex_scoped",
    "wallet_enriched",
    "wallet_has_prior_trades",
    "wallet_has_prior_trades_in_market",
    "wallet_has_cross_market_history",
    "market_timing_known",
    "wallet_has_resolved_priors",
}


def residualise_cohort(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Per-market expanding-mean residualisation with shift(1).

    The running mean at row i of market m is the mean of feature X over
    rows in market m with timestamp strictly less than i's timestamp —
    observable at trade time and therefore causal. NaN on the first row
    of each (market, feature) pair; filled with 0 (the expected residual
    value).

    Returns a copy of `df` with residualised feature columns; other
    columns (timestamps, IDs, labels, `market_implied_prob`, etc.) are
    passed through unchanged.
    """
    df = df.sort_values(["condition_id", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )
    out = df.copy()
    for feat in features:
        if feat in SKIP_RESIDUALISATION:
            continue
        x = pd.to_numeric(df[feat], errors="coerce")
        # groupby.transform avoids the `.apply` slowdown on expanding().mean
        running_mean = x.groupby(df["condition_id"]).transform(
            lambda s: s.expanding().mean().shift(1)
        )
        residual = x - running_mean
        out[feat] = residual.fillna(0.0)
    return out


def make_residualised_loader():
    """Build a replacement `load_cohorts` that residualises the three cohort
    frames right after disk read. Preserves the V1.Cohorts signature."""
    original_loader = V1.load_cohorts

    def _load_and_residualise() -> V1.Cohorts:
        cohorts = original_loader()
        feats = cohorts.features
        n_res = sum(1 for f in feats if f not in SKIP_RESIDUALISATION)
        n_skip = len(feats) - n_res
        print(
            f"[residualise] {n_res} continuous features residualised; "
            f"{n_skip} skipped (binary / already-normalised)"
        )
        t0 = time.time()
        tr = residualise_cohort(cohorts.train, feats)
        va = residualise_cohort(cohorts.val, feats)
        te = residualise_cohort(cohorts.test, feats)
        print(f"[residualise] done in {time.time() - t0:.1f}s")
        return V1.Cohorts(tr, va, te, feats)

    return _load_and_residualise


def main() -> None:
    # Redirect outputs before V2.main() captures the path
    V2.OUT_DIR = ROOT / "pontus" / "outputs" / "v2_residualised"
    V2.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Patch the loader in both V1 and V2 namespaces (V2 references V1.load_cohorts)
    new_loader = make_residualised_loader()
    V1.load_cohorts = new_loader
    V2.V1.load_cohorts = new_loader

    print(f"[out] {V2.OUT_DIR}")
    V2.main()


if __name__ == "__main__":
    main()
