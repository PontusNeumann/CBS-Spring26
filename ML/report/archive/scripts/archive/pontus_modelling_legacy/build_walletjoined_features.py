"""Bolt Layer 6 wallet-identity features onto Alex's idea1 cohort.

Mirror of `scripts/11_add_layer6.py` adapted for Alex's parquet schema:
  - join key:  taker (HF) -> wallet (enrichment), lowercase both sides
  - timestamp: uint64 unix seconds (no ISO parsing needed)
  - inputs:    alex/data/{train,test}_features.parquet  +  data/wallet_enrichment.parquet
  - outputs:   pontus/data/{train,test}_features_walletjoined.parquet

Adds the same 12 columns as 11_add_layer6.py:
  wallet_enriched, wallet_polygon_age_at_t_days, wallet_polygon_nonce_at_t,
  wallet_log_polygon_nonce_at_t, wallet_n_inbound_at_t, wallet_log_n_inbound_at_t,
  wallet_n_cex_deposits_at_t, wallet_cex_usdc_cumulative_at_t, wallet_log_cex_usdc_cum,
  days_from_first_usdc_to_t, wallet_funded_by_cex, wallet_funded_by_cex_scoped.

Causal contract: every per-trade feature uses only wallet events strictly before
the trade's `timestamp` (np.searchsorted side='left'). NaN preserved for un-enriched
rows on continuous features; binary flags default to 0 for un-enriched.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ALEX_DATA = ROOT / "alex" / "data"
ENRICH_IN = ROOT / "data" / "wallet_enrichment.parquet"
OUT_DIR = ROOT / "pontus" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WALLET_COL_HF = "taker"
TIMESTAMP_COL = "timestamp"

NEW_COLS = [
    "wallet_enriched",
    "wallet_polygon_age_at_t_days",
    "wallet_polygon_nonce_at_t",
    "wallet_log_polygon_nonce_at_t",
    "wallet_n_inbound_at_t",
    "wallet_log_n_inbound_at_t",
    "wallet_n_cex_deposits_at_t",
    "wallet_cex_usdc_cumulative_at_t",
    "wallet_log_cex_usdc_cum",
    "days_from_first_usdc_to_t",
    "wallet_funded_by_cex",
    "wallet_funded_by_cex_scoped",
]

LAYER6_NUMERIC_COLS = [
    "wallet_polygon_age_at_t_days",
    "wallet_polygon_nonce_at_t",
    "wallet_log_polygon_nonce_at_t",
    "wallet_n_inbound_at_t",
    "wallet_log_n_inbound_at_t",
    "wallet_n_cex_deposits_at_t",
    "wallet_cex_usdc_cumulative_at_t",
    "wallet_log_cex_usdc_cum",
    "days_from_first_usdc_to_t",
]


def build_wallet_index(enrich_ok: pd.DataFrame) -> dict:
    """Index enrichment rows by lowercase wallet address. O(n) over wallets."""
    idx: dict[str, dict] = {}
    for _, r in enrich_ok.iterrows():
        idx[r["wallet"].lower()] = {
            "polygon_first_tx_ts": int(r["polygon_first_tx_ts"])
            if not pd.isna(r["polygon_first_tx_ts"])
            else None,
            "first_usdc_inbound_ts": int(r["first_usdc_inbound_ts"])
            if not pd.isna(r["first_usdc_inbound_ts"])
            else None,
            "funded_by_cex": int(r["funded_by_cex"])
            if not pd.isna(r["funded_by_cex"])
            else 0,
            "outbound_ts": np.asarray(r["outbound_ts"], dtype=np.int64)
            if len(r["outbound_ts"])
            else np.array([], dtype=np.int64),
            "inbound_ts": np.asarray(r["inbound_ts"], dtype=np.int64)
            if len(r["inbound_ts"])
            else np.array([], dtype=np.int64),
            "cex_deposit_ts": np.asarray(r["cex_deposit_ts"], dtype=np.int64)
            if len(r["cex_deposit_ts"])
            else np.array([], dtype=np.int64),
            "cex_deposit_amounts_cumsum": np.cumsum(
                np.asarray(r["cex_deposit_amounts_usd"], dtype=np.float64)
            )
            if len(r["cex_deposit_amounts_usd"])
            else np.array([], dtype=np.float64),
        }
    return idx


def compute_layer6(features_df: pd.DataFrame, raw_df: pd.DataFrame, wallet_idx: dict) -> pd.DataFrame:
    """Compute the 12 Layer-6 features for one cohort split.

    `features_df` carries the 70 engineered features but no `taker` column;
    `raw_df` is the matching trade-level parquet from which we lift `taker`.
    Both must align row-for-row (same source order). Verified via assertion.
    """
    assert len(features_df) == len(raw_df), (
        f"row mismatch: features {len(features_df):,} vs raw {len(raw_df):,}"
    )

    n = len(features_df)
    polygon_age_at_t = np.full(n, np.nan, dtype=np.float64)
    polygon_nonce_at_t = np.zeros(n, dtype=np.int32)
    n_inbound_at_t = np.zeros(n, dtype=np.int32)
    n_cex_deposits_at_t = np.zeros(n, dtype=np.int32)
    cex_usdc_cum_at_t = np.zeros(n, dtype=np.float64)
    days_from_first_usdc = np.full(n, np.nan, dtype=np.float64)
    funded_by_cex_static = np.zeros(n, dtype=np.int8)
    funded_by_cex_scoped = np.zeros(n, dtype=np.int8)
    wallet_enriched = np.zeros(n, dtype=np.int8)

    wallets = raw_df[WALLET_COL_HF].str.lower().values
    timestamps = raw_df[TIMESTAMP_COL].astype(np.int64).to_numpy()

    t0 = time.time()
    for i in range(n):
        w = wallets[i]
        t_trade = timestamps[i]
        info = wallet_idx.get(w)
        if info is None:
            continue
        wallet_enriched[i] = 1

        if info["polygon_first_tx_ts"] is not None:
            age_s = max(0, t_trade - info["polygon_first_tx_ts"])
            polygon_age_at_t[i] = age_s / 86400.0

        ob = info["outbound_ts"]
        if len(ob) > 0:
            polygon_nonce_at_t[i] = int(np.searchsorted(ob, t_trade, side="left"))

        ib = info["inbound_ts"]
        if len(ib) > 0:
            n_inbound_at_t[i] = int(np.searchsorted(ib, t_trade, side="left"))

        cd = info["cex_deposit_ts"]
        if len(cd) > 0:
            k = int(np.searchsorted(cd, t_trade, side="left"))
            n_cex_deposits_at_t[i] = k
            if k > 0:
                cex_usdc_cum_at_t[i] = float(info["cex_deposit_amounts_cumsum"][k - 1])

        if info["first_usdc_inbound_ts"] is not None:
            d = (t_trade - info["first_usdc_inbound_ts"]) / 86400.0
            if d >= 0:
                days_from_first_usdc[i] = d

        funded_by_cex_static[i] = info["funded_by_cex"]
        if (
            info["funded_by_cex"] == 1
            and info["first_usdc_inbound_ts"] is not None
            and info["first_usdc_inbound_ts"] < t_trade
        ):
            funded_by_cex_scoped[i] = 1

        if (i + 1) % 200_000 == 0:
            print(
                f"    {i + 1:,} / {n:,} ({(i + 1) / n * 100:.1f}%, {time.time() - t0:.0f}s)"
            )

    out = features_df.copy()
    out["wallet_enriched"] = wallet_enriched
    out["wallet_polygon_age_at_t_days"] = polygon_age_at_t
    out["wallet_polygon_nonce_at_t"] = polygon_nonce_at_t.astype(np.float64)
    out["wallet_log_polygon_nonce_at_t"] = np.log1p(polygon_nonce_at_t).astype(np.float64)
    out["wallet_n_inbound_at_t"] = n_inbound_at_t.astype(np.float64)
    out["wallet_log_n_inbound_at_t"] = np.log1p(n_inbound_at_t).astype(np.float64)
    out["wallet_n_cex_deposits_at_t"] = n_cex_deposits_at_t.astype(np.float64)
    out["wallet_cex_usdc_cumulative_at_t"] = cex_usdc_cum_at_t
    out["wallet_log_cex_usdc_cum"] = np.log1p(cex_usdc_cum_at_t)
    out["days_from_first_usdc_to_t"] = days_from_first_usdc
    out["wallet_funded_by_cex"] = funded_by_cex_static
    out["wallet_funded_by_cex_scoped"] = funded_by_cex_scoped

    unenriched = out["wallet_enriched"] == 0
    out.loc[unenriched, LAYER6_NUMERIC_COLS] = np.nan

    return out


def main() -> None:
    t_total = time.time()

    print("loading wallet enrichment...")
    enrich = pd.read_parquet(ENRICH_IN)
    enrich_ok = enrich[enrich["fetch_status"] == "ok"].copy()
    print(f"  {len(enrich):,} total, {len(enrich_ok):,} ok ({len(enrich_ok)/len(enrich)*100:.1f}%)")

    print("indexing enrichment by lowercase wallet...")
    t1 = time.time()
    wallet_idx = build_wallet_index(enrich_ok)
    print(f"  indexed {len(wallet_idx):,} wallets in {time.time() - t1:.1f}s")

    for split in ("train", "test"):
        print(f"\n=== {split} ===")
        raw = pd.read_parquet(ALEX_DATA / f"{split}.parquet", columns=[WALLET_COL_HF, TIMESTAMP_COL])
        feats = pd.read_parquet(ALEX_DATA / f"{split}_features.parquet")
        print(f"  raw rows: {len(raw):,}  features rows: {len(feats):,}  features cols: {len(feats.columns)}")

        t2 = time.time()
        out = compute_layer6(feats, raw, wallet_idx)
        print(f"  computed Layer-6 in {time.time() - t2:.1f}s, output cols: {len(out.columns)}")

        cov_wallet = out["wallet_enriched"].mean()
        print(
            f"  coverage: {int(out['wallet_enriched'].sum()):,} / {len(out):,} "
            f"({cov_wallet:.1%}) trades enriched"
        )
        print(f"  funded_by_cex_scoped sum: {int(out['wallet_funded_by_cex_scoped'].sum()):,}")
        print(f"  median wallet_polygon_age (enriched): {out.loc[out.wallet_enriched==1, 'wallet_polygon_age_at_t_days'].median():.1f} days")

        out_path = OUT_DIR / f"{split}_features_walletjoined.parquet"
        out.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  wrote {out_path.relative_to(ROOT)}  ({size_mb:.1f} MB)")

    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
