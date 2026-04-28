"""Retry the wallets in `data/wallet_enrichment.parquet` whose fetch_status
is not 'ok'. Replaces those rows in place with the new fetch results.

Reuses `scripts/03_enrich_wallets.py::process_wallet` with the same
key rotation and concurrency. Atomic write via tmp-rename so a crash
mid-retry leaves the existing parquet intact.

Run:
    python pontus/scripts/retry_failed_wallets.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
ENRICH_SCRIPT = ROOT / "scripts" / "03_enrich_wallets.py"
OUT_PATH = ROOT / "data" / "wallet_enrichment.parquet"
STATUS_PATH = ROOT / "data" / "enrichment_progress.retry.json"


def _load_canonical_module():
    spec = importlib.util.spec_from_file_location("enrich_wallets_canonical", ENRICH_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_status(completed: int, total: int, recovered: int, start_ts: float, n_workers: int, n_keys: int) -> None:
    elapsed = time.time() - start_ts
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = max(0, total - completed)
    eta = remaining / rate if rate > 0 else None
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "retry_failed",
        "total_wallets": total,
        "completed": completed,
        "remaining": remaining,
        "pct_done": round(100 * completed / total, 2) if total else 0,
        "recovered_so_far": recovered,
        "rate_per_sec": round(rate, 2),
        "workers": n_workers,
        "keys_in_use": n_keys,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_human": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m",
        "eta_seconds": round(eta, 1) if eta else None,
        "eta_human": (
            f"{int(eta // 3600)}h {int((eta % 3600) // 60)}m" if eta else None
        ),
    }
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(STATUS_PATH)


def main() -> None:
    ew = _load_canonical_module()
    print(
        f"n_keys={len(ew.KEYS)}, n_workers={ew.N_WORKERS}, "
        f"per-key interval={ew.PER_KEY_INTERVAL}s",
        flush=True,
    )

    print("loading existing enrichment...", flush=True)
    df = pd.read_parquet(OUT_PATH)
    failed_mask = df["fetch_status"] != "ok"
    failed = df.loc[failed_mask].copy()
    if failed.empty:
        print("nothing to retry — all rows already ok.", flush=True)
        return

    todo = failed["wallet"].tolist()
    print(f"  retrying {len(todo):,} previously-failed wallets", flush=True)

    new_records: list[dict] = []
    total = len(todo)
    recovered = 0
    start_ts = time.time()

    pbar = tqdm(total=total, desc="retry", smoothing=0.05)
    with ThreadPoolExecutor(max_workers=ew.N_WORKERS) as exc:
        futures = {exc.submit(ew.process_wallet, w): w for w in todo}
        for i, fut in enumerate(as_completed(futures)):
            rec = fut.result()
            new_records.append(rec)
            if not rec.get("fetch_status", "").startswith("error"):
                recovered += 1
            pbar.update(1)
            if (i + 1) % 50 == 0:
                _write_status(i + 1, total, recovered, start_ts, ew.N_WORKERS, len(ew.KEYS))
                pbar.set_postfix(recovered=recovered)

    _write_status(total, total, recovered, start_ts, ew.N_WORKERS, len(ew.KEYS))
    pbar.close()
    elapsed = time.time() - start_ts
    print(
        f"\nretry complete: {recovered:,} of {total:,} recovered "
        f"({recovered/total*100:.1f}%) in {elapsed:.0f}s",
        flush=True,
    )

    # Replace rows: drop the previously-failed wallets, append the new results.
    new_df = pd.DataFrame(new_records).reindex(columns=df.columns)
    keep_df = df.loc[~failed_mask]
    combined = pd.concat([keep_df, new_df], ignore_index=True)

    # Sanity: one row per wallet, addresses preserved
    assert len(combined) == len(df), (
        f"row count drift: was {len(df):,}, now {len(combined):,}"
    )
    assert combined["wallet"].nunique() == len(combined), "duplicate wallets after retry"

    tmp = OUT_PATH.with_suffix(".parquet.retry_tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(OUT_PATH)
    print(f"wrote {OUT_PATH.relative_to(ROOT)} ({len(combined):,} rows)")

    final_ok = (combined["fetch_status"] == "ok").sum()
    final_err = len(combined) - final_ok
    print(f"final state: {final_ok:,} ok, {final_err:,} still failing")


if __name__ == "__main__":
    main()
