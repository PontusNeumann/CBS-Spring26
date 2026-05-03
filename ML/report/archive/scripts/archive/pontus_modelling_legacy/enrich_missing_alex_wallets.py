"""Enrich the wallets that Alex's idea1 cohort needs but our existing
`data/wallet_enrichment.parquet` doesn't yet cover.

Reuses `scripts/03_enrich_wallets.py` (canonical thread pool, retry, key
rotation) without modifying it. New rows are appended to the canonical
parquet via an atomic tmp-rename. Existing rows are left untouched.

Run:
    python pontus/scripts/enrich_missing_alex_wallets.py

Status:
    data/enrichment_progress.alex_extras.json (updated every 50 wallets)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
ENRICH_SCRIPT = ROOT / "scripts" / "03_enrich_wallets.py"
ALEX_TRAIN = ROOT / "alex" / "data" / "train.parquet"
ALEX_TEST = ROOT / "alex" / "data" / "test.parquet"
OUT_PATH = ROOT / "data" / "wallet_enrichment.parquet"
STATUS_PATH = ROOT / "data" / "enrichment_progress.alex_extras.json"
LOG_PATH = ROOT / "data" / "enrichment_alex_extras.log"


def _load_canonical_module():
    spec = importlib.util.spec_from_file_location("enrich_wallets_canonical", ENRICH_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _missing_wallets() -> list[str]:
    print("scanning Alex cohort takers...", flush=True)
    takers = pd.concat(
        [
            pd.read_parquet(ALEX_TRAIN, columns=["taker"]).taker,
            pd.read_parquet(ALEX_TEST, columns=["taker"]).taker,
        ]
    )
    # Preserve Alex's case (checksum). We do case-insensitive de-dup against
    # the existing enrichment table.
    alex_unique = takers.drop_duplicates().reset_index(drop=True)
    alex_lc_to_first_seen = (
        alex_unique.to_frame("taker")
        .assign(taker_lc=alex_unique.str.lower())
        .drop_duplicates("taker_lc")
    )
    print(f"  alex unique takers: {len(alex_lc_to_first_seen):,}", flush=True)

    existing = pd.read_parquet(OUT_PATH, columns=["wallet"])
    existing_lc = set(existing.wallet.str.lower())
    print(f"  existing enriched wallets: {len(existing_lc):,}", flush=True)

    missing = alex_lc_to_first_seen[~alex_lc_to_first_seen.taker_lc.isin(existing_lc)]
    out = missing.taker.tolist()
    print(f"  missing → to enrich: {len(out):,}", flush=True)
    return out


def _write_status(completed: int, total: int, failures: int, start_ts: float, n_workers: int, n_keys: int) -> None:
    elapsed = time.time() - start_ts
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = max(0, total - completed)
    eta = remaining / rate if rate > 0 else None
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "alex_extras",
        "total_wallets": total,
        "completed": completed,
        "remaining": remaining,
        "pct_done": round(100 * completed / total, 2) if total else 0,
        "failures": failures,
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


def _atomic_append(new_rows: list[dict]) -> None:
    """Read existing parquet, concat new rows, write to tmp, rename."""
    existing = pd.read_parquet(OUT_PATH)
    new_df = pd.DataFrame(new_rows)
    # Align column order to existing
    new_df = new_df.reindex(columns=existing.columns)
    combined = pd.concat([existing, new_df], ignore_index=True)
    tmp = OUT_PATH.with_suffix(".parquet.alex_extras_tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(OUT_PATH)


def main() -> None:
    ew = _load_canonical_module()
    print(
        f"n_keys={len(ew.KEYS)}, n_workers={ew.N_WORKERS}, "
        f"per-key interval={ew.PER_KEY_INTERVAL}s",
        flush=True,
    )

    todo = _missing_wallets()
    if not todo:
        print("nothing to enrich — all Alex takers are already covered.", flush=True)
        return

    new_records: list[dict] = []
    total = len(todo)
    failures = 0
    start_ts = time.time()
    last_checkpoint = 0
    CHECKPOINT_EVERY = 500

    pbar = tqdm(total=total, desc="alex_extras", smoothing=0.05)
    with ThreadPoolExecutor(max_workers=ew.N_WORKERS) as exc:
        futures = {exc.submit(ew.process_wallet, w): w for w in todo}
        for i, fut in enumerate(as_completed(futures)):
            rec = fut.result()
            new_records.append(rec)
            if rec.get("fetch_status", "").startswith("error"):
                failures += 1
            pbar.update(1)

            if (i + 1) % 50 == 0:
                _write_status(i + 1, total, failures, start_ts, ew.N_WORKERS, len(ew.KEYS))
                pbar.set_postfix(fails=failures)

            if (i + 1) - last_checkpoint >= CHECKPOINT_EVERY:
                _atomic_append(new_records[last_checkpoint:])
                last_checkpoint = i + 1

    if last_checkpoint < len(new_records):
        _atomic_append(new_records[last_checkpoint:])

    _write_status(len(new_records), total, failures, start_ts, ew.N_WORKERS, len(ew.KEYS))
    pbar.close()
    print(
        f"\ndone. enriched {len(new_records):,} new wallets "
        f"(failures: {failures}) in {time.time() - start_ts:.0f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
