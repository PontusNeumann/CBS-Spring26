"""Enrich taker wallets with on-chain activity history (causal-ready).

Reads taker wallet list from data/iran_strike_labeled.parquet, queries
Etherscan V2 `account/tokentx` endpoint for each wallet's full ERC-20 transfer
history on Polygon, and saves a rich per-wallet output:

  Time-invariant scalars (safe to use on any trade row for the wallet):
    - polygon_first_tx_ts                 first ERC-20 event timestamp
    - funded_by_cex (int)                 1 if first inbound USDC came from a known CEX hot wallet
    - cex_label (str)                     which CEX (binance/coinbase/kraken/okx/bybit/null)
    - first_usdc_inbound_ts               timestamp of first USDC received
    - first_usdc_inbound_amount_usd       amount of first USDC inflow

  Timestamp arrays (bisected at trade-time in build_dataset.py for strict
  causal per-trade features):
    - outbound_ts                 sorted list of outbound ERC-20 tx timestamps
    - inbound_ts                  sorted list of inbound ERC-20 tx timestamps
    - cex_deposit_ts              sorted list of timestamps receiving USDC from CEX
    - cex_deposit_amounts_usd     amounts paired with cex_deposit_ts

Runtime: 6 concurrent workers, 3 Etherscan keys (2 workers per key via
round-robin). HTTP keep-alive via per-thread requests.Session. Per-call
pacing keeps each key under its 5-rps free-tier cap. Expected wall-clock
~60 minutes for 55k wallets.

Resumable: checkpoints to parquet every 500 wallets. Re-running skips already
enriched wallets. Progress snapshot written to
data/enrichment_progress.json every 50 wallets for monitoring.

Requires ETHERSCAN_API_KEYS=key1,key2,key3 in .env at project root.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

_keys_raw = os.environ.get("ETHERSCAN_API_KEYS", "")
KEYS = [k.strip() for k in _keys_raw.split(",") if k.strip()]
if len(KEYS) < 1:
    sys.exit("ETHERSCAN_API_KEYS not set in .env (comma-separated). Abort.")

API_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 137  # Polygon

WORKERS_PER_KEY = 3  # 9 workers total when 3 keys present
N_WORKERS = max(1, len(KEYS) * WORKERS_PER_KEY)
PER_KEY_INTERVAL = 0.22  # 1/0.22 = 4.55 rps per key — just under 5 rps server cap

# Per-key rate limiter: one lock + next-allowed-time per key. All workers
# sharing a key queue on its slot. Enforces the 5 rps server cap regardless
# of how many workers are using the key.
_key_locks = {k: threading.Lock() for k in KEYS}
_key_next_allowed = {k: 0.0 for k in KEYS}

LABELED_IN = ROOT / "data" / "iran_strike_labeled.parquet"
OUT_PATH = ROOT / "data" / "wallet_enrichment.parquet"
STATUS_PATH = ROOT / "data" / "enrichment_progress.json"
LOG_PATH = ROOT / "data" / "enrichment.log"

# USDC contracts on Polygon
USDC_NATIVE = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"  # Circle native
USDC_BRIDGED = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"  # bridged from Ethereum
USDC_ADDRESSES = {USDC_NATIVE, USDC_BRIDGED}

# Known CEX Polygon hot-wallet addresses. Curated starter set — list can be
# expanded in future work. Coverage limitation documented in Methodology.
CEX_ADDRESSES: dict[str, str] = {
    # Binance
    "0xf977814e90da44bfa03b6295a0616a897441acec": "binance",
    "0x28c6c06298d514db089934071355e5743bf21d60": "binance",
    "0x5a52e96bacdabb82fd05763e25335261b270efcb": "binance",
    "0xe2fc31f816a9b94326492132018c3aecc4a93ae1": "binance",
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "coinbase",
    "0x503828976d22510aad0201ac7ec88293211d23da": "coinbase",
    "0xa910f92acdaf488fa6ef02174fb86208ad7722ba": "coinbase",
    "0x3cd751e6b0078be393132286c442345e5dc49699": "coinbase",
    # Kraken
    "0xe92d1a43df510f82c66382592a047d288f85226f": "kraken",
    "0x2b5634c42055806a59e9107ed44d43c426e58258": "kraken",
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": "kraken",
    # OKX
    "0x6e63c8afda2a0f309dfa32b5929f6d2b775a0c7d": "okx",
    "0x5041ed759dd4afc3a72b8192c143f72f4724081a": "okx",
    # Bybit
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40": "bybit",
    "0xee5b5b923ffce93a870b3104b7ca09c3db80047a": "bybit",
}
CEX_ADDRESSES = {a.lower(): l for a, l in CEX_ADDRESSES.items()}


# Per-thread session (HTTP keep-alive) and assigned key
_thread_local = threading.local()


def _get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
        _thread_local.last_call = 0.0
    return sess


def _thread_key() -> str:
    """Assign this thread a dedicated key based on thread id mod n_keys."""
    key = getattr(_thread_local, "api_key", None)
    if key is None:
        tid = threading.get_ident()
        key = KEYS[tid % len(KEYS)]
        _thread_local.api_key = key
    return key


def _acquire_key_slot(key: str) -> None:
    """Per-key rate limiter: block until this key's next slot is free.

    Sets the next-allowed-time to (now + PER_KEY_INTERVAL) so the next
    caller for this key waits. Single lock per key ensures all workers
    that share a key queue up strictly against its 5-rps cap.
    """
    with _key_locks[key]:
        now = time.time()
        wait = _key_next_allowed[key] - now
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        _key_next_allowed[key] = now + PER_KEY_INTERVAL


def fetch_tokentx(wallet: str, retries: int = 5) -> list[dict]:
    """Return token-transfer history for a wallet, sorted ascending by timestamp."""
    sess = _get_session()
    key = _thread_key()
    last_err: Exception | None = None
    for attempt in range(retries):
        _acquire_key_slot(key)
        try:
            r = sess.get(
                API_URL,
                params={
                    "chainid": CHAIN_ID,
                    "module": "account",
                    "action": "tokentx",
                    "address": wallet,
                    "startblock": 0,
                    "endblock": 99999999,
                    "sort": "asc",
                    "apikey": key,
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            status = str(data.get("status", ""))
            msg = data.get("message", "")
            if status == "1":
                return data.get("result", [])
            # Empty history returns status=0 with a "No transactions" message.
            if "No transactions" in msg or status == "0":
                result = data.get("result", [])
                if isinstance(result, list):
                    return result
            raise RuntimeError(f"API status {status}: {msg!r}")
        except Exception as e:
            last_err = e
            # Exponential-ish backoff on failure
            time.sleep(1.0 + attempt * 2)
    raise RuntimeError(f"giving up on {wallet}: {last_err}")


def analyse(wallet: str, txs: list[dict]) -> dict:
    """Derive per-wallet rich output from token-transfer list.

    Returns scalars (time-invariant) + timestamp arrays (for bisect-based
    per-trade features downstream).
    """
    w = wallet.lower()

    outbound_ts: list[int] = []
    inbound_ts: list[int] = []
    cex_deposit_ts: list[int] = []
    cex_deposit_amounts: list[float] = []

    first_tx_ts: int | None = None
    first_usdc_inbound_ts: int | None = None
    first_usdc_inbound_amount: float | None = None
    funded_by_cex = 0
    cex_label: str | None = None

    for tx in txs:
        try:
            ts = int(tx["timeStamp"])
        except (KeyError, ValueError, TypeError):
            continue
        if first_tx_ts is None:
            first_tx_ts = ts
        frm = tx.get("from", "").lower()
        to = tx.get("to", "").lower()
        contract = tx.get("contractAddress", "").lower()

        if frm == w:
            outbound_ts.append(ts)
        elif to == w:
            inbound_ts.append(ts)

            if contract in USDC_ADDRESSES:
                try:
                    decimals = int(tx.get("tokenDecimal", 6) or 6)
                    amt = int(tx.get("value", 0)) / (10**decimals)
                except (ValueError, TypeError):
                    amt = 0.0

                if first_usdc_inbound_ts is None:
                    first_usdc_inbound_ts = ts
                    first_usdc_inbound_amount = amt
                    label = CEX_ADDRESSES.get(frm)
                    if label is not None:
                        funded_by_cex = 1
                        cex_label = label

                if frm in CEX_ADDRESSES:
                    cex_deposit_ts.append(ts)
                    cex_deposit_amounts.append(amt)

    return {
        "wallet": wallet,
        "polygon_first_tx_ts": first_tx_ts,
        "funded_by_cex": funded_by_cex,
        "cex_label": cex_label,
        "first_usdc_inbound_ts": first_usdc_inbound_ts,
        "first_usdc_inbound_amount_usd": first_usdc_inbound_amount,
        "outbound_ts": outbound_ts,
        "inbound_ts": inbound_ts,
        "cex_deposit_ts": cex_deposit_ts,
        "cex_deposit_amounts_usd": cex_deposit_amounts,
        "n_tokentx": len(txs),
        "fetch_status": "ok" if txs else "empty",
    }


def process_wallet(wallet: str) -> dict:
    """Worker entry point: fetch + analyse a single wallet."""
    try:
        txs = fetch_tokentx(wallet)
        return analyse(wallet, txs)
    except Exception as e:
        return {
            "wallet": wallet,
            "polygon_first_tx_ts": None,
            "funded_by_cex": None,
            "cex_label": None,
            "first_usdc_inbound_ts": None,
            "first_usdc_inbound_amount_usd": None,
            "outbound_ts": [],
            "inbound_ts": [],
            "cex_deposit_ts": [],
            "cex_deposit_amounts_usd": [],
            "n_tokentx": 0,
            "fetch_status": f"error: {e}",
        }


def save_checkpoint(records: list[dict]) -> None:
    """Atomic write to parquet (tmp file + rename)."""
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    pd.DataFrame(records).to_parquet(tmp, index=False)
    tmp.replace(OUT_PATH)


def write_progress(completed: int, total: int, failures: int, start_ts: float) -> None:
    elapsed = time.time() - start_ts
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = max(0, total - completed)
    eta = remaining / rate if rate > 0 else None
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_wallets": total,
        "completed": completed,
        "remaining": remaining,
        "pct_done": round(100 * completed / total, 2) if total else 0,
        "failures": failures,
        "rate_per_sec": round(rate, 2),
        "workers": N_WORKERS,
        "keys_in_use": len(KEYS),
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
    print(
        f"n_keys={len(KEYS)}, n_workers={N_WORKERS}, per-key interval={PER_KEY_INTERVAL}s (={1 / PER_KEY_INTERVAL:.1f} rps/key)"
    )

    # Load wallet list
    df = pd.read_parquet(LABELED_IN, columns=["wallet"])
    wallets = df["wallet"].dropna().unique().tolist()
    print(f"{len(wallets):,} unique taker wallets to enrich")

    # Resume support
    done_records: list[dict] = []
    done_set: set[str] = set()
    if OUT_PATH.exists():
        prev = pd.read_parquet(OUT_PATH)
        done_records = prev.to_dict("records")
        done_set = set(prev["wallet"])
        print(
            f"resuming: {len(done_set):,} already enriched, "
            f"{len(wallets) - len(done_set):,} remaining"
        )

    todo = [w for w in wallets if w not in done_set]
    if not todo:
        print("nothing to do — all wallets enriched")
        return

    total = len(wallets)
    failures = 0
    start_ts = time.time()

    # Thread pool — each worker has its own key + session (lazy init per thread)
    pbar = tqdm(total=len(todo), desc="enrich", smoothing=0.05)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_wallet, w): w for w in todo}
        for i, fut in enumerate(as_completed(futures)):
            rec = fut.result()
            done_records.append(rec)
            if rec.get("fetch_status", "").startswith("error"):
                failures += 1
            pbar.update(1)

            if (i + 1) % 50 == 0:
                write_progress(len(done_records), total, failures, start_ts)
                pbar.set_postfix(fails=failures)
            if (i + 1) % 500 == 0:
                save_checkpoint(done_records)

    # Final save
    save_checkpoint(done_records)
    write_progress(len(done_records), total, failures, start_ts)
    pbar.close()
    print(f"done. wrote {len(done_records):,} rows to {OUT_PATH}")
    print(f"total failures: {failures}")


if __name__ == "__main__":
    main()
