"""
dashboard.py — live dashboard for the model sweep.

Run:
    .venv/bin/python alex/scripts/dashboard.py
    open http://localhost:9876

Stops with Ctrl+C. No Flask / no extra deps — stdlib http.server only.

Polls /api/status every 4s. /api/cohort cached at startup (slow read).

What it shows:
  - Top-line status: PID, elapsed, ETA, X/8 models complete
  - Cohort summary card (sizes, markets, class balance, features, cutoffs)
  - Best-model spotlight
  - Per-model rows with fold-AUC sparkline + top-3 features
  - Comparison table with v1/v2 baselines for reference
  - Per-market breakdown for best supervised model
  - Isolation Forest panel
  - Live log tail
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "outputs" / "sweep_idea1"
LOG_FILE = ROOT / ".scratch" / "sweep.log"
PID_FILE = Path("/tmp/sweep_pid")
DATA_DIR = ROOT / "data"
BASELINE_V1 = ROOT / "outputs" / "baselines" / "idea1_v1" / "metrics.json"
BASELINE_V2 = ROOT / "outputs" / "baselines" / "idea1_v2" / "metrics.json"

EXPECTED_MODELS = [
    "logreg_l2",
    "logreg_l1",
    "decision_tree",
    "random_forest",
    "hist_gbm",
    "pca_logreg",
    "mlp_keras",
    "iso_forest",
]

PORT = 9876


# ---------------------------------------------------------------------------
# Cohort cache (reads parquets once)
# ---------------------------------------------------------------------------


def load_cohort_stats() -> dict:
    """Read once at startup. Heavy parquet reads cached here."""
    try:
        import pandas as pd

        markets = pd.read_parquet(DATA_DIR / "markets_subset.parquet")
        train = pd.read_parquet(
            DATA_DIR / "train_features.parquet", columns=["market_id", "bet_correct"]
        )
        test = pd.read_parquet(
            DATA_DIR / "test_features.parquet", columns=["market_id", "bet_correct"]
        )
        feature_cols = json.loads((DATA_DIR / "feature_cols.json").read_text())
        return {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_markets_train": int(train["market_id"].nunique()),
            "n_markets_test": int(test["market_id"].nunique()),
            "class_balance_train": float(train["bet_correct"].mean()),
            "class_balance_test": float(test["bet_correct"].mean()),
            "n_features": len(feature_cols),
            "train_cutoff": "2026-02-28 06:35 UTC",
            "test_cutoff": "2026-04-07 23:59 UTC",
            "train_event": "Operation Epic Fury (US strike on Iran)",
            "test_event": "Trump announces Iran ceasefire",
            "train_volume_usd": float(
                markets[markets.cohort == "train"]["volume"].sum()
            ),
            "test_volume_usd": float(markets[markets.cohort == "test"]["volume"].sum()),
        }
    except Exception as e:
        return {"error": str(e)}


def load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


COHORT_STATS = load_cohort_stats()
BASELINES = {
    "v1": load_baseline(BASELINE_V1),
    "v2": load_baseline(BASELINE_V2),
}


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def is_running(pid_file: Path) -> tuple[bool, int | None, int]:
    if not pid_file.exists():
        return False, None, 0
    pid_text = pid_file.read_text().strip()
    if not pid_text.isdigit():
        return False, None, 0
    pid = int(pid_text)
    try:
        os.kill(pid, 0)
        ret = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime="],
            capture_output=True,
            text=True,
            timeout=2,
        )
        et = ret.stdout.strip()
        parts = [int(p) for p in et.replace("-", ":").split(":") if p.isdigit()]
        if len(parts) == 2:
            seconds = parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 4:
            seconds = parts[0] * 86400 + parts[1] * 3600 + parts[2] * 60 + parts[3]
        else:
            seconds = 0
        return True, pid, seconds
    except (OSError, ProcessLookupError):
        return False, pid, 0


def load_completed() -> dict[str, dict]:
    out = {}
    if not SWEEP_DIR.exists():
        return out
    for m_dir in SWEEP_DIR.iterdir():
        if not m_dir.is_dir():
            continue
        m_path = m_dir / "metrics.json"
        if not m_path.exists():
            continue
        try:
            data = json.loads(m_path.read_text())
            # Also load top features if available
            fi_path = m_dir / "feature_importance.json"
            if fi_path.exists():
                fi = json.loads(fi_path.read_text())
                # Take top 5 by absolute value
                top = sorted(fi.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                data["_top_features"] = [{"name": k, "value": v} for k, v in top]
            # Per-market
            pm_path = m_dir / "per_market_test.json"
            if pm_path.exists():
                data["_per_market"] = json.loads(pm_path.read_text())
            out[m_dir.name] = data
        except Exception:
            pass
    return out


def get_log_tail(n: int = 40) -> list[str]:
    if not LOG_FILE.exists():
        return []
    try:
        return LOG_FILE.read_text().splitlines()[-n:]
    except Exception:
        return []


def estimate_eta(elapsed: int, n_done: int, n_total: int) -> int | None:
    """Crude ETA: extrapolate from average per-model time so far."""
    if n_done == 0 or elapsed == 0:
        return None
    if n_done >= n_total:
        return 0
    avg_per_model = elapsed / n_done
    # Slow models (RF, MLP) are typically last; weight remaining by 1.5x
    remaining_weight = (n_total - n_done) * 1.5
    return int(avg_per_model * remaining_weight)


def get_status() -> dict:
    running, pid, elapsed = is_running(PID_FILE)
    completed = load_completed()
    log_tail = get_log_tail(40)
    eta = estimate_eta(elapsed, len(completed), len(EXPECTED_MODELS))
    # Find currently-running model from log
    current = None
    for line in reversed(log_tail):
        if "[" in line and "]" in line:
            chunk = line.split("[")[1].split("]")[0]
            if chunk in EXPECTED_MODELS and chunk not in completed:
                current = chunk
                break
    return {
        "running": running,
        "pid": pid,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "completed_models": completed,
        "expected_models": EXPECTED_MODELS,
        "current_running_model": current,
        "log_tail": log_tail,
        "baselines": BASELINES,
        "ts_now": time.time(),
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sweep Dashboard — idea 1</title>
<style>
  :root {
    --green: #34c759;
    --blue: #007aff;
    --red: #ff3b30;
    --grey: #a0a0a5;
    --bg: #f5f5f7;
    --fg: #1c1c1e;
    --card: #ffffff;
    --border: #e5e5ea;
    --highlight: #fff7d6;
  }
  * { box-sizing: border-box; }
  body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 1.5rem; color: var(--fg); background: var(--bg); }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { margin: 0 0 0.5rem; font-size: 1.5rem; }
  h2 { font-size: 0.85rem; margin: 1.5rem 0 0.6rem; color: #6e6e73; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
  .pill-on { background: rgba(52,199,89,0.18); color: #16703a; }
  .pill-off { background: rgba(255,59,48,0.18); color: #8b1f17; }
  .pill-info { background: rgba(0,122,255,0.15); color: #084da0; }
  .meta { color: #6e6e73; font-size: 13px; }
  .num { font-family: ui-monospace, SF Mono, Menlo, monospace; }
  .big-num { font-family: ui-monospace, monospace; font-size: 1.6rem; font-weight: 600; }
  .label { font-size: 11px; color: #6e6e73; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }
  .progress-track { display: grid; grid-template-columns: 22px 150px 1fr 100px; gap: 10px; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); }
  .progress-track:last-child { border-bottom: none; }
  .progress-track .icon { font-size: 14px; text-align: center; }
  .progress-track .name { font-family: ui-monospace, monospace; font-size: 13px; font-weight: 500; }
  .bar { height: 16px; background: #ececef; border-radius: 8px; overflow: hidden; position: relative; }
  .bar-fill { height: 100%; background: var(--green); transition: width 200ms; }
  .bar-fill.pending { background: #ddd; }
  .bar-fill.running { background: var(--blue); animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: 0.5 } }
  .status-text { font-size: 12px; color: #6e6e73; text-align: right; font-family: ui-monospace, monospace; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--border); }
  th { background: #f5f5f7; font-weight: 600; font-size: 12px; color: #6e6e73; text-transform: uppercase; letter-spacing: 0.04em; }
  td.n { font-family: ui-monospace, monospace; text-align: right; }
  tr.best { background: var(--highlight); }
  tr.best td:first-child::before { content: "★ "; color: #c8a000; }
  pre.log { background: #1c1c1e; color: #ddd; padding: 12px 16px; border-radius: 8px; max-height: 320px; overflow-y: auto; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.4; margin: 0; }
  .fold-bars { display: inline-flex; gap: 2px; align-items: end; height: 22px; }
  .fold-bar { width: 8px; background: var(--blue); border-radius: 1px 1px 0 0; }
  .top-feat { display: flex; align-items: center; gap: 6px; font-size: 11px; padding: 1px 0; font-family: ui-monospace, monospace; color: #444; }
  .top-feat .coef { width: 60px; text-align: right; color: #6e6e73; }
  .top-feat .feat-name { flex: 1; }
  .feat-bar { height: 4px; background: var(--blue); border-radius: 2px; }
  .feat-bar.neg { background: var(--red); }
  .delta { font-size: 11px; padding: 1px 6px; border-radius: 4px; font-family: ui-monospace, monospace; }
  .delta-pos { background: rgba(52,199,89,0.18); color: #16703a; }
  .delta-neg { background: rgba(255,59,48,0.18); color: #8b1f17; }
  .footer { margin-top: 2rem; color: #888; font-size: 11px; text-align: center; }
  details { cursor: pointer; }
  details summary { font-weight: 600; padding: 6px 0; }
  .market-cell { font-family: ui-monospace, monospace; font-size: 11px; color: #6e6e73; }
  .auc-cell { display: inline-block; padding: 1px 6px; border-radius: 3px; font-family: ui-monospace, monospace; font-weight: 600; }
  .auc-cell.high { background: rgba(52,199,89,0.18); color: #16703a; }
  .auc-cell.mid { background: rgba(255,204,0,0.18); color: #7a5800; }
  .auc-cell.low { background: rgba(255,59,48,0.18); color: #8b1f17; }
</style>
</head>
<body>
<div class="container">

<h1>Sweep Dashboard <span style="font-weight: 400; color: #888;">— idea 1</span></h1>
<div id="header" class="meta"></div>

<div id="cohort-card" class="card" style="margin-top: 1rem;"></div>

<h2>Top model</h2>
<div id="best-card" class="card"></div>

<h2>Models in flight</h2>
<div class="card" id="progress"></div>

<h2>Comparison <span class="meta" style="font-weight: 400; text-transform: none;">(v3 sweep + v1/v2 baselines for reference)</span></h2>
<div id="table"></div>

<h2>Per-market AUC <span class="meta" style="font-weight: 400; text-transform: none;">(top model)</span></h2>
<div id="per-market" class="card"></div>

<h2>Isolation Forest <span class="meta" style="font-weight: 400; text-transform: none;">(unsupervised insider detector)</span></h2>
<div id="iso" class="card"></div>

<h2>Log <span class="meta" style="font-weight: 400; text-transform: none;">(last 40)</span></h2>
<pre class="log" id="log"></pre>

<div class="footer">
  Auto-refreshes every 4s · sources: <code>alex/outputs/sweep_idea1/</code> + <code>alex/.scratch/sweep.log</code>
</div>
</div>

<script>
function fmtSecs(s) {
  if (s === null || s === undefined) return '—';
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${m}m ${sec}s`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}
function n(x, digits = 3) { return (x === null || x === undefined || isNaN(x)) ? '—' : Number(x).toFixed(digits); }
function fmtMillions(x) { return x === null || x === undefined ? '—' : `$${(x/1e6).toFixed(1)}M`; }
function aucClass(v) {
  if (v == null) return '';
  if (v >= 0.62) return 'high';
  if (v >= 0.55) return 'mid';
  return 'low';
}

let cohort = null;

async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  return r.json();
}

function renderCohort() {
  if (!cohort) return;
  const c = cohort;
  if (c.error) {
    document.getElementById('cohort-card').innerHTML = `<span class="pill pill-off">cohort load error</span> ${c.error}`;
    return;
  }
  document.getElementById('cohort-card').innerHTML = `
    <div class="grid-3">
      <div>
        <div class="label">Train cohort</div>
        <div class="big-num">${(c.n_train/1e6).toFixed(2)}M</div>
        <div class="meta">trades · ${c.n_markets_train} markets · ${fmtMillions(c.train_volume_usd)} volume</div>
        <div class="meta" style="margin-top: 6px;">📅 cutoff: ${c.train_cutoff}<br>🎯 ${c.train_event}</div>
      </div>
      <div>
        <div class="label">Test cohort</div>
        <div class="big-num">${(c.n_test/1e3).toFixed(0)}K</div>
        <div class="meta">trades · ${c.n_markets_test} markets · ${fmtMillions(c.test_volume_usd)} volume</div>
        <div class="meta" style="margin-top: 6px;">📅 cutoff: ${c.test_cutoff}<br>🎯 ${c.test_event}</div>
      </div>
      <div>
        <div class="label">Setup</div>
        <div class="big-num">${c.n_features}</div>
        <div class="meta">features · 5-fold GroupKFold</div>
        <div class="meta" style="margin-top: 6px;">class balance train: ${(c.class_balance_train*100).toFixed(1)}%<br>class balance test: ${(c.class_balance_test*100).toFixed(1)}%</div>
      </div>
    </div>`;
}

function renderHeader(d) {
  const pill = d.running ? '<span class="pill pill-on">RUNNING</span>'
                         : '<span class="pill pill-off">STOPPED</span>';
  const nDone = Object.keys(d.completed_models).length;
  const nTotal = d.expected_models.length;
  const etaTxt = d.eta_seconds !== null ? `ETA ~${fmtSecs(d.eta_seconds)}` : '';
  document.getElementById('header').innerHTML = `
    ${pill} &nbsp; PID ${d.pid ?? '—'} &nbsp; · &nbsp; elapsed ${fmtSecs(d.elapsed_seconds)} &nbsp; · &nbsp; ${etaTxt} &nbsp; · &nbsp; <strong>${nDone} of ${nTotal}</strong> complete${d.current_running_model ? ` (now: <code>${d.current_running_model}</code>)` : ''}`;
}

function renderBest(d) {
  const supervised = d.expected_models.filter(m => m !== 'iso_forest' && m in d.completed_models);
  if (supervised.length === 0) {
    document.getElementById('best-card').innerHTML = `<div class="meta">No supervised model has completed yet. Waiting for first results…</div>`;
    return;
  }
  const best = supervised.reduce((a, b) => (d.completed_models[a].test_calibrated_auc ?? 0) > (d.completed_models[b].test_calibrated_auc ?? 0) ? a : b);
  const c = d.completed_models[best];
  const v2auc = d.baselines?.v2?.test_calibrated_auc;
  const delta = (v2auc !== undefined && v2auc !== null) ? c.test_calibrated_auc - v2auc : null;
  const deltaPill = delta !== null
    ? `<span class="delta ${delta >= 0 ? 'delta-pos' : 'delta-neg'}">${delta >= 0 ? '+' : ''}${n(delta, 3)} vs v2</span>`
    : '';
  document.getElementById('best-card').innerHTML = `
    <div class="grid-3">
      <div>
        <div class="label">Best so far</div>
        <div class="big-num">${best}</div>
        <div class="meta">${c.n_features} features · ${(c.n_train/1e6).toFixed(2)}M train</div>
      </div>
      <div>
        <div class="label">Test AUC (calibrated)</div>
        <div class="big-num">${n(c.test_calibrated_auc, 3)}</div>
        <div class="meta">${deltaPill} &nbsp; CV OOF: ${n(c.cv_oof_auc, 3)}</div>
      </div>
      <div>
        <div class="label">Calibration</div>
        <div class="big-num">${n(c.test_calibrated_brier, 3)}</div>
        <div class="meta">Brier · ECE ${n(c.test_calibrated_ece, 3)} · per-market [${n(c.per_market_auc_min,2)}, ${n(c.per_market_auc_max,2)}]</div>
      </div>
    </div>`;
}

function renderProgress(d) {
  const expected = d.expected_models;
  const completed = d.completed_models;
  const lastLog = (d.log_tail || []).join('\\n');

  const html = expected.map(m => {
    const isDone = m in completed;
    const c = isDone ? completed[m] : null;
    let icon, fillClass, statusText, pct;
    if (isDone) {
      icon = '✓';
      fillClass = '';
      pct = 100;
      const auc = c.test_calibrated_auc ?? c.anomaly_score_test_auc;
      statusText = auc !== undefined ? `AUC ${n(auc, 3)}` : 'done';
    } else if (lastLog.includes(`[${m}]`)) {
      icon = '⟳';
      fillClass = 'running';
      pct = 40;
      statusText = 'running…';
    } else {
      icon = '·';
      fillClass = 'pending';
      pct = 0;
      statusText = 'pending';
    }

    let foldHtml = '';
    if (isDone && c.cv_fold_aucs) {
      const max = 0.8, min = 0.4;
      foldHtml = '<div class="fold-bars">' + c.cv_fold_aucs.map(a => {
        const h = Math.max(2, ((a - min) / (max - min)) * 22);
        return `<div class="fold-bar" title="fold AUC ${n(a,3)}" style="height: ${h}px"></div>`;
      }).join('') + '</div>';
    }

    let topFeats = '';
    if (isDone && c._top_features) {
      const maxAbs = Math.max(...c._top_features.map(f => Math.abs(f.value)));
      topFeats = '<div style="margin-top: 6px;">' + c._top_features.slice(0, 3).map(f => {
        const w = (Math.abs(f.value) / maxAbs) * 100;
        const cls = f.value < 0 ? 'neg' : '';
        return `
          <div class="top-feat">
            <div class="coef">${f.value > 0 ? '+' : ''}${n(f.value, 3)}</div>
            <div class="feat-name">${f.name}</div>
            <div class="feat-bar ${cls}" style="width: ${w * 0.4}px;"></div>
          </div>`;
      }).join('') + '</div>';
    }

    return `
      <div class="progress-track">
        <div class="icon">${icon}</div>
        <div>
          <div class="name">${m}</div>
          ${topFeats}
        </div>
        <div>
          <div class="bar"><div class="bar-fill ${fillClass}" style="width: ${pct}%"></div></div>
          ${foldHtml}
        </div>
        <div class="status-text">${statusText}</div>
      </div>`;
  }).join('');
  document.getElementById('progress').innerHTML = html;
}

function renderTable(d) {
  const supervised = d.expected_models.filter(m => m !== 'iso_forest' && m in d.completed_models);
  if (supervised.length === 0) {
    document.getElementById('table').innerHTML = '';
    return;
  }
  const completed = d.completed_models;
  const bestAuc = Math.max(...supervised.map(m => completed[m].test_calibrated_auc ?? 0));

  // Reference rows: v1 + v2
  const refRows = [];
  for (const [tag, b] of Object.entries(d.baselines || {})) {
    if (b == null) continue;
    refRows.push({
      isRef: true,
      name: `${tag} (16 features)${tag === 'v1' ? ' [price leak]' : ''}`,
      cv_oof: b.oof_auc,
      test_auc: b.test_calibrated_auc,
      test_brier: b.test_calibrated_brier,
      test_ece: b.test_calibrated_ece,
      pmAUC: '—',
    });
  }
  const sweepRows = supervised.map(m => {
    const c = completed[m];
    return {
      isRef: false,
      name: m,
      cv_oof: c.cv_oof_auc,
      test_auc: c.test_calibrated_auc,
      test_brier: c.test_calibrated_brier,
      test_ece: c.test_calibrated_ece,
      pmAUC: `[${n(c.per_market_auc_min, 2)}, ${n(c.per_market_auc_max, 2)}]`,
      isBest: c.test_calibrated_auc === bestAuc,
    };
  });
  const allRows = [...refRows, ...sweepRows];

  const html = `
    <table class="card" style="padding: 0; overflow: hidden;">
      <thead><tr>
        <th>model</th>
        <th class="n">CV OOF AUC</th>
        <th class="n">test AUC (cal)</th>
        <th class="n">test Brier</th>
        <th class="n">test ECE</th>
        <th class="n">per-market AUC</th>
      </tr></thead>
      <tbody>
        ${allRows.map(r => `
          <tr ${r.isBest ? 'class="best"' : ''} ${r.isRef ? 'style="opacity: 0.7; font-style: italic;"' : ''}>
            <td>${r.name}</td>
            <td class="n">${n(r.cv_oof, 3)}</td>
            <td class="n">${n(r.test_auc, 3)}</td>
            <td class="n">${n(r.test_brier, 3)}</td>
            <td class="n">${n(r.test_ece, 3)}</td>
            <td class="n">${r.pmAUC}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
  document.getElementById('table').innerHTML = html;
}

function renderPerMarket(d) {
  const supervised = d.expected_models.filter(m => m !== 'iso_forest' && m in d.completed_models);
  if (supervised.length === 0) {
    document.getElementById('per-market').innerHTML = '<div class="meta">Pending first model.</div>';
    return;
  }
  const best = supervised.reduce((a, b) =>
    (d.completed_models[a].test_calibrated_auc ?? 0) > (d.completed_models[b].test_calibrated_auc ?? 0) ? a : b);
  const pm = d.completed_models[best]._per_market;
  if (!pm) {
    document.getElementById('per-market').innerHTML = '<div class="meta">No per-market data.</div>';
    return;
  }
  const sorted = Object.entries(pm).sort((a, b) => b[1].auc - a[1].auc);
  const html = `
    <div class="meta" style="margin-bottom: 8px;">Showing per-market test AUC for <strong>${best}</strong>. Each row is one ceasefire-by-X market.</div>
    <table>
      <thead><tr>
        <th>market_id</th>
        <th class="n">trades</th>
        <th class="n">pos rate</th>
        <th class="n">AUC</th>
        <th class="n">Brier</th>
      </tr></thead>
      <tbody>
        ${sorted.map(([mid, m]) => `
          <tr>
            <td class="market-cell">${mid}</td>
            <td class="n">${m.n.toLocaleString()}</td>
            <td class="n">${n(m.pos_rate, 2)}</td>
            <td class="n"><span class="auc-cell ${aucClass(m.auc)}">${n(m.auc, 3)}</span></td>
            <td class="n">${n(m.brier, 3)}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
  document.getElementById('per-market').innerHTML = html;
}

function renderIso(d) {
  const c = d.completed_models?.iso_forest;
  if (!c) {
    document.getElementById('iso').innerHTML = '<div class="meta">Pending.</div>';
    return;
  }
  document.getElementById('iso').innerHTML = `
    <div class="grid-3">
      <div>
        <div class="label">Anomaly score → AUC</div>
        <div class="big-num">${n(c.anomaly_score_test_auc, 3)}</div>
        <div class="meta">does anomalous = informed?</div>
      </div>
      <div>
        <div class="label">Top-1% precision</div>
        <div class="big-num">${n(c.top_1pct_precision, 3)}</div>
        <div class="meta">of ${c.top_1pct_n.toLocaleString()} most-anomalous trades, fraction that won</div>
      </div>
      <div>
        <div class="label">Top-5% precision</div>
        <div class="big-num">${n(c.top_5pct_precision, 3)}</div>
        <div class="meta">corr with bet_correct: ${n(c.anomaly_score_target_corr_test, 3)}</div>
      </div>
    </div>`;
}

function renderLog(d) {
  document.getElementById('log').textContent = (d.log_tail || []).join('\\n');
  const el = document.getElementById('log');
  el.scrollTop = el.scrollHeight;
}

async function poll() {
  try {
    if (!cohort) cohort = await fetchJSON('/api/cohort');
    renderCohort();
    const d = await fetchJSON('/api/status');
    renderHeader(d);
    renderBest(d);
    renderProgress(d);
    renderTable(d);
    renderPerMarket(d);
    renderIso(d);
    renderLog(d);
  } catch (e) {
    document.getElementById('header').innerHTML = '<span class="pill pill-off">DISCONNECTED</span> ' + e;
  }
}
poll();
setInterval(poll, 4000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **kw):
        pass

    def do_GET(self):
        if self.path == "/api/status":
            body = json.dumps(get_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/cohort":
            body = json.dumps(COHORT_STATS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "max-age=600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    print(f"Sweep dashboard on http://localhost:{PORT}")
    print(f"  reading: {SWEEP_DIR}")
    print(f"  log: {LOG_FILE}")
    print(f"  pid: {PID_FILE}")
    print(
        f"  cohort cached at startup: n_train={COHORT_STATS.get('n_train')}, n_features={COHORT_STATS.get('n_features')}"
    )
    print("Ctrl+C to stop.")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
