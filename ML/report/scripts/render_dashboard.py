"""Live dashboard for enrichment progress.

Reads data/enrichment_progress.json and regenerates
data/enrichment_dashboard.html every 5 seconds. Open the HTML in a browser —
it meta-refreshes every 5s so it always shows fresh numbers.

Usage:
  nohup python scripts/render_dashboard.py > /dev/null 2>&1 &
  open data/enrichment_dashboard.html
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "data" / "enrichment_progress.json"
OUT_PATH = ROOT / "data" / "enrichment_dashboard.html"

# Keep a rolling history of (timestamp, completed) for sparkline
history: deque[tuple[float, int]] = deque(maxlen=60)


def render_html(status: dict | None) -> str:
    if status is None:
        pct = 0
        completed = 0
        total = "…"
        rate = "—"
        eta = "waiting for first checkpoint…"
        elapsed = "—"
        failures = "—"
        workers = "—"
        keys = "—"
        updated = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        sparkline_pts = ""
        status_dot = "#ffa500"  # orange — initializing
    else:
        pct = status.get("pct_done", 0)
        completed = f"{status.get('completed', 0):,}"
        total = f"{status.get('total_wallets', 0):,}"
        rate = f"{status.get('rate_per_sec', 0):.2f}"
        eta = status.get("eta_human") or "—"
        elapsed = status.get("elapsed_human", "—")
        failures = f"{status.get('failures', 0):,}"
        workers = status.get("workers", "—")
        keys = status.get("keys_in_use", "—")
        updated = status.get("updated_at", "")[11:19] + " UTC"
        status_dot = "#a8ff60"  # mint — running

        # Update history
        now = time.time()
        history.append((now, int(status.get("completed", 0))))
        # Derive sparkline: wallets-per-update over the last 60 samples
        if len(history) >= 2:
            pts = []
            for i in range(1, len(history)):
                dt = history[i][0] - history[i - 1][0]
                dn = history[i][1] - history[i - 1][1]
                pts.append(dn / dt if dt > 0 else 0)
            if pts:
                pmax = max(pts) or 1
                coords = " ".join(
                    f"{(i / max(1, len(pts) - 1)) * 280:.1f},{40 - (v / pmax) * 35:.1f}"
                    for i, v in enumerate(pts)
                )
                sparkline_pts = coords
            else:
                sparkline_pts = ""
        else:
            sparkline_pts = ""

    # SVG progress ring
    circumference = 2 * 3.14159 * 90
    offset = circumference * (1 - pct / 100)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="5">
<title>Enrichment — {pct:.1f}%</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0a0a0a;
    color: #e5e5e5;
    font: 400 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
  }}
  .container {{
    max-width: 720px;
    width: 100%;
  }}
  header {{
    text-align: center;
    margin-bottom: 48px;
  }}
  .title {{
    font-size: 24px;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
  }}
  .sub {{
    color: #888;
    font-size: 13px;
  }}
  .ring-wrap {{
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 48px;
  }}
  .ring {{
    position: relative;
    width: 220px;
    height: 220px;
  }}
  .ring svg {{
    transform: rotate(-90deg);
  }}
  .ring-bg {{
    fill: none;
    stroke: #1a1a1a;
    stroke-width: 8;
  }}
  .ring-fg {{
    fill: none;
    stroke: #a8ff60;
    stroke-width: 8;
    stroke-linecap: round;
    stroke-dasharray: {circumference:.2f};
    stroke-dashoffset: {offset:.2f};
    transition: stroke-dashoffset 0.5s ease;
  }}
  .ring-center {{
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }}
  .ring-pct {{
    font-size: 44px;
    font-weight: 600;
    letter-spacing: -0.04em;
  }}
  .ring-label {{
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 32px;
  }}
  .stat {{
    background: #151515;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 16px;
  }}
  .stat-label {{
    font-size: 11px;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
  }}
  .stat-val {{
    font-size: 20px;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: #e5e5e5;
  }}
  .stat-sub {{
    font-size: 11px;
    color: #666;
    margin-top: 2px;
  }}
  .sparkline-card {{
    background: #151515;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
  }}
  .sparkline-card-label {{
    font-size: 11px;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
  }}
  .sparkline {{
    display: block;
    width: 100%;
    height: 40px;
  }}
  .sparkline polyline {{
    fill: none;
    stroke: #a8ff60;
    stroke-width: 1.5;
    stroke-linecap: round;
    stroke-linejoin: round;
  }}
  footer {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-size: 12px;
    color: #555;
  }}
  .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {status_dot};
    animation: pulse 2s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(0.85); }}
  }}
  @media (max-width: 560px) {{
    .stats {{ grid-template-columns: repeat(2, 1fr); }}
    .ring {{ width: 180px; height: 180px; }}
  }}
</style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title">On-chain enrichment</div>
      <div class="sub">Polygon wallet history → causal feature arrays</div>
    </header>

    <div class="ring-wrap">
      <div class="ring">
        <svg width="220" height="220">
          <circle class="ring-bg" cx="110" cy="110" r="90"/>
          <circle class="ring-fg" cx="110" cy="110" r="90"/>
        </svg>
        <div class="ring-center">
          <div class="ring-pct">{pct:.1f}%</div>
          <div class="ring-label">{completed} / {total}</div>
        </div>
      </div>
    </div>

    <div class="stats">
      <div class="stat">
        <div class="stat-label">Rate</div>
        <div class="stat-val">{rate}</div>
        <div class="stat-sub">wallets/sec</div>
      </div>
      <div class="stat">
        <div class="stat-label">ETA</div>
        <div class="stat-val">{eta}</div>
        <div class="stat-sub">to completion</div>
      </div>
      <div class="stat">
        <div class="stat-label">Elapsed</div>
        <div class="stat-val">{elapsed}</div>
        <div class="stat-sub">since start</div>
      </div>
      <div class="stat">
        <div class="stat-label">Failures</div>
        <div class="stat-val">{failures}</div>
        <div class="stat-sub">{workers} workers · {keys} keys</div>
      </div>
    </div>

    <div class="sparkline-card">
      <div class="sparkline-card-label">Throughput — last ~{min(60, len(history))} samples</div>
      <svg class="sparkline" viewBox="0 0 280 40" preserveAspectRatio="none">
        <polyline points="{sparkline_pts}"/>
      </svg>
    </div>

    <footer>
      <div class="dot"></div>
      <span>Last updated {updated} · auto-refreshes every 5s</span>
    </footer>
  </div>
</body>
</html>
"""


def main() -> None:
    print(f"rendering to {OUT_PATH} every 5s — open in a browser")
    while True:
        status = None
        if STATUS_PATH.exists():
            try:
                status = json.loads(STATUS_PATH.read_text())
            except Exception:
                status = None
        OUT_PATH.write_text(render_html(status))
        time.sleep(5)


if __name__ == "__main__":
    main()
