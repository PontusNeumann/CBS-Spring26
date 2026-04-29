# Notes index

Markdown docs for Alex's idea1 workspace. Files are tagged by status:

- **Canonical** — load-bearing for the report. Read these first.
- **Reference** — pre-pivot but still valid for narrow questions.
- **Historical** — session logs and snapshots. Read only for archaeology.

When in doubt, prefer canonical files. If a reference file conflicts with a canonical one, canonical wins.

---

## Canonical (post-2026-04-28)

| File | Purpose | Update when |
|---|---|---|
| [`alex-approach.md`](alex-approach.md) | Single-source pipeline summary written for Pontus to read cold. RQ, cohort, features, models, results, sensitivity, comparison points, scoped next steps. | RQ changes, cohort changes, feature count changes, headline finding changes |
| [`design-decisions.md`](design-decisions.md) | Append-only log of design decisions (D-001..D-037). Each entry: Decision / Alternatives / Justification / Implications / Status. | Any time a non-trivial decision is made or revised. Add new D-NNN entry, never edit retroactively (mark old as superseded if needed) |
| [`pressure-test.md`](pressure-test.md) | Register of 18 falsification tests. Verdicts (PASS / FAIL / NOTE / UNTESTED), full results, headline findings. | After running a pressure test, when a verdict changes, when adding a new test |
| [`data-manifest.md`](data-manifest.md) | Schema + per-file row counts + SHA-256 checksums for the data parquets distributed via GitHub release. | Each time the data release is republished |
| [`../v4_final_ml_pipeline/`](../v4_final_ml_pipeline/) | The locked end-to-end ML/DL pipeline (top-level in alex workspace). Cleaned + renumbered scripts (01-12), `_common.py` shared utilities, `README.md` + `pipeline.md` runbooks. Runs against Pontus's v4 wallet-augmented parquets. | When pipeline stages, scripts, or wall times change |

---

## Reference (pre-pivot, still authoritative on narrow topics)

| File | Purpose | Status |
|---|---|---|
| [`feature-exclusion-list.md`](feature-exclusion-list.md) | Catalogue of features deliberately excluded from v3.5 (e.g., `side`, `outcomeIndex`, `log_trade_value_usd`). With reasons. | Still authoritative — feature exclusions haven't changed since pivot |
| [`sister-market-features.md`](sister-market-features.md) | Spec for cross-rung sister-market aggregates considered but not built. | Reference only — not implemented in v3.5 |
| [`test-cohort-no-bias.md`](test-cohort-no-bias.md) | Argument that the test-cohort selection doesn't introduce selection bias. | Still authoritative — argument is unchanged |
| [`new-chat-prompt.md`](new-chat-prompt.md) | Kickoff prompt template for resuming work in a fresh chat session. | Reference only — update when workspace structure changes meaningfully |

---

## Historical (session logs, snapshots, archaeology)

| File | Purpose | Why kept |
|---|---|---|
| [`session-learnings-2026-04-22.md`](session-learnings-2026-04-22.md) | Distilled learnings from the 2026-04-22 session (cohort building, feature engineering v1→v3.5). | Useful audit trail for why we ended up with v3.5 features. Don't rely on for current state. |
| [`session-learnings-2026-04-28.md`](session-learnings-2026-04-28.md) | Distilled learnings from the 2026-04-28 session (cost-bug discovery, pressure testing, +14% corrected ROI). | Snapshot before the RQ revision. The revision itself is in design-decisions.md D-001 / D-034. |

---

## When to add a new file here

- **New session learnings** → append to a `session-learnings-YYYY-MM-DD.md` (one per significant session, append-only after publishing)
- **New canonical doc** → add to the Canonical table above and update [`../README.md`](../README.md)'s "Read in this order" list
- **New reference** → add to the Reference table; if it might conflict with a canonical doc, flag the relationship explicitly

Keep the 4-canonical list short. If a fifth canonical emerges, consider whether it should fold into one of the existing four instead.
