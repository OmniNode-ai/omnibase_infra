# Declared-state reconcile model (OMN-19935, task C0)

TLA+ model of one reconcile of a surface's grants against a declared set, with
concurrent hand edits, two reconcilers and an observability flag. C4 (the
reconcile orchestrator) cites this directory by path.

| Property | Kind | Mutation (`MUT`) that must break it |
| -- | -- | -- |
| `Converges` (apply and a hand edit converge) | temporal | `leak_lock` (a refusal keeps the lock) |
| `NoInterleave` (two reconciles never apply together) | invariant | `no_lock` |
| `NoStaleApply` (a stale plan is refused) | invariant | `no_fresh` |
| `NoNeededRemoved` (no declared grant removed) | invariant | `no_filter` |
| `NoApplyOnUnobservable` | invariant | `no_observable_guard` |

Run (needs `tla2tools.jar`; on the lab it runs in `eclipse-temurin:17-jre-alpine`):

    TLA2TOOLS_JAR=/path/tla2tools.jar ./run_checks.sh results

`results/` holds the committed TLC output for `Model.cfg` and each `mut_*.cfg`,
and `model.sha256`, the digest of the spec plus cfg files those outputs were
produced from. `tests/unit/formal/test_declared_state_reconcile_model.py`
fails when the digest, the passing model or any mutation's violation drifts.
Bounds: 3 grants, 2 reconcilers, 2 hand edits, 1 observability flip.
