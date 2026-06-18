# OMN-13277 — Harden RuntimeLocal event-driven initial-payload resolution

## Problem

`RuntimeLocal._run_event_driven` (`src/omnibase_infra/runtime/runtime_local.py`)
resolved the initial command model **only** from the contract's top-level
`input_model`. When that key was absent, `_build_initial_payload({})` returned
`None` and the runtime published a degenerate `{correlation_id}`-only payload to
the first subscribe topic. For any node whose command model carries required
fields, the downstream adapter validation then failed — silently masking the
real cause and dropping every caller-supplied field.

This was the root-cause class behind OMN-13253 (`onex skill dod_verify` published
`{correlation_id}` only, dropping `ticket_id`, so `ModelDodVerifyStartCommand`
validation failed even though the `--input` file carried `ticket_id`). Per-node
contract patches (top-level `input_model` on each node) were the only line of
defense; the sibling nodes named in OMN-13253 (`node_handler_correctness_gate`,
`node_closeout_verifier_compute`, `node_ledger_stats_compute`) were latently
exposed to the same failure.

## What changed

`src/omnibase_infra/runtime/runtime_local.py`:

1. Added `_coerce_model_spec` (static): coerces a contract model reference —
   either a dotted `"module.ClassName"` string or a mapping using `class` (top-
   level `input_model`) or `name` (routing `event_model`) — into a uniform
   `{module, class}` spec, or `None` when no module/class is present.

2. Added `_resolve_event_driven_payload_spec`: resolves the initial-payload
   model spec in order — top-level `input_model` → first routing entry's
   `event_model` → that entry's `handler.input_model` — mirroring the resolution
   `_run_single_handler` already performs. Returns the spec plus a human-readable
   source label, or `None`.

3. Rewrote step 6 of `_run_event_driven`:
   - Resolves the payload model via the helper instead of consulting only
     `input_model`.
   - If NEITHER a top-level `input_model` NOR an `event_model` /
     `handler.input_model`-derivable model can be resolved, it **fails loud**:
     logs and raises a typed `ModelOnexError`
     (`EnumCoreErrorCode.INVALID_INPUT`) whose message names the offending
     contract and the wired handlers — instead of publishing a degenerate
     payload. The raise is recorded as `EnumWorkflowResult.FAILED` at the
     `run_async` boundary (which already catches `ModelOnexError`), so the
     failure surfaces with a clear cause rather than a downstream validation
     error.
   - If a spec resolves but the model cannot be imported/instantiated, it also
     fails loud rather than silently degrading.
   - The degenerate `{correlation_id}`-only `json.dumps` publish branch was
     removed entirely.

## Tests

`tests/unit/runtime/test_run_event_driven_operation_match.py` — extended with two
OMN-13277 cases (file now 5 tests, all passing):

- `test_event_model_only_contract_seeds_full_payload` — a contract with **no**
  top-level `input_model`, a `payload_type_match` `event_model` carrying a
  required `ticket_id`, and an `--input` file. Asserts the handler receives
  `ticket_id == "OMN-99999"`, proving the full payload was seeded from the
  routing entry's `event_model` rather than dropped to `{correlation_id}`.
- `test_no_resolvable_payload_model_fails_loud` — an `operation_match` contract
  with no `event_model`, no `handler.input_model`, and no top-level
  `input_model`. Asserts `result == FAILED` and that the logged error names the
  contract (`test_no_resolvable_model`) and states it
  `could not resolve an initial-payload model`.

The three pre-existing OMN-13141 tests in the same file (operation_match boot,
empty-module spy, payload_type_match import guard) remain green — the fail-loud
check runs after the wiring loop, so the import spy assertion still holds.

## Verification

```
$ uv run pytest tests/unit/runtime/test_run_event_driven_operation_match.py -v
5 passed
```

Full local gate (ruff format + ruff check + mypy --strict + full pytest +
pre-commit) was run green in the worktree before push. The runtime sweep
(previously 312/312) is preserved — this change is additive resolution + a
fail-loud guard on a previously silent degrade path; no public signature or
contract schema changed.

## Impact

Removes the entire "event-driven node silently drops caller input" class:
`event_model`-only and `handler.input_model`-only contracts now seed the full
payload, and an unresolvable contract fails loud naming the contract/handlers
instead of publishing a degenerate payload. Per-node `input_model` contract
patches become belt-and-suspenders rather than the only defense (OMN-13253 +
its named sibling nodes).
