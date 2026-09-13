# OMN-18306 recorded terminal delegation receipt

`failed_no_accepted_attempt_receipt.json` is the verbatim receipt one real
`onex delegate` run produced on 2026-09-13 against the `.201` lab model
endpoint (`omnimarket 0.4.75@b0169103`, `omnibase_infra` at `origin/dev`),
correlation `04b8d0e3-852f-48f5-b873-92a0ca7185fd`. It is the shape the OMN-18306
defect suppressed: a delegation that reached a terminal state with **no
accepted routing attempt**.

It was captured by running the real `run_delegate` path with only
`_write_local_run_files` disabled, because that writer's raise is the defect —
with it enabled the receipt never reaches stdout at all, which is the whole
ticket.

## What the run did, and why the failure is deterministic

`task_type=complex_reasoning`, whose `escalation_policy.tier_order` is
`[local, claude]`. The prompt measured 14,149 tokens against the
`local-heavy-reasoning` backend's declared 8,000-token grounding budget
(OMN-18297), so rung 1 was skipped before the call with
`failure_class=context_too_large`. Rung 2 then failed with a provider
`503`. Ladder exhausted; terminal status `failed`,
`terminal_failure_cause=provider_error`, zero accepted attempts.

The over-budget skip on rung 1 is deterministic for any prompt above the
declared budget. The rung-2 provider error is not, but the fixture is a
recording — the tests read it, they do not re-run it.

## Redactions

Four fields were replaced before committing. None is read by the writer under
test and none carries terminal evidence:

| Field | Replacement | Why |
| -- | -- | -- |
| `terminal_payload.prompt_text`, `handler_result.prompt_text` | a marker naming the original byte count | the 56,593-byte input prompt, not evidence about the run |
| `capture_log` | a marker | local filesystem paths |
| `runtime_identity.host`, `.execution_locus`, `.interpreter` | `REDACTED-HOST` / `REDACTED-LOCAL-PATH` | workstation hostname and local paths |

Everything the tests assert on — `status`, `terminal_failure_cause`,
`error_message`, `metrics.cost_usd`, and both `attempts` entries with their
backend ids, tiers, failure classes, token measurements and acceptance
decisions — is byte-unchanged.

## What this recording does NOT cover

The run produced **no model content** (`response` is empty) because no rung
answered. A terminal failure that DID produce content — a rung answers and the
quality gate then rejects it, the shape described on OMN-18306 from correlation
`8fa7dc9d-0cbd-4d02-a35d-8e43f5714776` — was not reproducible on demand on
2026-09-13: three further live runs over the same prompt all passed the
grounding gate and terminalized `completed`. The content path is asserted
against the writer's field mapping instead, and that gap is stated rather than
papered over.
