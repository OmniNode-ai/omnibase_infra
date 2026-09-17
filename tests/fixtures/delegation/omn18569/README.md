# OMN-18569 recorded DISPATCHED delegation receipt

`dispatched_envelope_carrier_receipt.json` is the verbatim
`ModelSkillResult[ModelReceiptRuntimeSummary]` that one real `onex delegate`
run printed on stdout on 2026-09-17, dispatched to the `.201` dev lane
(compose project `omnibase-infra`, ports 8085/8086) with
`--bus kafka --lane dev --locus deployed-lane`. Correlation
`83aa8b6c-8189-49f0-953d-80c8f015ed0a`, run
`d0c7356a-4f02-41ef-aee3-a28c009a57ad`, `omnibase_infra 0.38.30`,
`omnimarket 0.4.111@98fee097`.

It is the shape the OMN-18569 defect suppressed: a delegation whose terminal
arrives inside its **event envelope**, so the delegation fields sit under
`result.terminal_payload.payload` rather than on `result.terminal_payload`.

## What the run did

`task_type=summarization`, prompt "List the first five prime numbers in
ascending order, separated by commas, and nothing else." The lane's local rung
answered `2, 3, 5, 7, 11` on the first attempt at quality 1.0 against a 0.8
bar, `acceptance_decision=accept`, `escalation_count=0`, `cost_usd=0`. The run
exited **0** with a correct answer.

It still took the `ModelReceiptRuntimeSummary` branch, and that is the point:
`receipt_mode` builds the typed `ModelSkillResult[JsonValue]` receipt only when
the run is success-like **and** a handler result exists, and a dispatched run
hosts no handlers, so `handler_result` is `null` on every dispatched run
regardless of outcome.

Despite exit 0 and a correct answer, `result.txt`, `receipt.json` and
`run.json` were **not written** and nothing said so. That is what this fixture
pins.

## Provenance, stated because it matters

This is the CLI's own stdout, saved by the probe lane
(`b5h1-delegation-probe-201-1058`) as it ran. It is a recording, not a
reconstruction: no field was rebuilt from the runtime's
`workflow_result.json`, and no shape here was hand-authored.

## Redactions

Five fields were rewritten before committing. None is read by the code under
test; each carries host identity rather than evidence about the run.

| Field | Replacement | Why |
| -- | -- | -- |
| `runtime_identity.host` | `REDACTED-HOST` | workstation hostname |
| `runtime_identity.execution_locus`, `.interpreter` | `REDACTED-LOCAL-PATH` | local paths |
| `runtime_identity.config_source`, `result.workflow`, `result.orchestrator_distribution` | local venv prefix -> `REDACTED-LOCAL-PATH` | local paths |
| `result.dispatch_target` (broker address only) | `REDACTED-LAB-BROKER` | private lab host address |

This diverges from `../omn18306/`, which left absolute local paths in
`result.workflow` and `runtime_identity.config_source`. The prefixes are
stripped here because this repository is public; the contract path's tail is
kept intact because `_delegation_result` scopes the unwrap by finding
`node_delegate_skill_orchestrator` inside that string, so redacting it
wholesale would make the fixture pass for the wrong reason.

Everything the tests assert on is byte-unchanged: `status`, `exit_code`,
`correlation_id`, `run_id`, and the whole of `result.terminal_payload`
including the envelope fields, the nested delegation payload, its `response`,
and the single attempt record with its tier, backend id, model id, quality
verdict and acceptance decision. The prompt is kept verbatim as well — it is
91 bytes of arithmetic and carries nothing that needs hiding.

## What this recording does NOT cover

One accepted attempt on the first rung. It says nothing about a dispatched run
that escalates, or one that terminalizes failed; the escalation and
no-accepted-attempt paths are covered by `../omn18306/` and by the synthesized
summary receipts in `tests/unit/cli/test_cli_delegate.py`, on the bare carrier
shape. A dispatched run with no accepted rung is not recorded anywhere yet, and
that gap is stated rather than papered over.
