# OMN-12548 — Dispatch-Selection Parity Gate (S0 GO signal, epic OMN-12525)

This directory holds the OMN-12548 evidence artifacts. The **canonical, CI-consumed
fixture** lives inside the test tree so CI reads it without a cross-repo path:

| Artifact | Path |
|----------|------|
| **Mode-A oracle fixture** (canonical) | `omnibase_infra/tests/fixtures/dispatch_parity/baseline-selection-v2.json` |
| Regeneration harness | `omnibase_infra/tests/fixtures/dispatch_parity/harness.py` |
| Parity test | `omnibase_infra/tests/integration/runtime/test_dispatch_selection_parity.py` |
| CI gate | `omnibase_infra/.github/workflows/dispatch-parity-gate.yml` |
| Determinism audit (JSON) | `docs/evidence/OMN-12548/determinism-audit.json` |
| Determinism audit (report) | `docs/evidence/OMN-12548/determinism-audit-report.md` |

> **Fixture location note (scope override of design §4):** the design placed the
> fixture under `$OMNI_HOME/docs/evidence/OMN-12548/`. The CI-consumed copy is
> instead committed **inside `omnibase_infra`** (`tests/fixtures/dispatch_parity/`)
> so the parity test reads it at CI time with a local path. The `omni_home`
> evidence dir should carry a one-line pointer back to this copy (trivial docs
> follow-up).

## Regenerate the fixture

```bash
cd $OMNI_HOME/omnibase_infra
uv run python -m tests.fixtures.dispatch_parity.harness \
  --out tests/fixtures/dispatch_parity/baseline-selection-v2.json \
  --audit-out docs/evidence/OMN-12548/determinism-audit.json
```

The fixture is byte-deterministic modulo the `header.generated_at_utc` timestamp; the
CI gate diffs a fresh regeneration against the committed copy (ignoring that field)
to catch a stale oracle.

## P0 live-trace outcomes (folded into the fixture header)

- **P0-1 (guard-tripped orchestrators):** live trace on the `.201` stability-test
  lane (2026-07-02) CONFIRMED `node_rsd_orchestrator` routes ZERO dispatchers and
  falls through to DLQ (`onex.dlq.omnibase-infra.rsd.v1`, which did not exist → event
  dropped). The static oracle REFINES design D5: **4 of 6** guard-tripped
  orchestrators are uniformly NO_DISPATCHER; **2** (`node_chain_orchestrator`,
  `node_registration_orchestrator`) are MIXED — some handler entries dodge the guard
  and DO register routes. The gate pins the observed per-topic behavior as-is
  (design D1); the outage is a High finding under OMN-12525, fixed behind the gate.
- **P0-2 (contract source):** entry-point discovery (`discover_contracts()` →
  `wire_from_manifest()`) builds the initial dispatch table on both the main and
  effects runtimes; `KafkaContractSource` is a post-freeze incremental listener on
  the effects runtime only. The fixture's entry-point corpus is the correct oracle
  source (design D7).

## Corpus scope (design D6)

The infra CI venv (`uv sync`) covers exactly the `onex.nodes` distributions inside
`omnibase_infra`'s dependency closure: **omnibase-core, omnibase-infra,
onex-change-control**. Sibling/downstream packages (**omnimarket, omniclaude,
omniintelligence, omnimemory**) are structurally OUTSIDE that closure (omnimarket
depends on infra, not the reverse) and are **EXPECTED-EXCLUDED** — recorded in the
fixture header with a reason, never silent. A full cross-repo corpus job that
installs the siblings is tracked under OMN-12525. The corpus-guard test FAILS if a
**required** package is missing from the CI venv.
