# OMN-11294 Delegate-Skill Stability Runtime Hot-Patch Proof

Date: 2026-05-20
Runtime host: `<user>@<onex-host>`
Runtime lane: `stability-test`

## Context

`OMN-11073` fixed the runtime-owned delegation dispatch port in source and merged
as `omnibase_infra#1624`, but the stability runtime on the runtime host still had a stale
loaded copy of `RuntimeDelegationDispatchPort`.

The stale runtime copy caused direct delegate-skill integration to fail before
the request reached the downstream delegation workflow.

## Pre-Patch Failure

Probe surface:

- Command topic: `onex.cmd.omnimarket.delegate-skill.v1`
- Terminal topic: `onex.evt.omnimarket.delegate-skill-completed.v1`
- Runtime: `omninode-stability-test-runtime`
- Correlation: `8933e3a1-fa4d-40ac-9eab-99e3f88fea1e`

Terminal payload summary:

```json
{
  "status": "failed",
  "correlation_id": "8933e3a1-fa4d-40ac-9eab-99e3f88fea1e",
  "task_type": "test",
  "error_message": "RuntimeDelegationDispatchPort.dispatch() got an unexpected keyword argument 'quality_contract_mode'"
}
```

The deployed source file inside the container had the correct signature under
`/app/.venv/lib/python3.12/site-packages/omnimarket/...`, but the runtime DI
path injects `omnibase_infra.runtime.service_delegation_dispatch_port` from
`/app/src/omnibase_infra/runtime/service_delegation_dispatch_port.py`.

That runtime-owned injected port was stale.

## Hot-Patch Action

The current source file from `omnibase_infra/main` was copied into both
stability runtime containers:

```bash
scp \
  $OMNI_HOME/omnibase_infra/src/omnibase_infra/runtime/service_delegation_dispatch_port.py \
  <user>@<onex-host>:/tmp/service_delegation_dispatch_port.py

ssh <user>@<onex-host> '
  docker cp /tmp/service_delegation_dispatch_port.py \
    omninode-stability-test-runtime:/app/src/omnibase_infra/runtime/service_delegation_dispatch_port.py
  docker cp /tmp/service_delegation_dispatch_port.py \
    omninode-stability-test-runtime-effects:/app/src/omnibase_infra/runtime/service_delegation_dispatch_port.py
  docker restart omninode-stability-test-runtime omninode-stability-test-runtime-effects
'
```

This was a stability-lane runtime patch only. Production containers were not
changed.

## Post-Patch Health

Health checks after restart:

```bash
ssh <user>@<onex-host> 'curl -fsS http://localhost:18085/health'
ssh <user>@<onex-host> 'curl -fsS http://localhost:18086/health'
```

Observed state:

- `:18085` main runtime: `healthy`
- `:18086` effects runtime: `healthy`
- main local ingress: enabled and running
- main active packages: `omnibase_infra`, `omnimarket`, `omniclaude`

Delegate-skill consumer group:

```bash
ssh <user>@<onex-host> '
  docker exec omnibase-infra-stability-test-redpanda \
    rpk group describe \
    stability-test.omnimarket.node_delegate_skill_orchestrator.consume.1.0.0.__i.stability-test-main.__t.onex.cmd.omnimarket.delegate-skill.v1
'
```

Observed state:

- group state: `Stable`
- members: `1`
- total lag: `0`

## Passing Delegation Proof

Both probes used the real direct Codex adapter path:

- command name: `delegate_skill.orchestrate`
- command topic: `onex.cmd.omnimarket.delegate-skill.v1`
- success terminal topic: `onex.evt.omnimarket.delegate-skill-completed.v1`
- failure terminal topic: `onex.evt.omnimarket.delegate-skill-failed.v1`
- target runtime: `runtime://omninode-pc/stability-test/main`
- backend model: `cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit`

### Document Probe

Correlation: `9378451a-f77e-4d8e-8934-ae368548b848`

Terminal summary:

```json
{
  "status": "completed",
  "task_type": "document",
  "provider": "local",
  "model_name": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
  "quality_gate_passed": true,
  "quality_score": 1.0,
  "metrics": {
    "input_tokens": 71,
    "output_tokens": 161,
    "total_tokens": 232,
    "tokens_to_compliance": 232,
    "compliance_attempts": 1,
    "cost_savings_usd": 0.002628,
    "latency_ms": 925
  },
  "error_message": ""
}
```

Runtime logs showed the matching terminal publication:

```text
Published output event to onex.evt.omnimarket.delegate-skill-completed.v1
correlation_id=9378451a-f77e-4d8e-8934-ae368548b848
```

### Test Probe

Correlation: `cb5d0378-0de9-4ea1-84a3-164fc1709c1c`

Terminal summary:

```json
{
  "status": "completed",
  "task_type": "test",
  "provider": "local",
  "model_name": "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
  "quality_gate_passed": true,
  "quality_score": 1.0,
  "metrics": {
    "input_tokens": 91,
    "output_tokens": 303,
    "total_tokens": 394,
    "tokens_to_compliance": 394,
    "compliance_attempts": 1,
    "cost_savings_usd": 0.004818,
    "latency_ms": 1712
  },
  "error_message": ""
}
```

Runtime logs showed the matching terminal publication:

```text
Published output event to onex.evt.omnimarket.delegate-skill-completed.v1
correlation_id=cb5d0378-0de9-4ea1-84a3-164fc1709c1c
```

## Negative Gate Check

A deliberately underspecified `test` task that asked for only `OK` reached the
model and returned content, then failed the quality gate as expected.

Correlation: `7d04799a-f083-46d0-bd9e-9e5812f30387`

Failure summary:

```text
TASK_MISMATCH: missing @pytest.mark.unit;
TASK_MISMATCH: failed covers_edge_cases;
TASK_MISMATCH: failed covers_error_paths
```

This proves the post-patch delegate-skill path is not merely returning success;
it is still enforcing the configured task-class quality gate.

## Follow-Up

The durable source fix already exists on `omnibase_infra/main`. The remaining
operational requirement is to rebuild and redeploy the stability runtime image
from current source, then rerun the two passing probes without manual container
file copies.

Do not treat the manual `docker cp` patch as a release artifact.
