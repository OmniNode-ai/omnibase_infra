# The manual deploy trigger is built from the command contract (OMN-16442)

Moved from the module docstring of `scripts/deploy-agent/deploy_agent/trigger.py`
so the file stays within its shrink-only size ratchet.

## What broke

`deploy-agent-trigger.sh` used to hand-write the command JSON in an embedded
Python snippet:

```python
envelope = {
    "correlation_id": ..., "git_ref": ..., "reason": ...,
    "requested_by": ..., "scope": "runtime", "services": [],
}
```

The agent that consumes it validates against
`deploy_agent.events.ModelRebuildRequested`, which requires `runtime_lane` and
forbids `reason`. The two drifted apart and nothing noticed, because nothing
compared them. Measured on the dev lane host at 2026-09-08T12:14:27Z, verbatim
from the agent's journal:

```
Invalid payload (correlation_id=2b017562-...): 2 validation errors for
ModelRebuildRequested / runtime_lane / Field required / reason / Extra
inputs are not permitted
Rejected command: invalid_payload
```

So no operator invocation of the sanctioned script could ever be accepted, and
the script's own header forbids hand-building the JSON (correctly: the HMAC is
computed over exact bytes). The entry point was closed in both directions.

## Transport

The old script read unprefixed `KAFKA_SASL_USERNAME` / `KAFKA_SASL_PASSWORD`,
while the lane env file carries the dev SCRAM principal under `DEV_`-prefixed
names, so its SASL block was silently skipped. Its `kcat` and `rpk` branches
then hardcoded `SASL_SSL` / `PLAIN` against a broker that has been
`SASL_PLAINTEXT` + `SCRAM-SHA-256` since 2026-09-07 (OMN-18012 Phase B). Neither
tool was on PATH on the host, so resolution fell through to a third branch that
published through `docker exec ... rpk` with no authentication at all.

The module now calls
`deploy_agent.kafka_config.load_deploy_agent_kafka_config_from_env`, the loader
the agent itself starts from, so there is one declared transport read by both
ends.
