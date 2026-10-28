# omnibase_infra

omnibase_infra provides ONEX infrastructure implementations, service integrations, and runtime tooling.

Documentation: [knowledge_base](https://github.com/OmniNode-ai/knowledge_base).

## Dev work-ledger reads

`onex ledger query --bus-lane dev` calls the work-ledger read node and prints its full JSON answer including freshness and parity. For ledger-ask and catchup configure `ONEX_LEDGER_BUS_READ_COMMAND='onex ledger query --bus-lane dev --stdin'`. The machine's issued dev credential and declared lane transport are resolved automatically. `--stdin` preserves the caller's correlation and filters and refuses query filter flags. Failed reads are nonzero and print no answer on stdout. Production is refused. This client does not flip ledger authority or certify seven exact parity days or concurrent claim-token uniqueness.

```shell
export ONEX_LEDGER_BUS_READ_COMMAND='onex ledger query --bus-lane dev --stdin'
onex ledger query --bus-lane dev --query rows --ticket OMN-20740
```
