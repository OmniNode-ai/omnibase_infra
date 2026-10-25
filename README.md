# omnibase_infra

omnibase_infra provides ONEX infrastructure implementations, service integrations, and runtime tooling.

Documentation: [knowledge_base](https://github.com/OmniNode-ai/knowledge_base). Dev work-ledger reads: `onex ledger query --bus-lane dev` calls the read node and prints its full JSON answer including freshness and parity; for ledger-ask and catchup set `ONEX_LEDGER_BUS_READ_COMMAND='onex ledger query --bus-lane dev --stdin'`. The issued dev credential and declared lane transport resolve automatically; `--stdin` preserves the caller's correlation and filters and refuses query filter flags. Failed reads are nonzero with no stdout answer; production is refused. The client does not flip ledger authority or certify seven exact parity days or concurrent claim-token uniqueness.
