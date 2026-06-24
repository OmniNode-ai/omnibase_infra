# Contract-Store Durability Audit — Cold-Runtime Census Reconstruction

**Priority:** P1.5
**Date:** 2026-06-11
**Scope:** Prove that a COLD runtime can reconstruct the full registered
contract census WITHOUT relying on retained registration events on
`onex.evt.platform.node-registration.v1` (`cleanup.policy=delete`, 7-day
retention).

## Verdict

**Durability is PROVEN.** The cold-start contract census is reconstructed
entirely from the **image-bundled filesystem manifest**, independent of Kafka
retention. The delete-retention `node-registration.v1` topic feeds ONLY a
post-freeze, additive dynamic-materialization listener that does not replay
history on a cold start. No fix to the contract store is required. A sweep
check ships to detect the one residual gap: a contract that exists ONLY via the
dynamic path and is therefore not durable.

## Evidence

### 1. Cold-start handler resolution is filesystem-sourced, not Kafka-sourced

`omnibase_infra/src/omnibase_infra/runtime/runtime_host_process.py`
`_resolve_handler_descriptors()` resolves the boot census via
`HandlerSourceResolver`. The default mode is **HYBRID**
(`_load_handler_source_config()` returns `EnumHandlerSourceMode.HYBRID` when no
config is supplied — line ~2790). HYBRID merges:

- `HandlerBootstrapSource` — hardcoded bootstrap handlers, and
- `PluginLoaderContractSource(contract_paths=...)` — filesystem contract scan.

`KafkaContractSource` is the active discovery source **only** when
`effective_mode == EnumHandlerSourceMode.KAFKA_EVENTS`. In HYBRID/BOOTSTRAP/
CONTRACT modes the boot census never consults Kafka.

### 2. Live runtime confirms filesystem provenance

Read-only probe of stability-test (`<onex-host>:18085`):

```
GET /v1/introspection/manifest
contract_path = /app/.venv/lib/python3.12/site-packages/omnimarket/nodes/
                node_ab_compare_reducer/contract.yaml
package_name  = omnimarket   package_version = 0.4.3
```

The registered census points at pip-installed package files baked into the
runtime image — durable, version-pinned, and unaffected by any topic purge.

### 3. The delete-retention topic feeds only the dynamic listener

`node-registration.v1` (`SUFFIX_NODE_REGISTRATION`) is consumed by
`_start_dynamic_contract_listener()` /
`_on_dynamic_contract_event()`. These run AFTER the dispatch engine
is frozen and only ADD handlers on top of the already-resolved filesystem
census via `KafkaContractSource.on_contract_registered()` +
`materialize_cached_contract()`.

Live topic config (stability-test redpanda, read-only `rpk topic describe`):

```
cleanup.policy = delete
retention.ms   = 604800000   (7 days)
PARTITION  LOG-START-OFFSET  HIGH-WATERMARK
0          4                 7
1          1                 5
2          0                 1
3          4                 6
4          2                 4
5          2                 4
```

`LOG-START-OFFSET > 0` on five of six partitions proves the log head is ALREADY
truncated by retention — yet the live runtime census is intact, because the
census does not come from this topic.

### 4. The dynamic listener never replays history on cold start

The Kafka event bus default offset policy is `latest`:

- `event_bus/configs/kafka_event_bus_config.yaml`:
  `auto_offset_reset: "${KAFKA_AUTO_OFFSET_RESET:-latest}"`
- `ModelKafkaEventBusConfig.auto_offset_reset` default = `"latest"`.

The dynamic-contract listener subscribes with `EnumConsumerGroupPurpose.CONSUME`
through this bus, so on a cold start with a fresh consumer group it reads from
`latest` — it does NOT replay retained registration events. Consequently a
compacted backup of `node-registration.v1` would NOT change cold-start behavior:
the runtime simply does not re-read that topic to rebuild census.

## Residual gap (covered by the sweep check)

A contract that is registered EXCLUSIVELY via the dynamic path — i.e. it is NOT
bundled in the image filesystem manifest — is non-durable: it vanishes on a cold
runtime restart unless its registrant re-publishes a registration event after
the new runtime is live. This is the only retention-adjacent failure mode, and
it is a *provisioning* gap, not a store-durability gap.

**Enforcement:** `node_runtime_sweep`
(`omnimarket/src/omnimarket/nodes/node_runtime_sweep`) gains a
`NON_DURABLE_CONTRACT` check (`_check_census_durability`). Given the live
registered census and the durable `durable_node_names` manifest, it flags any
live-registered contract absent from the durable source as a CRITICAL finding.
When no manifest is supplied the check is skipped. Unit-tested in
`omnimarket/tests/test_golden_chain_runtime_sweep.py`.

## Why no store fix ships

The ticket conditions a fix (compaction or re-seed-on-boot) on the store being
retention-dependent. The audit proves it is NOT: cold-start census comes from
the durable image filesystem, and the dynamic listener resets to `latest` and
never replays history. Adding compaction or a re-seed would be dead code against
the actual cold-start path. The correct, minimal change is the sweep check that
detects dynamic-only contracts — the single real residual gap.
