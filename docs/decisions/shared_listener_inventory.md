# Shared listener inventory (OMN-22377, rule A2 and the end of section 4)

Part of the OMN-22151 stage-2 design (task T1). The runtime serves the shared host
socket only while it reads ready; this page records what can flip readiness and what
is served from two runtimes at once. `tests/unit/services/test_shared_listener_inventory.py`
fails when a readiness probe or an `http_mount` is added without an entry below.

Rule A2: a probe that can flip without a fault of its own runtime is a T1 failure.
None below can.

## Readiness inputs of `readiness_check` (`runtime_host_process.py`)

`ready = _is_running and not _is_draining and event_bus_ready and supplemental_ready`

| Input | Flips on | Fault of this runtime? |
| -- | -- | -- |
| `_is_running` | the runtime has not finished start, or has stopped | yes: boot, or its own stop |
| `_is_draining` | `runtime.stop()` (the kernel, after the listener drain of A5) | yes: its own stop |
| `event_bus_ready` | an empty assignment or a dead consume loop for a required topic; the bus cause is a rebalance revocation | revocation is caught before it happens (`SharedListener.begin_revocation`, latch first); a dead consume task is a fault |
| supplemental probes | see the next table | yes |

## Supplemental readiness probes

Every `register_readiness_probe(` call site in `src/`, with the fault that flips it.

- probe-site: src/omnibase_infra/runtime/service_kernel.py x1
- probe-site: src/omnibase_infra/runtime/core_runtime/kernel_glue.py x1

| Probe | Call site | Flips when | Flips without a fault of its runtime? |
| -- | -- | -- | -- |
| contract-attach gate (`CONTRACT_ATTACH_PROBE_NAME`) | `service_kernel.py`, armed before the wiring interleave | a required command-topic contract is still pending, `NOT_READY` or `FAILED` (`ContractAttachReadinessGate.status`). After boot `record()` only moves contracts toward `ATTACHED` (the NOT_READY reconciliation loop); it never regresses an attached contract | no. A boot-time state of this runtime's own wiring. A runtime attaches the shared socket only after it has read ready, so a boot-time `false` never occurs on an attached runtime |
| `core_runtime_loop` (`readiness_snapshot` of the core runtime handle) | `core_runtime/kernel_glue.py` | the dispatch loop task ended unexpectedly (`is_loop_healthy`), or the phantom-subscription probe flagged a core-runtime topic: assigned, HWM advanced by this runtime's own polls, LAG 0 and zero messages dispatched (`PhantomAlarmMonitor.evaluate`) | no. Both inputs are this runtime's own task and its own poll and dispatch counters. Note for T2: a paused standby must keep `polled_by_topic` at zero, or the phantom rule reads a paused partition as a phantom |

## Contract HTTP mounts and local-ingress handlers (served by both runtimes during an overlap)

The listener's second `web.Application` is built by the same `ServiceHealth._build_app`
as the in-container one, including the `AsgiMounts` middleware, so every mount and the
`/skill` ingress answer on the shared socket too.

- http-mount: src/omnibase_infra/nodes/node_contract_resolver_bridge/contract.yaml

| Surface | In-process singleton state | Disposition |
| -- | -- | -- |
| `http_mount` of `node_contract_resolver_bridge` (`/api/nodes`, factory `services.contract_resolver.main:create_mounted_app`) | `app.state.event_bus` and `app.state.container`, both per process, both read by request handlers; no cache, no counter, no module-level mutable state found in `contract_resolver/main.py` or `routes.py` | stateless per request, served by both runtimes like two replicas. No gate to `active` |
| `POST /skill` (`ServiceHealth._handle_skill`) | none in the handler: it validates the body and calls `RuntimeHostProcess.dispatch_local_ingress_request`; the routes come from the contracts' `local_ingress` declarations (`discover_runtime_local_ingress_routes`) | the dispatch path holds no singleton state. **Not audited**: the individual handlers behind each contract-declared local-ingress route. A handler that keeps in-process state must be gated to `active` by its own lane before the standby slot (T2) is installed |
| `/health`, `/health/detailed`, `/v1/introspection/manifest` | read-only views of the process | none |

## Open items this page does not close

- The rebalance listener that calls `begin_revocation` and `end_revocation` does not exist in
  `src/` yet (`on_partitions_revoked` has no caller); T2 owns it. T1 supplies the API only.
- `role`, `epoch` and `consuming` in the `/ready` body read `role`, `epoch` and
  `is_consuming` from the runtime when it has them; an active runtime without them reads
  `active`, `0`, `true`.
