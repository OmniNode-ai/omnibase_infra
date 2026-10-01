# Lab-tenant benchmark: BEFORE (arm local-stack), 2026-10-01T00:15Z

OMN-20213. Harness `scripts/bench_lab_tenant.py` at edceb7af3. h105 (omnibook) and h101
(Stickybeatz) each ran their full dogfood stack; three repetitions per host, 20 s apart. The
.201 dependency side was sampled three times while the satellites were in this arm.

- Latencies are measured inside the runtime container (`omninode-dogfood-runtime`) with the
  runtime's own bus and database settings. Bus round trip: publish, consume, synchronous
  commit, ack event, ack received (N=200 after 10 warm-up). Projection: publish on
  the steel-onslaught terminal topic, which only the event-ledger projection consumes, then wait
  until the `event_ledger` row with that correlation id is visible (N=50 after 3 warm-up).
  Health: the container's own healthcheck URL (N=5).
- Every run removed what it wrote: 53 `event_ledger` rows per repetition deleted by correlation
  id, the projection topic trimmed back, and both per-run bench topics deleted. The per-run echo
  group was already absent.
- Cold start was not timed. The harness restarts the runtime pair only with `--allow-restart` on
  the lab-tenant arm. The last starts were h105 2026-09-27T20:25:13Z and h101
  2026-09-28T11:45:22Z.
- Placement: both copies of `landing_placement.py` read each host as the landing controller
  does: the installed omni plugin (0.4.151) and the controller's deployed copy. h101 is refused
  admission in every repetition (mem_avail 7.1 GB < 10 GB, and macOS memory pressure level 2).
- The .201 dev lane had redeployed its app tier about 24 minutes before the dependency sample
  (`dev_lane.min_uptime_s`), so its app-container memory is post-restart. The
  `tenant-omn20213-proof-f7ff` prefix on the .201 broker was created by another lane, not by this
  harness.
- Machine paths in the JSON are home-relative (`~/`).

Satellites: median (min-max) over repetitions

| metric | h101 local-stack (n=3) | h105 local-stack (n=3) |
|---|---|---|
| host available (free+inactive+spec) GB | 7.167 (6.891-7.193) | 11.286 (11.274-11.328) |
| host free GB | 0.06 (0.059-0.066) | 0.11 (0.064-0.156) |
| host inactive GB | 7.098 (6.8-7.126) | 11.158 (11.133-11.19) |
| host speculative GB | 0.01 (0.008-0.026) | 0.019 (0.015-0.044) |
| host compressor GB | 14.234 (14.188-14.79) | 6.743 (6.702-6.746) |
| host wired GB | 2.761 (2.758-2.773) | 2.148 (2.146-2.178) |
| swap used MiB | 7389.56 (7389.56-7397.5) | 3112.56 (3112.56-3112.56) |
| memory pressure level | 2 (2-2) | 1 (1-1) |
| Docker VM configured GB | 23.75 (23.75-23.75) | 16 (16-16) |
| Docker VM process RSS GB | 6.229 (5.635-6.275) | 11.657 (11.648-11.676) |
| Docker VM process footprint GB | 24 (24-24) | 16 (16-16) |
| containers total (docker stats) GB | 4.046 (4.044-4.047) | 4.32 (4.318-4.321) |
| bus round trip p50 ms | 0.577 (0.522-0.605) | 0.389 (0.382-0.394) |
| bus round trip p95 ms | 0.775 (0.705-0.779) | 0.465 (0.459-0.476) |
| bus round trip p99 ms | 0.889 (0.804-1.71) | 0.583 (0.499-0.762) |
| bus publish->consume p50 ms | 0.146 (0.133-0.155) | 0.104 (0.102-0.105) |
| projection publish->row p50 ms | 6.823 (6.71-7.557) | 6.697 (6.541-6.775) |
| projection publish->row p95 ms | 8.045 (7.879-8.163) | 7.408 (7.395-7.589) |
| projection publish->row p99 ms | 8.061 (7.963-10.79) | 7.71 (7.431-7.788) |
| health probe p50 ms | 0.845 (0.708-0.956) | 0.544 (0.49-0.579) |
| host NTP offset ms | 4.619 (4.252-4.629) | 9.107 (8.977-9.912) |
| placement.controller.admitted | 0 (0-0) | 1 (1-1) |
| placement.controller.load1 | 3.55 (2.69-3.76) | 2.23 (2.22-2.39) |
| placement.controller.mem_avail_gb | 7.146 (7.142-7.176) | 11.26 (11.25-11.271) |
| placement.controller.slots | 0 (0-0) | 1 (1-1) |
| placement.plugin_cache.admitted | 0 (0-0) | 1 (1-1) |
| placement.plugin_cache.load1 | 3.55 (2.66-3.76) | 2.23 (2.22-2.39) |
| placement.plugin_cache.mem_avail_gb | 7.126 (7.119-7.154) | 11.288 (11.254-11.295) |
| placement.plugin_cache.slots | 1 (1-1) | 3 (3-3) |
| mem.container.omnibase-infra-dogfood-delegation-fault-429_mib | - | 173.4 (173.2-173.7) |
| mem.container.omnibase-infra-dogfood-delegation-fault-503_mib | - | 107.3 (107.2-107.4) |
| mem.container.omnibase-infra-dogfood-migration-gate_mib | 1.8 (1.6-1.9) | 1.7 (1.4-2.8) |
| mem.container.omnibase-infra-dogfood-postgres_mib | 201.8 (201.2-202.2) | 216.3 (216.1-216.9) |
| mem.container.omnibase-infra-dogfood-redpanda_mib | 2022.4 (2021.4-2022.4) | 1986.6 (1986.6-1986.6) |
| mem.container.omnibase-infra-dogfood-valkey_mib | 9.9 (9.9-10.4) | 10.2 (9.8-10.4) |
| mem.container.omnimarket-dogfood-projection-api_mib | 309.8 (309.5-310) | 146.6 (146.2-146.7) |
| mem.container.omninode-air-runner-1_mib | - | 626.1 (626.1-626.1) |
| mem.container.omninode-dogfood-runtime-effects_mib | 525 (525-525.3) | 478.4 (478.3-479.1) |
| mem.container.omninode-dogfood-runtime_mib | 654.5 (654.5-654.8) | 676.4 (676.3-676.9) |
| mem.container.omninode-mini-runner-1_mib | 417.1 (417.1-418.1) | - |

Dependency side, by the arm the satellites were in: median (min-max)

| metric | h201 while local-stack (2026-10-01T00:18, n=3) |
|---|---|
| cpu.busy_cores | 10.56 (5.29-25.02) |
| cpu.load1 | 35.38 (11.75-36.26) |
| cpu.load15 | 14.59 (12.87-16.26) |
| cpu.load5 | 16.45 (10.84-20.83) |
| cpu.ncpu | 32 (32-32) |
| dev_lane.count | 26 (26-26) |
| dev_lane.cpu_percent_total | 63.27 (48.86-65.97) |
| dev_lane.mem_total_gb | 2.665 (2.592-3.018) |
| dev_lane.min_uptime_s | 1478.2 (1418.4-1533) |
| mem.Cached_gb | 4.2 (2.499-9.761) |
| mem.MemAvailable_gb | 67.072 (66.804-69.065) |
| mem.MemFree_gb | 2.263 (0.961-14.501) |
| mem.MemTotal_gb | 91.958 (91.958-91.958) |
| mem.SwapFree_gb | 153.599 (153.548-155.656) |
| mem.SwapTotal_gb | 200 (200-200) |
| pg.connections.<none> | 5 (5-5) |
| pg.connections.infisical_db | 6 (3-7) |
| pg.connections.keycloak | 1 (1-1) |
| pg.connections.omnibase_infra | 19 (17-19) |
| pg.connections.omnidash_analytics | 10 (10-17) |
| pg.connections.omniintelligence | 1 (1-1) |
| pg.connections.omninode_cloud | 2 (2-2) |
| pg.connections.postgres | 1 (1-1) |
| pg.connections_total | 45 (44-49) |
| redpanda.bytes_consume_per_s | 292387 (142845-293165) |
| redpanda.bytes_follower_consume_per_s | 0 (0-0) |
| redpanda.bytes_produce_per_s | 98001.9 (91410.5-138004) |
| redpanda.records_fetched.tenant-omn20213-proof-f7ff_per_s | 0 (0-0.061) |
| redpanda.records_fetched.tenant-omninode_per_s | 0 (0-0) |
| redpanda.records_fetched.tenant-onex-lab-house_per_s | 0 (0-0) |
| redpanda.records_fetched_per_s | 58.29 (11.329-86.537) |
| redpanda.records_produced.tenant-omn20213-proof-f7ff_per_s | 0 (0-0) |
| redpanda.records_produced.tenant-omninode_per_s | 0 (0-0) |
| redpanda.records_produced.tenant-onex-lab-house_per_s | 0 (0-0) |
| redpanda.records_produced_per_s | 33.658 (30.492-50.773) |
