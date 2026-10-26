# The dev bus on three brokers (OMN-21430)

Design and cutover plan for moving the dev bus from one Redpanda broker to a
three-broker Raft cluster on three hosts. **Nothing here is live.** The dev
lane's `redpanda` service in `docker/docker-compose.infra.yml` is unchanged
until the cutover below is run, and the cutover is a separate change.

## Why

The dev bus is one broker, `omnibase-infra-redpanda` (v24.3.18, `--smp 1
--memory 12G`, port 19092) on h201. On 2026-10-25/26 it crashed three times
(SIGSEGV 20:56:59Z, libc++abi abort 22:48:33Z, SIGSEGV 00:59:10Z; lane
broker-crash-cause-0110). Docker restarted it each time, and each crash stopped
all bus traffic and the dev runtime. At 01:21Z it held 9.0 GiB of its 12G arena
in swap on h201 (lane h201-swap-0120). It carries 4631 partitions and 16632
consumer groups, 16337 of them Empty.

## Members

`hosts.tsv` is the member list; one env file per host (`<host>.env.example`).

| Member | Host | Why this host |
|---|---|---|
| h201 | Linux | Holds the existing broker. It moves into this compose project with its data volume and node id, so its footprint on h201 is the same as today and no broker is added there. |
| h202 | Linux, 32 cores | Second Linux host. **Down for repair since 2026-10-23**: the cutover waits for it. |
| h200 | macOS, Docker Desktop | The third host with memory to spare. Docker Desktop's VM is 16 GiB today (`docker info`) and also runs another dev lane, so it must be raised to at least 32 GiB first. |

Not chosen: a second broker on h201 (short of memory; one host is one failure),
h203 (about 300 ms from the house hosts, and Raft writes wait on a majority),
h101 and h105 (about 13 GiB free each).

Ports per member: Kafka `external` 19092 (the client bootstrap list), Kafka
`internal` 19093 (transition only, see cutover step 4), Raft rpc 33145, admin
API 9644 on the host loopback only (OMN-20260). On h200 another dev lane's broker
uses loopback ports 42192/42644, so there is no collision; on h201 the
stability broker is 39092 and the CI bus 21092.

## Memory

Each member runs `--smp 1 --memory 12G --reserve-memory 0M --overprovisioned`
in a container with `mem_limit` 14g and `memswap_limit` 14g:

- 12G is the floor OMN-20266 measured: 6G crashed 12 times in 32 hours at
  2,409 partitions with seastar allocation failures. With RF3 every member
  carries a replica of every partition, so each member holds what the single
  broker holds today; 12G per member is not a reduction.
- The 2G above the arena is for the process outside seastar.
- `memswap_limit == mem_limit` refuses swap. Tonight's broker had 9.0 GiB of
  its arena in swap while host swap traffic ran at up to 163 MB/s.
- Shrinking the partition and group counts (delete the Empty groups, lower
  `group_offset_retention_sec`, as broker-crash-cause-0110 proposed) is what
  would allow a smaller arena later, after a soak.

## Replication and the write quorum

- Cluster config: `default_topic_replications: 3`,
  `internal_topic_replication_factor: 3`.
- Every `onex.cmd.*` and `onex.evt.*` topic and `__consumer_offsets` at RF 3.
  The other topics move to RF 3 in the same pass: a topic left at RF 1 stops
  when its one broker stops.
- Redpanda replicates each partition with Raft. An `acks=all` write is
  acknowledged when 2 of 3 replicas have it, so the cluster keeps taking writes
  and group commits with one member stopped and refuses them with two stopped.
  That is the min-ISR of 2 the design wants; Redpanda's quorum gives it without
  a `min.insync.replicas` setting.
- Producers that must not lose a write on a leader change use `acks=all`.

## Proof (scratch, h200)

`tests/lab/dev_bus_ha/scratch_three_broker_proof.sh` runs the migration path in throwaway
containers and removes them on exit. The recorded run is
`evidence/omn-21430-scratch-proof.txt` (2026-10-26, h200, v24.3.18, three 1G
brokers on one Docker host):

1. One broker, the 377 `onex.cmd.*`/`onex.evt.*` topics of
   `src/omnibase_infra/topics/` at RF 1, a group with offset 5 committed.
2. Two brokers joined; the RF of all 377 topics and `__consumer_offsets` was
   raised to 3 in place. It settled in 339 s; health: 3 nodes, 0 leaderless,
   0 under-replicated, 0 topics below RF 3.
3. The group's committed offset was still 5.
4. The original broker stopped: an `acks=all` write landed (offsets 10-14),
   a group read committed offset 15 (coordinator moved to node 1), and the reply
   event landed.
5. Two brokers stopped: the `acks=all` write did not land (timed out, rc 124).
6. Brokers restarted: healthy, 0 under-replicated.

Not proven by that run: three separate hosts (h202 is down and h203 is
unavailable to lanes; the run used one Docker host), the rpc-address change of
step 2 below, the transition listener of step 4, the rollback, and a GitHub
write through the dev runtime (the run used rpk as producer and consumer).

## Cutover order

Preconditions: h202 is back; h200's Docker Desktop VM is at least 32 GiB; the
Redpanda version is decided (lane redpanda-patch-test-0200 tests newer
releases); the readiness probe no longer describes every group, and the Empty
groups are deleted.

1. **Record** `rpk cluster info`, `rpk topic list` with RF, and
   `rpk group describe` for the declared groups, to compare after each step.
2. **Re-home the h201 broker** (one short bus outage). Stop the dev lane's
   `redpanda` service, then start this compose on h201 with `h201.env`, whose
   `DEV_BUS_HA_DATA_VOLUME` is the dev broker's volume
   (`omnibase-infra-redpanda-data`). Its rpc address changes from
   `redpanda:33145` to the h201 tailnet address, which the other members must
   reach. Check: node id unchanged, all topics present, group offsets match
   step 1. Rehearse this step on a scratch volume first: it is unproven.
3. **Join h202 and h200**: start this compose with `h202.env`, then `h200.env`.
   Check: `rpk redpanda admin brokers list` shows 3 active.
4. **Transition listener.** Until clients move, the runtime reaches the bus on
   the old `internal` listener. Every member advertises an `internal` listener
   at its host address (port 19093), so partitions led by h202 or h200 stay
   visible to those clients.
5. **Raise replication**, in batches of 50 topics:
   `rpk cluster config set default_topic_replications 3`,
   `rpk cluster config set internal_topic_replication_factor 3`, then
   `rpk topic alter-config <topics> --set replication.factor=3` for
   `onex.cmd.*`, `onex.evt.*` and `__consumer_offsets` first, then the rest.
   Watch `rpk cluster partitions move-status` and `rpk cluster health` until
   0 under-replicated. The scratch run moved 378 topics in 339 s; the live
   cluster has about 12 times the partitions.
6. **Move clients to the three-host list**:
   `<h201-address>:19092,<h202-address>:19092,<h200-address>:19092`. That is
   `KAFKA_BOOTSTRAP_SERVERS` and `KAFKA_BROKER_ALLOWLIST` in the `x-runtime`
   env of `docker-compose.infra.yml` (today `redpanda:9092` and `redpanda:`),
   the host scripts that use `localhost:19092`, and the lane configs that name
   the dev bus. The CI bus (21092) and the stability broker (39092) are separate
   and do not change.
7. **Prove** with one broker stopped: a GitHub write through the dev runtime is
   answered (OMN-21430 AC2), and `rpk topic list` shows no `onex.cmd.*` or
   `onex.evt.*` topic below RF 3 (AC1). Restart the member.
8. Remove the `internal` listener once no client uses it.

## Rollback

- **Before step 5** (no replicas on the new members): stop h202 and h200,
  decommission them (`rpk redpanda admin brokers decommission <id>`), stop the
  h201 member and start the dev lane's `redpanda` service on the same volume.
- **After step 5**: set `replication.factor=1` on the raised topics, then
  decommission h202 and h200 (which moves any remaining replicas to the h201 member),
  confirm the h201 member leads every partition, then restore as above and point clients
  back at `redpanda:9092`. Decommissioning with RF 3 on three members cannot
  place three replicas on two, so lowering RF comes first.
- Neither rollback path is proven by the scratch run.
