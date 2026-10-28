#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# OMN-21430: scratch proof of the three-broker dev bus, in throwaway containers.
#
# Never touches any live broker: every object this
# script creates is named omn21430-scratch-* and is removed on exit (trap).
# Run it on a lab host other than h201 (it needs ~5 GB of Docker memory).
#
# What it shows, in order:
#   1. one broker, RF1 topics, a consumer group with committed offsets
#      (the shape the dev bus has today);
#   2. two brokers join that cluster; default and internal RF go to 3 and
#      every onex.cmd.* / onex.evt.* topic plus __consumer_offsets is raised
#      to RF3 in place (the migration path: no topic or group is recreated);
#   3. the group's committed offset survives the move;
#   4. with the ORIGINAL broker stopped, an acks=all write lands, a
#      consumer-group read commits, and the reply event lands;
#   5. negative control: with two of three brokers stopped, an acks=all write
#      fails (Raft majority is 2 of 3).
#
# Usage: scratch_three_broker_proof.sh <topics-file>
#   <topics-file>: one topic name per line (the onex.cmd.* / onex.evt.* set).
set -euo pipefail

TOPICS_FILE="${1:?usage: $0 <topics-file>}"
IMAGE="${REDPANDA_IMAGE:-redpandadata/redpanda:v24.3.18}"
NET=omn21430-scratch-net
P=omn21430-scratch-rp
GROUP=omn21430-scratch-runtime
CMD_TOPIC=$(grep -m1 '^onex\.cmd\.' "$TOPICS_FILE")
EVT_TOPIC=$(grep -m1 '^onex\.evt\.' "$TOPICS_FILE")

cleanup() {
  echo "== cleanup"
  for n in 1 2 3; do docker rm -f -v "$P$n" >/dev/null 2>&1 || true; done
  docker network rm "$NET" >/dev/null 2>&1 || true
  docker ps -a --filter "name=omn21430-scratch" --format '{{.Names}}' | sed 's/^/left: /'
  echo "cleanup done"
}
trap cleanup EXIT

rpk_on() { local n=$1; shift; docker exec -i "$P$n" rpk "$@"; }

start_broker() {
  local n=$1 seeds=$2
  docker run -d --name "$P$n" --hostname "$P$n" --network "$NET" \
    --memory 1536m --memory-swap 1536m \
    "$IMAGE" redpanda start \
    --kafka-addr "internal://0.0.0.0:9092" \
    --advertise-kafka-addr "internal://$P$n:9092" \
    --rpc-addr "0.0.0.0:33145" --advertise-rpc-addr "$P$n:33145" \
    --smp 1 --memory 1G --reserve-memory 0M --overprovisioned \
    --default-log-level=warn \
    ${seeds:+--seeds "$seeds"} >/dev/null
}

wait_brokers() {
  local want=$1 via=$2
  for _ in $(seq 1 60); do
    local have
    have=$(rpk_on "$via" redpanda admin brokers list 2>/dev/null | grep -c 'active' || true)
    if [ "$have" -ge "$want" ] && rpk_on "$via" cluster health 2>/dev/null | grep -q 'Healthy:.*true'; then
      return 0
    fi
    sleep 2
  done
  echo "FAIL: cluster did not reach $want healthy brokers"; rpk_on "$via" cluster health || true; return 1
}

docker network create "$NET" >/dev/null

echo "== 1. one broker, RF1 (today's shape)"
start_broker 1 ""
wait_brokers 1 1
rpk_on 1 cluster config set topic_partitions_per_shard 7000 >/dev/null
rpk_on 1 cluster config set topic_memory_per_partition 1048576 >/dev/null
xargs -n 50 docker exec "${P}1" rpk topic create -r 1 -p 1 <"$TOPICS_FILE" | { grep -vc OK || true; } | sed 's/^/topic create non-OK lines: /'
echo "topics on broker: $(rpk_on 1 topic list | grep -cE '^onex\.(cmd|evt)\.')"
for i in $(seq 1 10); do echo "pre-move-$i"; done | rpk_on 1 topic produce "$CMD_TOPIC" --acks=-1 >/dev/null
rpk_on 1 topic consume "$CMD_TOPIC" -g "$GROUP" -n 5 -f '%v\n' >/dev/null
rpk_on 1 group describe "$GROUP" | { grep -E "^$CMD_TOPIC|TOPIC" || true; }

echo "== 2. two brokers join; RF raised to 3 in place"
start_broker 2 "${P}1:33145"
start_broker 3 "${P}1:33145"
wait_brokers 3 1
rpk_on 1 redpanda admin brokers list
rpk_on 1 cluster config set default_topic_replications 3 >/dev/null
rpk_on 1 cluster config set internal_topic_replication_factor 3 >/dev/null
t0=$(date +%s)
{ cat "$TOPICS_FILE"; echo __consumer_offsets; } |
  xargs -n 50 docker exec "${P}1" rpk topic alter-config --set replication.factor=3 >/dev/null
for _ in $(seq 1 150); do
  moving=$(rpk_on 1 cluster partitions move-status 2>/dev/null | grep -cE '^(onex\.|__consumer_offsets|kafka/)' || true)
  if [ "$moving" -eq 0 ] && rpk_on 1 cluster health | grep -qE 'Under-replicated partitions:\s*\[\]'; then break; fi
  sleep 2
done
echo "RF raise settled in $(( $(date +%s) - t0 )) s"
rpk_on 1 cluster health
below=$(rpk_on 1 topic list -i | awk 'NR>1 && $0 ~ /^(onex\.|__consumer_offsets)/ && $3 < 3' | wc -l | tr -d ' ')
total=$(rpk_on 1 topic list -i | awk 'NR>1 && $0 ~ /^onex\.(cmd|evt)\./' | wc -l | tr -d ' ')
echo "onex.cmd/evt topics: $total; topics (incl. __consumer_offsets) with RF below 3: $below"
rpk_on 1 topic list -i | { grep -E '^__consumer_offsets' || true; }

echo "== 3. committed offset survived the move"
rpk_on 1 group describe "$GROUP" | { grep -E "^$CMD_TOPIC|TOPIC" || true; }

echo "== 4. stop the ORIGINAL broker; write, group commit, reply"
docker stop "${P}1" >/dev/null
sleep 12
rpk_on 2 redpanda admin brokers list || true
for i in $(seq 1 5); do echo "one-down-cmd-$i"; done | rpk_on 2 topic produce "$CMD_TOPIC" --acks=-1 2>&1 | tail -2
rpk_on 2 topic consume "$CMD_TOPIC" -g "$GROUP" -n 10 -f '%v\n' | tail -3
echo "reply-for-one-down-cmd" | rpk_on 2 topic produce "$EVT_TOPIC" --acks=-1 2>&1
rpk_on 2 group describe "$GROUP" | { grep -E "^$CMD_TOPIC|TOPIC|STATE|COORDINATOR" || true; }
rpk_on 2 topic consume "$EVT_TOPIC" -o start -n 1 -f '%v\n'

echo "== 5. negative control: two of three stopped, acks=all write must fail"
docker stop "${P}2" >/dev/null
sleep 12
if echo "two-down" | docker exec -i "${P}3" timeout 20 rpk topic produce "$CMD_TOPIC" --acks=-1 --delivery-timeout 15s >/dev/null 2>&1; then
  echo "NEGATIVE CONTROL FAILED: write landed with two brokers stopped"; exit 1
else
  echo "negative control held: write refused or timed out with two brokers stopped (rc=$?)"
fi

echo "== 6. brokers back; cluster healthy"
docker start "${P}1" "${P}2" >/dev/null
wait_brokers 3 3
rpk_on 3 cluster health
echo "PROOF COMPLETE"
