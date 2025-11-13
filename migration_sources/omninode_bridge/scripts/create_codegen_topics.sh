#!/bin/bash

set -e

echo "Creating codegen topics with configurations..."
echo ""

# Function to check if topic exists
topic_exists() {
  docker exec omninode-bridge-redpanda rpk topic list 2>/dev/null | grep -q "^$1"
  return $?
}

# Function to create topic if it doesn't exist
create_topic_if_not_exists() {
  local topic_name=$1
  shift

  if topic_exists "$topic_name"; then
    echo "Topic '$topic_name' already exists, skipping..."
  else
    echo "Creating topic '$topic_name'..."
    docker exec omninode-bridge-redpanda rpk topic create "$topic_name" "$@"
  fi
}

# Request topics (omniclaude → omniarchon) - 3 partitions each
echo "Creating request topics..."
create_topic_if_not_exists omninode_codegen_request_analyze_v1 \
  --partitions 3 \
  --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

create_topic_if_not_exists omninode_codegen_request_validate_v1 \
  --partitions 3 \
  --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

create_topic_if_not_exists omninode_codegen_request_pattern_v1 \
  --partitions 3 \
  --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

create_topic_if_not_exists omninode_codegen_request_mixin_v1 \
  --partitions 3 \
  --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

# Response topics (omniarchon → omniclaude) - 3 partitions each
echo "Creating response topics..."
create_topic_if_not_exists omninode_codegen_response_analyze_v1 \
  --partitions 3 \
  --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

create_topic_if_not_exists omninode_codegen_response_validate_v1 \
  --partitions 3 \
  --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

create_topic_if_not_exists omninode_codegen_response_pattern_v1 \
  --partitions 3 \
  --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

create_topic_if_not_exists omninode_codegen_response_mixin_v1 \
  --partitions 3 \
  --replicas 1 \
  --topic-config retention.ms=604800000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

# Status topic (real-time updates) - 6 partitions
echo "Creating status topic..."
create_topic_if_not_exists omninode_codegen_status_session_v1 \
  --partitions 6 \
  --replicas 1 \
  --topic-config retention.ms=259200000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

# Dead letter queues - 1 partition each
echo "Creating DLQ topics..."
create_topic_if_not_exists omninode_codegen_dlq_analyze_v1 \
  --partitions 1 \
  --replicas 1 \
  --topic-config retention.ms=2592000000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

create_topic_if_not_exists omninode_codegen_dlq_validate_v1 \
  --partitions 1 \
  --replicas 1 \
  --topic-config retention.ms=2592000000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

create_topic_if_not_exists omninode_codegen_dlq_pattern_v1 \
  --partitions 1 \
  --replicas 1 \
  --topic-config retention.ms=2592000000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

create_topic_if_not_exists omninode_codegen_dlq_mixin_v1 \
  --partitions 1 \
  --replicas 1 \
  --topic-config retention.ms=2592000000 \
  --topic-config cleanup.policy=delete \
  --topic-config compression.type=gzip

echo ""
echo "================================================"
echo "All 13 codegen topics created successfully!"
echo "================================================"
echo ""
echo "Topic Summary:"
echo "- 4 request topics (3 partitions each)"
echo "- 4 response topics (3 partitions each)"
echo "- 1 status topic (6 partitions)"
echo "- 4 DLQ topics (1 partition each)"
echo ""
echo "Listing all topics:"
docker exec omninode-bridge-redpanda rpk topic list | grep omninode_codegen
echo ""
echo "Topic creation complete!"
