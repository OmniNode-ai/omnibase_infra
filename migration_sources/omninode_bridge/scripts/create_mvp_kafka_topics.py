#!/usr/bin/env python3
"""
Create Kafka Topics for MVP Event-Driven Architecture

Creates all topics defined in event contracts for:
- Node generation workflow
- Metrics aggregation
- Pattern storage
- Intelligence gathering
- Orchestration coordination

Usage:
    poetry run python scripts/create_mvp_kafka_topics.py [--environment dev|staging|prod]
"""

import os
import subprocess
import sys

# Kafka topic configuration from event contracts
KAFKA_TOPICS = {
    # Node generation workflow
    "NODE_GENERATION_REQUESTED": {
        "topic": "dev.omninode-bridge.codegen.generation-requested.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "description": "User requests node generation via CLI",
    },
    "NODE_GENERATION_STARTED": {
        "topic": "dev.omninode-bridge.codegen.generation-started.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "description": "Orchestrator begins generation workflow",
    },
    "NODE_GENERATION_STAGE_COMPLETED": {
        "topic": "dev.omninode-bridge.codegen.stage-completed.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "description": "Pipeline stage completes",
    },
    "NODE_GENERATION_COMPLETED": {
        "topic": "dev.omninode-bridge.codegen.generation-completed.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 2592000000,  # 30 days
        "description": "Generation workflow successful",
    },
    "NODE_GENERATION_FAILED": {
        "topic": "dev.omninode-bridge.codegen.generation-failed.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 2592000000,  # 30 days
        "description": "Generation workflow failed",
    },
    # Metrics aggregation
    "GENERATION_METRICS_RECORDED": {
        "topic": "dev.omninode-bridge.codegen.metrics-recorded.v1",
        "partitions": 1,  # Single partition for ordered aggregation
        "replication_factor": 1,
        "retention_ms": 7776000000,  # 90 days
        "description": "Aggregated metrics recorded",
    },
    # Pattern storage
    "PATTERN_STORAGE_REQUESTED": {
        "topic": "dev.omniarchon.intelligence.pattern-storage-requested.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "description": "Request to store successful pattern",
    },
    "PATTERN_STORED": {
        "topic": "dev.omniarchon.intelligence.pattern-stored.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "description": "Pattern successfully stored",
    },
    # Intelligence gathering
    "INTELLIGENCE_QUERY_REQUESTED": {
        "topic": "dev.omniarchon.intelligence.query-requested.v1",
        "partitions": 5,  # Higher throughput for queries
        "replication_factor": 1,
        "retention_ms": 604800000,
        "description": "Request RAG intelligence for generation",
    },
    "INTELLIGENCE_QUERY_COMPLETED": {
        "topic": "dev.omniarchon.intelligence.query-completed.v1",
        "partitions": 5,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "description": "Intelligence gathering complete",
    },
    # Orchestration
    "ORCHESTRATOR_CHECKPOINT_REACHED": {
        "topic": "dev.omninode-bridge.codegen.checkpoint-reached.v1",
        "partitions": 1,  # Sequential processing
        "replication_factor": 1,
        "retention_ms": 86400000,  # 1 day
        "description": "Interactive checkpoint for user validation",
    },
    "ORCHESTRATOR_CHECKPOINT_RESPONSE": {
        "topic": "dev.omninode-bridge.codegen.checkpoint-response.v1",
        "partitions": 1,
        "replication_factor": 1,
        "retention_ms": 86400000,
        "description": "User response to checkpoint",
    },
}


def get_kafka_container() -> str:
    """Get Kafka/Redpanda container name"""
    # Try common container names
    containers = [
        "omninode-bridge-redpanda",
        "omninode-bridge-kafka",
        "redpanda",
        "kafka",
    ]

    for container in containers:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            return result.stdout.strip()

    raise RuntimeError(
        "No Kafka/Redpanda container found. Please start services with docker-compose."
    )


def topic_exists(container: str, topic: str) -> bool:
    """Check if topic already exists using exact matching"""
    # Use rpk topic describe for exact matching instead of substring matching
    result = subprocess.run(
        ["docker", "exec", container, "rpk", "topic", "describe", topic],
        capture_output=True,
        text=True,
    )

    # Return code 0 means topic exists, non-zero means it doesn't
    return result.returncode == 0


def create_topic(container: str, config: dict) -> bool:
    """Create a single Kafka topic"""
    topic = config["topic"]
    partitions = config["partitions"]
    replication_factor = config["replication_factor"]
    retention_ms = config["retention_ms"]

    # Check if exists
    if topic_exists(container, topic):
        print(f"✓ Topic already exists: {topic}")
        return True

    # Create topic
    cmd = [
        "docker",
        "exec",
        container,
        "rpk",
        "topic",
        "create",
        topic,
        "--partitions",
        str(partitions),
        "--replicas",
        str(replication_factor),
        "--topic-config",
        f"retention.ms={retention_ms}",
    ]

    print(f"Creating topic: {topic}")
    print(f"  Partitions: {partitions}")
    print(f"  Replication: {replication_factor}")
    print(
        f"  Retention: {retention_ms}ms ({retention_ms / (1000 * 60 * 60 * 24):.0f} days)"
    )

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Created topic: {topic}\n")
        return True
    else:
        print(f"✗ Failed to create topic: {topic}")
        print(f"  Error: {result.stderr}\n")
        return False


def list_topics(container: str):
    """List all topics"""
    print("\n" + "=" * 80)
    print("Existing Kafka Topics")
    print("=" * 80)

    result = subprocess.run(
        ["docker", "exec", container, "rpk", "topic", "list"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Failed to list topics: {result.stderr}")


def verify_topics(container: str, topics: list[str]):
    """Verify all topics were created"""
    print("\n" + "=" * 80)
    print("Topic Verification")
    print("=" * 80)

    missing = []
    for topic_name, config in KAFKA_TOPICS.items():
        topic = config["topic"]
        if topic_exists(container, topic):
            print(f"✓ {topic_name}: {topic}")
        else:
            print(f"✗ {topic_name}: {topic} - MISSING")
            missing.append(topic)

    if missing:
        print(f"\n⚠️  Warning: {len(missing)} topics not found")
        return False
    else:
        print(f"\n✅ All {len(KAFKA_TOPICS)} topics verified!")
        return True


def main():
    """Main execution"""
    print("=" * 80)
    print("MVP Kafka Topic Creation")
    print("=" * 80)
    print(f"Creating {len(KAFKA_TOPICS)} topics for event-driven architecture\n")

    # Get environment
    environment = os.getenv("KAFKA_ENVIRONMENT", "dev")
    if len(sys.argv) > 1 and sys.argv[1] in ["dev", "staging", "prod"]:
        environment = sys.argv[1]

    print(f"Environment: {environment}")

    # Update topic names for environment
    if environment != "dev":
        for config in KAFKA_TOPICS.values():
            config["topic"] = config["topic"].replace("dev.", f"{environment}.")

    # Get Kafka container
    try:
        container = get_kafka_container()
        print(f"Kafka container: {container}\n")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create topics
    print("=" * 80)
    print("Creating Topics")
    print("=" * 80 + "\n")

    success_count = 0
    failure_count = 0

    for topic_name, config in KAFKA_TOPICS.items():
        print(f"[{success_count + failure_count + 1}/{len(KAFKA_TOPICS)}] {topic_name}")
        print(f"Description: {config['description']}")

        if create_topic(container, config):
            success_count += 1
        else:
            failure_count += 1

    # List all topics
    list_topics(container)

    # Verify creation
    all_verified = verify_topics(container, list(KAFKA_TOPICS.keys()))

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Created: {success_count}")
    print(f"✗ Failed: {failure_count}")
    print(f"Total: {len(KAFKA_TOPICS)}")

    if all_verified and failure_count == 0:
        print("\n✅ All topics created successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some topics failed to create")
        sys.exit(1)


if __name__ == "__main__":
    main()
