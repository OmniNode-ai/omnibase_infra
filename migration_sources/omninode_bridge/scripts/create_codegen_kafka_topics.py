#!/usr/bin/env python3
"""
Create Kafka Topics for Contract-First Code Generation Pipeline

Creates all topics for the autonomous code generation workflow:
- Request topics (analyze, validate, pattern, mixin)
- Response topics (analyze, validate, pattern, mixin)
- Status topic (session tracking with 6 partitions)
- Dead letter queue topics (4 DLQ topics with _dlq suffix)

Usage:
    poetry run python scripts/create_codegen_kafka_topics.py [--environment dev|staging|prod]
"""

import os
import subprocess
import sys

# Kafka topic configuration for code generation pipeline
KAFKA_TOPICS = {
    # Analysis request/response
    "CODEGEN_REQUEST_ANALYZE": {
        "topic": "dev.omninode-bridge.codegen.request-analyze.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "description": "PRD analysis requests from omniclaude to omniarchon",
    },
    "CODEGEN_RESPONSE_ANALYZE": {
        "topic": "dev.omninode-bridge.codegen.response-analyze.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "description": "PRD analysis responses from omniarchon to omniclaude",
    },
    # Validation request/response
    "CODEGEN_REQUEST_VALIDATE": {
        "topic": "dev.omninode-bridge.codegen.request-validate.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "description": "Code validation requests from omniclaude to omniarchon",
    },
    "CODEGEN_RESPONSE_VALIDATE": {
        "topic": "dev.omninode-bridge.codegen.response-validate.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "description": "Code validation responses from omniarchon to omniclaude",
    },
    # Pattern matching request/response
    "CODEGEN_REQUEST_PATTERN": {
        "topic": "dev.omninode-bridge.codegen.request-pattern.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "description": "Pattern matching requests from omniclaude to omniarchon",
    },
    "CODEGEN_RESPONSE_PATTERN": {
        "topic": "dev.omninode-bridge.codegen.response-pattern.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "description": "Pattern matching responses from omniarchon to omniclaude",
    },
    # Mixin recommendation request/response
    "CODEGEN_REQUEST_MIXIN": {
        "topic": "dev.omninode-bridge.codegen.request-mixin.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "description": "Mixin recommendation requests from omniclaude to omniarchon",
    },
    "CODEGEN_RESPONSE_MIXIN": {
        "topic": "dev.omninode-bridge.codegen.response-mixin.v1",
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "description": "Mixin recommendation responses from omniarchon to omniclaude",
    },
    # Session status (higher partition count for throughput)
    "CODEGEN_STATUS_SESSION": {
        "topic": "dev.omninode-bridge.codegen.status-session.v1",
        "partitions": 6,  # Higher throughput for status updates
        "replication_factor": 1,
        "retention_ms": 86400000,  # 1 day (shorter retention for status)
        "description": "Code generation session status updates",
    },
    # Dead letter queues
    "CODEGEN_REQUEST_ANALYZE_DLQ": {
        "topic": "dev.omninode-bridge.codegen.request-analyze.v1.dlq",
        "partitions": 1,  # DLQs typically need fewer partitions
        "replication_factor": 1,
        "retention_ms": 2592000000,  # 30 days (longer retention for debugging)
        "description": "Dead letter queue for failed analyze requests",
    },
    "CODEGEN_REQUEST_VALIDATE_DLQ": {
        "topic": "dev.omninode-bridge.codegen.request-validate.v1.dlq",
        "partitions": 1,
        "replication_factor": 1,
        "retention_ms": 2592000000,  # 30 days
        "description": "Dead letter queue for failed validate requests",
    },
    "CODEGEN_REQUEST_PATTERN_DLQ": {
        "topic": "dev.omninode-bridge.codegen.request-pattern.v1.dlq",
        "partitions": 1,
        "replication_factor": 1,
        "retention_ms": 2592000000,  # 30 days
        "description": "Dead letter queue for failed pattern requests",
    },
    "CODEGEN_REQUEST_MIXIN_DLQ": {
        "topic": "dev.omninode-bridge.codegen.request-mixin.v1.dlq",
        "partitions": 1,
        "replication_factor": 1,
        "retention_ms": 2592000000,  # 30 days
        "description": "Dead letter queue for failed mixin requests",
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
    print("Contract-First Code Generation Kafka Topic Creation")
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
