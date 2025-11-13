#!/usr/bin/env python3
"""
Validate Redpanda Codegen Topics

This script validates that all 13 codegen topics are created with correct configurations.
Can be run repeatedly to check cluster health.
"""

import subprocess
import sys

# Expected topic configurations
EXPECTED_TOPICS = {
    # Request topics (3 partitions, 7 day retention)
    "omninode_codegen_request_analyze_v1": {
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,  # 7 days
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    "omninode_codegen_request_validate_v1": {
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    "omninode_codegen_request_pattern_v1": {
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    "omninode_codegen_request_mixin_v1": {
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    # Response topics (3 partitions, 7 day retention)
    "omninode_codegen_response_analyze_v1": {
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    "omninode_codegen_response_validate_v1": {
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    "omninode_codegen_response_pattern_v1": {
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    "omninode_codegen_response_mixin_v1": {
        "partitions": 3,
        "replication_factor": 1,
        "retention_ms": 604800000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    # Status topic (6 partitions, 3 day retention)
    "omninode_codegen_status_session_v1": {
        "partitions": 6,
        "replication_factor": 1,
        "retention_ms": 259200000,  # 3 days
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    # DLQ topics (1 partition, 30 day retention)
    "omninode_codegen_dlq_analyze_v1": {
        "partitions": 1,
        "replication_factor": 1,
        "retention_ms": 2592000000,  # 30 days
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    "omninode_codegen_dlq_validate_v1": {
        "partitions": 1,
        "replication_factor": 1,
        "retention_ms": 2592000000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    "omninode_codegen_dlq_pattern_v1": {
        "partitions": 1,
        "replication_factor": 1,
        "retention_ms": 2592000000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
    "omninode_codegen_dlq_mixin_v1": {
        "partitions": 1,
        "replication_factor": 1,
        "retention_ms": 2592000000,
        "cleanup_policy": "delete",
        "compression_type": "gzip",
    },
}


def run_rpk_command(args: list[str]) -> str:
    """Execute rpk command in Redpanda container."""
    cmd = ["docker", "exec", "omninode-bridge-redpanda", "rpk"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"rpk command failed: {result.stderr}")
    return result.stdout


def check_redpanda_health() -> bool:
    """Check if Redpanda is healthy."""
    try:
        output = run_rpk_command(["cluster", "health"])
        # Check for "Healthy:                          true" (with spaces)
        return "true" in output and "Healthy" in output
    except Exception as e:
        print(f"❌ Redpanda health check failed: {e}")
        return False


def list_topics() -> list[str]:
    """List all topics in the cluster."""
    output = run_rpk_command(["topic", "list"])
    topics = []
    for line in output.strip().split("\n")[1:]:  # Skip header
        if line.strip():
            # Format: NAME  PARTITIONS  REPLICAS
            parts = line.split()
            if parts:
                topics.append(parts[0])
    return topics


def get_topic_details(topic_name: str) -> dict:
    """Get detailed information about a topic."""
    output = run_rpk_command(
        ["topic", "describe", topic_name, "--print-partitions=false"]
    )

    details = {"partitions": 0, "replication_factor": 0, "configs": {}}

    for line in output.strip().split("\n"):
        if "PARTITIONS" in line:
            parts = line.split()
            if len(parts) >= 2:
                details["partitions"] = int(parts[1])
        elif "REPLICAS" in line:
            parts = line.split()
            if len(parts) >= 2:
                details["replication_factor"] = int(parts[1])
        elif "KEY" in line and "VALUE" in line:
            # Config section header
            continue
        elif line.strip() and not any(
            x in line for x in ["NAME", "PARTITIONS", "REPLICAS", "CONFIGS"]
        ):
            # Config line: KEY  VALUE  SOURCE
            parts = line.split(None, 2)
            if len(parts) >= 2:
                key = parts[0].strip()
                value = parts[1].strip()
                details["configs"][key] = value

    return details


def validate_topic_config(
    topic_name: str, expected: dict, actual: dict
) -> tuple[bool, list[str]]:
    """Validate topic configuration matches expected values."""
    issues = []

    # Check partitions
    if actual["partitions"] != expected["partitions"]:
        issues.append(
            f"  ❌ Partitions: expected {expected['partitions']}, got {actual['partitions']}"
        )

    # Check replication factor
    if actual["replication_factor"] != expected["replication_factor"]:
        issues.append(
            f"  ❌ Replication: expected {expected['replication_factor']}, got {actual['replication_factor']}"
        )

    # Check configs
    configs = actual["configs"]

    # Retention
    if "retention.ms" in configs:
        actual_retention = int(configs["retention.ms"])
        if actual_retention != expected["retention_ms"]:
            issues.append(
                f"  ❌ Retention: expected {expected['retention_ms']}ms, got {actual_retention}ms"
            )
    else:
        issues.append("  ❌ Retention: not configured")

    # Cleanup policy
    if "cleanup.policy" in configs:
        if configs["cleanup.policy"] != expected["cleanup_policy"]:
            issues.append(
                f"  ❌ Cleanup: expected {expected['cleanup_policy']}, got {configs['cleanup.policy']}"
            )
    else:
        issues.append("  ❌ Cleanup policy: not configured")

    # Compression type
    if "compression.type" in configs:
        if configs["compression.type"] != expected["compression_type"]:
            issues.append(
                f"  ❌ Compression: expected {expected['compression_type']}, got {configs['compression.type']}"
            )
    else:
        issues.append("  ❌ Compression: not configured")

    return len(issues) == 0, issues


def test_produce_consume(
    topic_name: str = "omninode_codegen_request_analyze_v1",
) -> bool:
    """Test basic produce/consume functionality.

    This test validates end-to-end Kafka functionality by:
    1. Producing a unique test message
    2. Verifying successful production with partition and offset info
    3. Consuming recent messages and verifying our test message is present

    Note: For multi-partition topics, successful production validates the system is working.
    """
    import re
    import time
    import uuid

    # Use unique test message to avoid conflicts with existing messages
    test_id = str(uuid.uuid4())[:8]
    test_message = f"test_validation_message_{test_id}"

    try:
        # Produce message
        cmd = [
            "docker",
            "exec",
            "-i",
            "omninode-bridge-redpanda",
            "rpk",
            "topic",
            "produce",
            topic_name,
            "--key",
            "validation_test",
        ]
        # Add newline to test message for proper rpk input formatting
        result = subprocess.run(
            cmd, input=f"{test_message}\n", capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"❌ Failed to produce test message: {result.stderr}")
            return False

        # Extract partition and offset from produce output
        # Output format: "Produced to partition X at offset Y with timestamp Z."
        # Check both stdout and stderr as rpk might output to either
        produce_output = result.stdout + result.stderr
        partition_match = re.search(r"partition (\d+)", produce_output)
        offset_match = re.search(r"offset (\d+)", produce_output)

        if not partition_match or not offset_match:
            print("❌ Failed to parse produce output")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False

        partition = int(partition_match.group(1))
        offset = int(offset_match.group(1))

        print(
            f"✅ Successfully produced test message to {topic_name} (partition={partition}, offset={offset})"
        )

        # Wait for message to be committed
        time.sleep(0.5)

        # Consume the specific message we just produced from its partition and offset
        try:
            cmd = [
                "docker",
                "exec",
                "omninode-bridge-redpanda",
                "rpk",
                "topic",
                "consume",
                topic_name,
                "--partitions",
                str(partition),
                "--offset",
                f"{offset}:{offset+1}",  # Read exactly our message using offset range
                "--format",
                "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)

            if result.returncode != 0:
                print(f"❌ Failed to consume test message: {result.stderr}")
                return False

            output = result.stdout

            # Verify the test message was consumed
            if test_message in output or test_id in output:
                print(f"✅ Successfully consumed test message from {topic_name}")
                return True
            else:
                print("❌ Test message not found in consumed output")
                print(f"   Expected substring: {test_id}")
                print(f"   Got: {output[:300]}...")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Consume operation timed out")
            return False

    except Exception as e:
        print(f"❌ Produce/consume test failed: {e}")
        return False


def main():
    """Main validation routine."""
    print("=" * 70)
    print("Redpanda Codegen Topics Validation")
    print("=" * 70)
    print()

    # Step 1: Check Redpanda health
    print("1. Checking Redpanda health...")
    if not check_redpanda_health():
        print("❌ Redpanda is not healthy. Exiting.")
        sys.exit(1)
    print("✅ Redpanda is healthy")
    print()

    # Step 2: List all topics
    print("2. Listing topics...")
    try:
        all_topics = list_topics()
        codegen_topics = [t for t in all_topics if "omninode_codegen" in t]
        print(f"✅ Found {len(codegen_topics)} codegen topics")
        print()
    except Exception as e:
        print(f"❌ Failed to list topics: {e}")
        sys.exit(1)

    # Step 3: Validate each topic
    print("3. Validating topic configurations...")
    print()

    all_valid = True
    missing_topics = []

    for topic_name, expected_config in EXPECTED_TOPICS.items():
        if topic_name not in codegen_topics:
            missing_topics.append(topic_name)
            all_valid = False
            continue

        print(f"   {topic_name}")

        try:
            actual_config = get_topic_details(topic_name)
            is_valid, issues = validate_topic_config(
                topic_name, expected_config, actual_config
            )

            if is_valid:
                print(
                    f"   ✅ Partitions: {actual_config['partitions']}, "
                    f"Replicas: {actual_config['replication_factor']}"
                )
            else:
                all_valid = False
                for issue in issues:
                    print(issue)
        except Exception as e:
            print(f"   ❌ Failed to validate: {e}")
            all_valid = False

        print()

    if missing_topics:
        print("❌ Missing topics:")
        for topic in missing_topics:
            print(f"   - {topic}")
        print()

    # Step 4: Test produce/consume
    print("4. Testing produce/consume functionality...")
    produce_consume_passed = test_produce_consume()
    if not produce_consume_passed:
        print("❌ Produce/consume test failed - end-to-end flow validation failed")
        all_valid = False
    print()

    # Summary
    print("=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print("Total topics expected: 13")
    print(f"Total topics found: {len(codegen_topics)}")
    print(f"Missing topics: {len(missing_topics)}")
    print()

    if all_valid and len(missing_topics) == 0:
        print("✅ All validations passed!")
        print()
        print("Topic Breakdown:")
        print("  - 4 request topics (3 partitions each, 7 day retention)")
        print("  - 4 response topics (3 partitions each, 7 day retention)")
        print("  - 1 status topic (6 partitions, 3 day retention)")
        print("  - 4 DLQ topics (1 partition each, 30 day retention)")
        sys.exit(0)
    else:
        print("❌ Validation failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
