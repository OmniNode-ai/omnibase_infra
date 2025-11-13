#!/usr/bin/env python3
"""Example usage of CodegenDLQMonitor for tracking failed code generation events.

This example demonstrates:
1. Initializing the DLQ monitor
2. Getting DLQ statistics
3. Configuring alert thresholds
4. Monitoring lifecycle management

For full functionality, ensure Kafka/Redpanda is running with DLQ topics created.

Usage:
    poetry run python examples/codegen_dlq_monitor_example.py
"""

import asyncio

from omninode_bridge.monitoring import CodegenDLQMonitor


async def main():
    """Demonstrate CodegenDLQMonitor usage."""
    print("CodegenDLQMonitor Example")
    print("=" * 60)

    # Example 1: Initialize with default configuration
    print("\n1. Initializing DLQ Monitor (default config)...")
    monitor = CodegenDLQMonitor(
        kafka_config={"bootstrap_servers": "localhost:29092"},
        alert_threshold=10,
    )
    print(f"   Monitor initialized: {monitor}")
    print(f"   Alert threshold: {monitor.alert_threshold}")
    print(f"   Monitored topics: {monitor.DLQ_TOPICS}")

    # Example 2: Get initial statistics
    print("\n2. Getting DLQ Statistics...")
    stats = await monitor.get_dlq_stats()
    print(f"   Total DLQ messages: {stats['total_dlq_messages']}")
    print(f"   Is monitoring: {stats['is_monitoring']}")
    print(f"   Per-topic counts: {stats['dlq_counts']}")

    # Example 3: Custom configuration with environment variables
    print("\n3. Custom Configuration Example...")
    import os

    os.environ["DLQ_ALERT_THRESHOLD"] = "25"
    custom_monitor = CodegenDLQMonitor()
    print(f"   Custom threshold from env: {custom_monitor.alert_threshold}")

    # Example 4: Demonstrate start/stop (without actually connecting to Kafka)
    print("\n4. Lifecycle Management...")
    print(f"   Monitor running: {monitor.is_running}")
    print("   (Note: Call monitor.start_monitoring() to begin consuming DLQ messages)")
    print("   (Note: Call monitor.stop_monitoring() for graceful shutdown)")

    # Example 5: Demonstrate statistics format
    print("\n5. Statistics Format Example...")
    print("   Statistics include:")
    print("   - dlq_counts: Per-topic message counts")
    print("   - total_dlq_messages: Sum of all DLQ messages")
    print("   - alert_threshold: Configured alert threshold")
    print("   - is_monitoring: Whether monitor is currently running")
    print("   - last_alert_times: Timestamps of last alerts per topic")
    print("   - timestamp: Current timestamp")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nTo run with live Kafka monitoring:")
    print("  1. Ensure Kafka/Redpanda is running (docker-compose up)")
    print("  2. Create DLQ topics (see MVP plan)")
    print("  3. Call: await monitor.start_monitoring()")
    print("  4. DLQ messages will be automatically tracked and alerted")


if __name__ == "__main__":
    asyncio.run(main())
