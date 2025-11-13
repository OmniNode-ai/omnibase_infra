#!/usr/bin/env python3
"""
Create Kafka topics for agent performance metrics.

Creates 5 topics with proper configuration:
- dev.agent.metrics.routing.v1
- dev.agent.metrics.state-ops.v1
- dev.agent.metrics.coordination.v1
- dev.agent.metrics.workflow.v1
- dev.agent.metrics.ai-quorum.v1

Plus 1 alert topic:
- dev.agent.alerts.v1
"""

import asyncio
import logging
import os
import sys

from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import KafkaError, TopicAlreadyExistsError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Topic configurations
TOPICS = [
    {
        "name": "dev.agent.metrics.routing.v1",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
            "compression.type": "snappy",
            "cleanup.policy": "delete",
        },
    },
    {
        "name": "dev.agent.metrics.state-ops.v1",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
            "compression.type": "snappy",
            "cleanup.policy": "delete",
        },
    },
    {
        "name": "dev.agent.metrics.coordination.v1",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
            "compression.type": "snappy",
            "cleanup.policy": "delete",
        },
    },
    {
        "name": "dev.agent.metrics.workflow.v1",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
            "compression.type": "snappy",
            "cleanup.policy": "delete",
        },
    },
    {
        "name": "dev.agent.metrics.ai-quorum.v1",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
            "compression.type": "snappy",
            "cleanup.policy": "delete",
        },
    },
    {
        "name": "dev.agent.alerts.v1",
        "partitions": 3,
        "replication_factor": 1,
        "config": {
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
            "compression.type": "snappy",
            "cleanup.policy": "delete",
        },
    },
]


async def create_topics(
    bootstrap_servers: str,
    topics_config: list[dict],
    dry_run: bool = False,
) -> None:
    """
    Create Kafka topics.

    Args:
        bootstrap_servers: Kafka bootstrap servers
        topics_config: List of topic configurations
        dry_run: If True, only print topics without creating
    """
    if dry_run:
        logger.info("DRY RUN - Topics that would be created:")
        for topic in topics_config:
            logger.info(
                f"  - {topic['name']}: "
                f"{topic['partitions']} partitions, "
                f"replication={topic['replication_factor']}, "
                f"retention={topic['config']['retention.ms']}ms"
            )
        return

    admin_client = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers)

    try:
        await admin_client.start()
        logger.info(f"Connected to Kafka: {bootstrap_servers}")

        # Create NewTopic objects
        new_topics = []
        for topic_config in topics_config:
            new_topic = NewTopic(
                name=topic_config["name"],
                num_partitions=topic_config["partitions"],
                replication_factor=topic_config["replication_factor"],
                topic_configs=topic_config["config"],
            )
            new_topics.append(new_topic)

        # Create topics
        create_results = await admin_client.create_topics(
            new_topics=new_topics,
            validate_only=False,
        )

        # Check results
        for topic_config, (topic, error_code) in zip(
            topics_config, create_results.values(), strict=False
        ):
            if error_code == 0:
                logger.info(f"✅ Created topic: {topic_config['name']}")
            else:
                logger.error(
                    f"❌ Failed to create topic {topic_config['name']}: "
                    f"error_code={error_code}"
                )

    except TopicAlreadyExistsError as e:
        logger.warning(f"⚠️  Some topics already exist: {e}")
    except KafkaError as e:
        logger.error(f"❌ Kafka error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise
    finally:
        await admin_client.close()


async def main():
    """Main entry point."""
    # Get Kafka bootstrap servers from environment
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "192.168.86.200:29092",  # Default to remote server
    )

    # Check for dry run flag
    dry_run = "--dry-run" in sys.argv

    logger.info("=" * 60)
    logger.info("Agent Metrics Kafka Topics Creation")
    logger.info("=" * 60)
    logger.info(f"Bootstrap servers: {bootstrap_servers}")
    logger.info(f"Topics to create: {len(TOPICS)}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)

    try:
        await create_topics(bootstrap_servers, TOPICS, dry_run=dry_run)
        logger.info("✅ Topic creation completed successfully")
    except Exception as e:
        logger.error(f"❌ Topic creation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
