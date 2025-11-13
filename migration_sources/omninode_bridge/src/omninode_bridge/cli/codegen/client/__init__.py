"""
Kafka client for CLI operations.

Provides simplified Kafka producer/consumer interface for CLI tools.
"""

from .kafka_client import CLIKafkaClient

__all__ = ["CLIKafkaClient"]
