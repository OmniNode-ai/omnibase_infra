"""
Protocols for type safety in omninode_bridge.

This package provides Protocol definitions for structural typing
across the bridge infrastructure.

Protocols:
    SupportsQuery: Database connection protocol for query execution
    KafkaClientProtocol: Kafka client protocol for event publishing
"""

from omninode_bridge.protocols.protocol_database import SupportsQuery
from omninode_bridge.protocols.protocol_kafka_client import KafkaClientProtocol

__all__ = ["SupportsQuery", "KafkaClientProtocol"]
