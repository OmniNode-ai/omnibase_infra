"""Kafka message format enumeration."""

from enum import Enum


class EnumKafkaMessageFormat(str, Enum):
    """Kafka message format enumeration."""

    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    STRING = "string"
    BINARY = "binary"
    XML = "xml"
