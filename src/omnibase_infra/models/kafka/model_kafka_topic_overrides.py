"""Kafka topic configuration overrides model."""

from typing import Optional
from pydantic import BaseModel, Field


class ModelKafkaTopicOverrides(BaseModel):
    """Kafka topic configuration overrides model."""
    
    # Retention settings
    retention_ms: Optional[int] = Field(
        default=None, 
        description="Message retention time in milliseconds"
    )
    retention_bytes: Optional[int] = Field(
        default=None, 
        description="Maximum size of log before deleting old log segments"
    )
    
    # Segment settings
    segment_ms: Optional[int] = Field(
        default=None, 
        description="Log segment time in milliseconds"
    )
    segment_bytes: Optional[int] = Field(
        default=None, 
        description="Log segment size in bytes"
    )
    segment_jitter_ms: Optional[int] = Field(
        default=None, 
        description="Maximum jitter to subtract from segment.ms"
    )
    
    # Message settings
    max_message_bytes: Optional[int] = Field(
        default=None, 
        description="Maximum size of a message in bytes"
    )
    message_format_version: Optional[str] = Field(
        default=None, 
        description="Message format version"
    )
    message_timestamp_type: Optional[str] = Field(
        default=None, 
        description="Message timestamp type (CreateTime or LogAppendTime)"
    )
    message_timestamp_difference_max_ms: Optional[int] = Field(
        default=None, 
        description="Maximum difference between message timestamp and broker timestamp"
    )
    
    # Compression settings
    compression_type: Optional[str] = Field(
        default=None, 
        description="Compression type (uncompressed, gzip, snappy, lz4, zstd)"
    )
    
    # Cleanup settings
    cleanup_policy: Optional[str] = Field(
        default=None, 
        description="Log cleanup policy (delete, compact, or delete,compact)"
    )
    delete_retention_ms: Optional[int] = Field(
        default=None, 
        description="Time to retain delete tombstone markers"
    )
    min_cleanable_dirty_ratio: Optional[float] = Field(
        default=None, 
        description="Minimum ratio of dirty log to total log for compaction"
    )
    min_compaction_lag_ms: Optional[int] = Field(
        default=None, 
        description="Minimum time a message will remain uncompacted"
    )
    max_compaction_lag_ms: Optional[int] = Field(
        default=None, 
        description="Maximum time a message will remain uncompacted"
    )
    
    # Replication settings
    min_in_sync_replicas: Optional[int] = Field(
        default=None, 
        description="Minimum number of replicas that must acknowledge a write"
    )
    unclean_leader_election_enable: Optional[bool] = Field(
        default=None, 
        description="Enable unclean leader election"
    )
    
    # Index settings
    index_interval_bytes: Optional[int] = Field(
        default=None, 
        description="Number of bytes between index entries"
    )
    
    # Flush settings
    flush_messages: Optional[int] = Field(
        default=None, 
        description="Number of messages to accumulate before forcing a flush"
    )
    flush_ms: Optional[int] = Field(
        default=None, 
        description="Maximum time to wait before forcing a flush"
    )
    
    # Follower settings
    follower_replication_throttled_replicas: Optional[str] = Field(
        default=None, 
        description="List of follower replicas for throttling"
    )
    leader_replication_throttled_replicas: Optional[str] = Field(
        default=None, 
        description="List of leader replicas for throttling"
    )
    
    # File settings
    file_delete_delay_ms: Optional[int] = Field(
        default=None, 
        description="Time to wait before deleting a file from filesystem"
    )
    
    # Pre-allocation settings
    preallocate: Optional[bool] = Field(
        default=None, 
        description="Pre-allocate disk space for log segments"
    )