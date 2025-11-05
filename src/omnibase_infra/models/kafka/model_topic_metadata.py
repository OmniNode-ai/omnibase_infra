"""Topic metadata model for topic registry."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ModelTopicMetadata(BaseModel):
    """Kafka topic metadata and validation information.

    Tracks topic naming, configuration, and validation status for
    the topic registry system.
    """

    topic_name: str = Field(description="Fully qualified topic name")
    base_name: str = Field(description="Base topic name without environment prefix")
    environment: str = Field(description="Environment prefix: dev, staging, prod")

    # Topic configuration
    partition_count: int = Field(ge=1, description="Number of partitions")
    replication_factor: int = Field(ge=1, description="Replication factor")

    # Validation
    is_valid: bool = Field(default=True, description="Whether topic name is valid")
    validation_errors: list[str] = Field(
        default_factory=list,
        description="List of validation errors if invalid",
    )

    # Metadata
    description: str | None = Field(default=None, description="Topic description")
    owner: str | None = Field(default=None, description="Topic owner/team")
    tags: list[str] = Field(default_factory=list, description="Topic tags for organization")

    # Status
    exists: bool = Field(default=False, description="Whether topic exists in cluster")
    created_at: datetime | None = Field(
        default=None,
        description="Topic creation timestamp",
    )
    last_updated: datetime | None = Field(
        default=None,
        description="Last metadata update timestamp",
    )

    # Message statistics (if available)
    message_count: int | None = Field(
        default=None,
        ge=0,
        description="Approximate message count",
    )
    total_size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Total size in bytes",
    )

    @field_validator("topic_name")
    @classmethod
    def validate_topic_name(cls, v: str) -> str:
        """Validate topic name format.

        Args:
            v: Topic name to validate

        Returns:
            Validated topic name

        Raises:
            ValueError: If topic name is invalid
        """
        if not v:
            raise ValueError("Topic name cannot be empty")

        if len(v) > 249:
            raise ValueError("Topic name too long (max 249 characters)")

        # Check for valid characters (alphanumeric, hyphen, underscore, dot)
        invalid_chars = set(v) - set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        if invalid_chars:
            raise ValueError(f"Topic name contains invalid characters: {invalid_chars}")

        # Cannot start with underscore (Kafka internal topics)
        if v.startswith("_"):
            raise ValueError("Topic name cannot start with underscore (reserved for internal topics)")

        # Cannot be just dots
        if v.strip(".") == "":
            raise ValueError("Topic name cannot consist only of dots")

        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment prefix.

        Args:
            v: Environment prefix to validate

        Returns:
            Validated environment prefix

        Raises:
            ValueError: If environment is invalid
        """
        valid_environments = {"dev", "staging", "prod", "test", "local"}
        if v not in valid_environments:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_environments}")
        return v

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
