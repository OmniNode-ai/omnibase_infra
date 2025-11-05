"""Kafka topic registry for RedPanda integration.

Provides topic naming validation, environment-specific prefixes, and topic metadata
tracking for the PostgreSQL-RedPanda event bus integration. Implements a simple
registry system without schema registry (for now).

Following ONEX infrastructure patterns with strongly typed configuration.
"""

import logging
from datetime import datetime

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

from ...models.kafka import ModelTopicMetadata


class KafkaTopicRegistry:
    """
    Enterprise Kafka topic registry with validation and metadata tracking.

    Features:
    - Topic naming validation and standards enforcement
    - Environment-specific prefixes (dev, staging, prod)
    - Topic metadata tracking and discovery
    - Simple registry without schema validation (future enhancement)
    - Integration with ONEX infrastructure patterns
    """

    def __init__(
        self,
        environment: str = "dev",
        prefix_separator: str = ".",
    ):
        """Initialize Kafka topic registry.

        Args:
            environment: Environment prefix (dev, staging, prod)
            prefix_separator: Separator between environment and topic name
        """
        self.environment = environment
        self.prefix_separator = prefix_separator
        self.topics: dict[str, ModelTopicMetadata] = {}

        # Topic naming rules
        self.max_topic_length = 249
        self.valid_environments = {"dev", "staging", "prod", "test", "local"}

        # Logging
        self.logger = logging.getLogger(f"{__name__}.KafkaTopicRegistry")
        self.logger.info(f"Initialized topic registry for environment: {environment}")

    def register_topic(
        self,
        base_name: str,
        partition_count: int = 1,
        replication_factor: int = 1,
        description: str | None = None,
        owner: str | None = None,
        tags: list[str] | None = None,
    ) -> ModelTopicMetadata:
        """Register a new topic in the registry.

        Args:
            base_name: Base topic name without environment prefix
            partition_count: Number of partitions
            replication_factor: Replication factor
            description: Optional topic description
            owner: Optional topic owner/team
            tags: Optional topic tags

        Returns:
            Topic metadata
        """
        try:
            # Generate fully qualified topic name
            topic_name = self.build_topic_name(base_name)

            # Validate topic name
            validation_errors = self._validate_topic_name(topic_name, base_name)

            # Create metadata
            metadata = ModelTopicMetadata(
                topic_name=topic_name,
                base_name=base_name,
                environment=self.environment,
                partition_count=partition_count,
                replication_factor=replication_factor,
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors,
                description=description,
                owner=owner,
                tags=tags or [],
                created_at=datetime.now(),
            )

            # Store in registry
            self.topics[topic_name] = metadata

            self.logger.info(
                f"Registered topic '{topic_name}' (base: '{base_name}') "
                f"with {partition_count} partitions",
            )

            return metadata

        except Exception as e:
            self.logger.error(f"Error registering topic '{base_name}': {e}")
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Failed to register topic: {e!s}",
            ) from e

    def build_topic_name(self, base_name: str) -> str:
        """Build fully qualified topic name with environment prefix.

        Args:
            base_name: Base topic name without prefix

        Returns:
            Fully qualified topic name
        """
        return f"{self.environment}{self.prefix_separator}{base_name}"

    def parse_topic_name(self, topic_name: str) -> tuple[str, str]:
        """Parse topic name to extract environment and base name.

        Args:
            topic_name: Fully qualified topic name

        Returns:
            Tuple of (environment, base_name)
        """
        parts = topic_name.split(self.prefix_separator, 1)

        if len(parts) == 2 and parts[0] in self.valid_environments:
            return parts[0], parts[1]

        # No valid environment prefix found
        return self.environment, topic_name

    def get_topic(self, topic_name: str) -> ModelTopicMetadata:
        """Get topic metadata by name.

        Args:
            topic_name: Topic name to retrieve

        Returns:
            Topic metadata
        """
        if topic_name not in self.topics:
            raise OnexError(
                code=CoreErrorCode.RESOURCE_NOT_FOUND,
                message=f"Topic '{topic_name}' not found in registry",
            )

        return self.topics[topic_name]

    def list_topics(
        self,
        environment: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ModelTopicMetadata]:
        """List topics in the registry with optional filtering.

        Args:
            environment: Filter by environment
            tags: Filter by tags (any match)

        Returns:
            List of topic metadata
        """
        results = list(self.topics.values())

        # Filter by environment
        if environment:
            results = [t for t in results if t.environment == environment]

        # Filter by tags
        if tags:
            results = [t for t in results if any(tag in t.tags for tag in tags)]

        return results

    def validate_topic_name(self, topic_name: str) -> bool:
        """Validate a topic name against registry rules.

        Args:
            topic_name: Topic name to validate

        Returns:
            True if valid, False otherwise
        """
        environment, base_name = self.parse_topic_name(topic_name)
        validation_errors = self._validate_topic_name(topic_name, base_name)
        return len(validation_errors) == 0

    def _validate_topic_name(self, topic_name: str, base_name: str) -> list[str]:
        """Internal validation logic for topic names.

        Args:
            topic_name: Fully qualified topic name
            base_name: Base topic name

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check length
        if len(topic_name) > self.max_topic_length:
            errors.append(
                f"Topic name too long: {len(topic_name)} > {self.max_topic_length}",
            )

        # Check for invalid characters
        invalid_chars = set(topic_name) - set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.",
        )
        if invalid_chars:
            errors.append(f"Invalid characters in topic name: {invalid_chars}")

        # Check for internal topic prefix
        if base_name.startswith("_"):
            errors.append("Topic name cannot start with underscore (reserved)")

        # Check for dots-only name
        if base_name.strip(".") == "":
            errors.append("Topic name cannot consist only of dots")

        # Check for empty name
        if not base_name:
            errors.append("Topic base name cannot be empty")

        return errors

    def update_topic_metadata(
        self,
        topic_name: str,
        exists: bool | None = None,
        message_count: int | None = None,
        total_size_bytes: int | None = None,
    ) -> ModelTopicMetadata:
        """Update topic metadata with runtime information.

        Args:
            topic_name: Topic name to update
            exists: Whether topic exists in cluster
            message_count: Approximate message count
            total_size_bytes: Total size in bytes

        Returns:
            Updated topic metadata
        """
        if topic_name not in self.topics:
            raise OnexError(
                code=CoreErrorCode.RESOURCE_NOT_FOUND,
                message=f"Topic '{topic_name}' not found in registry",
            )

        metadata = self.topics[topic_name]

        if exists is not None:
            metadata.exists = exists

        if message_count is not None:
            metadata.message_count = message_count

        if total_size_bytes is not None:
            metadata.total_size_bytes = total_size_bytes

        metadata.last_updated = datetime.now()

        return metadata

    def remove_topic(self, topic_name: str) -> None:
        """Remove a topic from the registry.

        Args:
            topic_name: Topic name to remove
        """
        if topic_name not in self.topics:
            self.logger.warning(f"Topic '{topic_name}' not found in registry")
            return

        del self.topics[topic_name]
        self.logger.info(f"Removed topic '{topic_name}' from registry")

    def clear_registry(self) -> None:
        """Clear all topics from the registry."""
        count = len(self.topics)
        self.topics.clear()
        self.logger.info(f"Cleared {count} topics from registry")
