"""OmniNode Topic Specification Model for event bus routing."""

import os

from pydantic import BaseModel, Field

from omnibase_infra.enums.enum_omninode_topic_class import EnumOmniNodeTopicClass


class ModelOmniNodeTopicSpec(BaseModel):
    """
    OmniNode Topic Specification following five-tier hierarchy.

    Topic Format: <env>.<tenant>.<context>.<class>.<topic>.<v>
    Example: dev.omnibase.onex.evt.postgres-query-completed.v1
    """

    env: str = Field(
        default_factory=lambda: os.getenv("OMNINODE_ENV", "dev"),
        description="Environment: dev, staging, prod",
    )
    tenant: str = Field(
        default_factory=lambda: os.getenv("OMNINODE_TENANT", "omnibase"),
        description="Tenant identifier",
    )
    context: str = Field(
        default_factory=lambda: os.getenv("OMNINODE_CONTEXT", "onex"),
        description="Context/domain identifier",
    )
    topic_class: EnumOmniNodeTopicClass = Field(
        description="Topic class (evt, cmd, qrs, etc.)",
    )
    topic_name: str = Field(
        description="Specific topic name (kebab-case)",
    )
    version: str = Field(
        default="v1",
        description="Topic version",
    )

    def to_topic_string(self) -> str:
        """Generate the full topic string."""
        return f"{self.env}.{self.tenant}.{self.context}.{self.topic_class.value}.{self.topic_name}.{self.version}"

    @classmethod
    def for_postgres_query_completed(
        cls, correlation_id: str | None = None,
    ) -> "ModelOmniNodeTopicSpec":
        """Create topic spec for PostgreSQL query completed events."""
        return cls(
            topic_class=EnumOmniNodeTopicClass.EVT,
            topic_name="postgres-query-completed",
        )

    @classmethod
    def for_postgres_query_failed(
        cls, correlation_id: str | None = None,
    ) -> "ModelOmniNodeTopicSpec":
        """Create topic spec for PostgreSQL query failed events."""
        return cls(
            topic_class=EnumOmniNodeTopicClass.EVT,
            topic_name="postgres-query-failed",
        )

    @classmethod
    def for_postgres_health_check(cls) -> "ModelOmniNodeTopicSpec":
        """Create topic spec for PostgreSQL health check responses."""
        return cls(
            topic_class=EnumOmniNodeTopicClass.QRS,
            topic_name="postgres-health-response",
        )
