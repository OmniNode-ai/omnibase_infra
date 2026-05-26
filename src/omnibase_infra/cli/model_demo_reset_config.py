# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Configuration model for the demo reset engine.

.. versionadded:: 0.9.1
"""

from __future__ import annotations

import os
import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

__all__: list[str] = [
    "ModelDemoResetConfig",
]

# Default values imported as module-level constants to avoid circular imports.
# These match the constants in ``service_demo_reset.py``.
_DEFAULT_PROJECTION_TABLE: Final[str] = "registration_projections"

_DEFAULT_CONSUMER_GROUP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(registration|projector|introspection)", re.IGNORECASE
)

_DEFAULT_TOPIC_PREFIXES: Final[tuple[str, ...]] = (
    "onex.evt.platform.",  # onex-topic-allow: pending contract auto-wiring
    "onex.cmd.platform.",  # onex-topic-allow: pending contract auto-wiring
    "onex.evt.omniintelligence.",  # onex-topic-allow: pending contract auto-wiring
    "onex.cmd.omniintelligence.",  # onex-topic-allow: pending contract auto-wiring
    "onex.evt.omniclaude.",  # onex-topic-allow: pending contract auto-wiring
    # "onex.evt.agent." removed: agent-status topic renamed to onex.evt.omniclaude.agent-status.v1  # onex-topic-allow: pending contract auto-wiring
    # which is already covered by the "onex.evt.omniclaude." prefix (OMN-2846).  # onex-topic-allow: pending contract auto-wiring
)


class ModelDemoResetConfig(BaseModel):
    """Configuration for the demo reset engine.

    Note:
        ``consumer_group_pattern`` is typed as ``re.Pattern`` which Pydantic
        handles natively via its ``Pattern`` validator.

    Attributes:
        postgres_dsn: PostgreSQL connection string.
        kafka_bootstrap_servers: Kafka broker address(es).
        purge_topics: Whether to delete messages from demo topics.
        projection_table: Table name for projector state.
        consumer_group_pattern: Regex to match demo consumer groups.
        demo_topic_prefixes: Topic prefixes considered demo-scoped.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    postgres_dsn: str = ""
    kafka_bootstrap_servers: str = ""
    purge_topics: bool = False
    projection_table: str = _DEFAULT_PROJECTION_TABLE
    consumer_group_pattern: re.Pattern[str] = Field(
        default=_DEFAULT_CONSUMER_GROUP_PATTERN
    )
    demo_topic_prefixes: tuple[str, ...] = _DEFAULT_TOPIC_PREFIXES

    @classmethod
    def from_env(cls, *, purge_topics: bool = False) -> ModelDemoResetConfig:
        """Create config from environment variables.

        Reads OMNIBASE_INFRA_DB_URL and KAFKA_BOOTSTRAP_SERVERS from
        the environment. Falls back to empty strings if not set.

        Args:
            purge_topics: Whether to purge demo topic data.

        Returns:
            ModelDemoResetConfig populated from environment.
        """
        return cls(
            postgres_dsn=os.environ.get("OMNIBASE_INFRA_DB_URL", ""),
            kafka_bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", ""),
            purge_topics=purge_topics,
        )
