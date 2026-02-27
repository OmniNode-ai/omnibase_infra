# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Skill lifecycle observability consumer (OMN-2934).

Consumes skill-started and skill-completed events from Kafka and persists
them to PostgreSQL for omnidash skill monitoring pages.
"""

from omnibase_infra.services.observability.skill_lifecycle.config import (
    ConfigSkillLifecycleConsumer,
)
from omnibase_infra.services.observability.skill_lifecycle.consumer import (
    SkillLifecycleConsumer,
)
from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
    WriterSkillLifecyclePostgres,
)

__all__ = [
    "ConfigSkillLifecycleConsumer",
    "SkillLifecycleConsumer",
    "WriterSkillLifecyclePostgres",
]
