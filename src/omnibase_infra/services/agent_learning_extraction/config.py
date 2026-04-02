# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Configuration for agent learning extraction consumer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.topics.platform_topic_suffixes import (
    SUFFIX_OMNICLAUDE_SESSION_ENDED,
    SUFFIX_OMNICLAUDE_TOOL_EXECUTED,
)


class ModelAgentLearningExtractionConfig(BaseModel):
    """Config for the learning extraction consumer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_ended_topic: str = Field(
        default=SUFFIX_OMNICLAUDE_SESSION_ENDED,
        description="Topic to consume session-ended events from",
    )
    tool_executed_topic: str = Field(
        default=SUFFIX_OMNICLAUDE_TOOL_EXECUTED,
        description="Topic to consume tool-executed events from",
    )
    consumer_group: str = Field(
        default="local.omnibase-infra.agent-learning-extraction.consume.v1",
        description="Kafka consumer group ID",
    )
    llm_summary_url: str = Field(
        default="http://localhost:8001/v1/chat/completions",
        description="Qwen3-14B endpoint for generating resolution summaries",
    )
    llm_summary_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Timeout for LLM summary generation",
    )
    tool_event_buffer_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="How long to buffer tool-executed events waiting for session-ended",
    )
    min_tools_used: int = Field(
        default=1,
        ge=0,
        description="Minimum tools_used_count in session-ended event to qualify",
    )
