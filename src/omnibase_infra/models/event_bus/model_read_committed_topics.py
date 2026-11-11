# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The topics a Kafka consumer reads committed records only (OMN-22430).

An aborted transaction on a topic of the host desired/observed state plan must
never reach a reader. The declaration is DATA: ``read_committed_topics.yaml``
lists the topics, shipped as a resource of the wheel. aiokafka's
``isolation_level`` is a consumer setting, not a field of the subscription
contract (which forbids extra fields), so the setting is infra-owned:
``consumer_kwargs`` is the one place it is spelled and every consumer
constructor of the event bus calls it.
"""

from __future__ import annotations

import re
from importlib import resources

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_RESOURCE_PACKAGE = "omnibase_infra.event_bus"
_RESOURCE_NAME = "read_committed_topics.yaml"
_TOPIC_NAME = re.compile(r"^onex\.evt\.omnimarket\.[a-z][a-z0-9-]*\.v[0-9]+$")
_ISOLATION_LEVEL = "read_committed"

# The plan's declared event topics, built from parts so no full topic name is
# spelled in code. The declaration may list any of these and nothing else.
_PREFIX = ".".join(("onex", "evt", "omnimarket"))
_DECLARED_TOPICS = frozenset(
    f"{_PREFIX}.{name}.v1"
    for name in ("host-layer-decided", "host-desired-state", "host-observed-state")
)


class ModelReadCommittedTopics(BaseModel):
    """The validated declaration of read committed topics."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    schema_version: str = Field(min_length=1)
    topics: tuple[str, ...] = Field(min_length=1)

    @field_validator("topics")
    @classmethod
    def _topics_are_declared(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for topic in value:
            if not _TOPIC_NAME.match(topic):
                raise ValueError(f"{topic!r} is not a well-formed event topic name")
            if topic not in _DECLARED_TOPICS:
                raise ValueError(f"{topic!r} is not one of the declared event topics")
        if len(set(value)) != len(value):
            raise ValueError("a topic is listed twice")
        return value

    @classmethod
    def load(cls) -> ModelReadCommittedTopics:
        """Read the packaged declaration; an unavailable resource fails startup."""
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="load_read_committed_topics",
        )
        try:
            text = (
                resources.files(_RESOURCE_PACKAGE)
                .joinpath(_RESOURCE_NAME)
                .read_text(encoding="utf-8")
            )
            return cls.model_validate(yaml.safe_load(text))
        except (OSError, ModuleNotFoundError, yaml.YAMLError, ValueError) as exc:
            raise ProtocolConfigurationError(
                f"read committed topic declaration {_RESOURCE_NAME} is unavailable "
                f"or invalid: {type(exc).__name__}",
                context=context,
            ) from exc

    def consumer_kwargs(self, topic: str) -> dict[str, str]:
        """Return the consumer setting for ``topic`` (canonical, not namespaced)."""
        return {"isolation_level": _ISOLATION_LEVEL} if topic in self.topics else {}
