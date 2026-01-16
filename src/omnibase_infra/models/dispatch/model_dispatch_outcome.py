# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dispatch outcome model for representing dispatcher output topics.

This model replaces the union pattern `str | list[str] | None` that was used
for dispatcher return types. By using a single model type with a list field,
we eliminate the 3-way union while preserving all functionality.

.. versionadded:: 0.6.0
    Created as part of Union Reduction Phase 2 (OMN-1002).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult


class ModelDispatchOutcome(BaseModel):
    """Outcome of a dispatch operation containing zero or more output topics.

    This model normalizes dispatcher outputs into a consistent structure:
    - Empty list: No output topics (equivalent to returning None)
    - Single-item list: One output topic (equivalent to returning str)
    - Multi-item list: Multiple output topics (equivalent to returning list[str])

    Attributes:
        topics: List of output topics produced by the dispatcher.
            Empty list indicates no outputs.

    Example:
        >>> # No output topics
        >>> outcome = ModelDispatchOutcome()
        >>> outcome.is_empty
        True

        >>> # Single output topic
        >>> outcome = ModelDispatchOutcome(topics=["dev.user.created.v1"])
        >>> outcome.single_topic
        'dev.user.created.v1'

        >>> # Multiple output topics
        >>> outcome = ModelDispatchOutcome(
        ...     topics=["dev.user.created.v1", "dev.audit.log.v1"]
        ... )
        >>> len(outcome.topics)
        2

    .. versionadded:: 0.6.0
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topics: list[str] = Field(
        default_factory=list,
        description="Output topics produced by the dispatcher. Empty list means no outputs.",
    )

    @overload
    @classmethod
    def from_legacy_output(cls, output: None) -> ModelDispatchOutcome: ...

    @overload
    @classmethod
    def from_legacy_output(cls, output: str) -> ModelDispatchOutcome: ...

    @overload
    @classmethod
    def from_legacy_output(cls, output: list[str]) -> ModelDispatchOutcome: ...

    @overload
    @classmethod
    def from_legacy_output(
        cls, output: ModelDispatchResult
    ) -> ModelDispatchOutcome: ...

    @classmethod
    def from_legacy_output(cls, output: object) -> ModelDispatchOutcome:
        """Create from legacy dispatcher output format.

        This factory method handles the conversion from the old 3-way union
        pattern to the new unified model structure. The implementation uses
        ``object`` type with runtime checks to avoid adding unions; callers
        should use the overloaded signatures for type safety.

        Args:
            output: Legacy dispatcher output which can be:
                - str: Single output topic
                - list[str]: Multiple output topics
                - None: No output topics
                - ModelDispatchResult: Protocol-based dispatcher result

        Returns:
            ModelDispatchOutcome with normalized topics list.

        Raises:
            ProtocolConfigurationError: If output is not str, list[str], None, or ModelDispatchResult.

        Example:
            >>> ModelDispatchOutcome.from_legacy_output(None).topics
            []
            >>> ModelDispatchOutcome.from_legacy_output("topic.v1").topics
            ['topic.v1']
            >>> ModelDispatchOutcome.from_legacy_output(["a.v1", "b.v1"]).topics
            ['a.v1', 'b.v1']

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Changed implementation signature from union to ``object`` with
            overloads for type safety. This supports union reduction (OMN-1002).
        .. versionchanged:: 0.6.2
            Added support for ModelDispatchResult to allow protocol-based
            dispatchers to be registered with MessageDispatchEngine.
        """
        if output is None:
            return cls(topics=[])
        elif isinstance(output, str):
            return cls(topics=[output])
        elif isinstance(output, list):
            return cls(topics=list(output))
        else:
            # Check for ModelDispatchResult by duck-typing (has 'outputs' attribute)
            # to avoid circular import at runtime
            if hasattr(output, "outputs"):
                outputs_attr = getattr(output, "outputs", None)
                if outputs_attr is not None and hasattr(outputs_attr, "topics"):
                    return cls(topics=list(outputs_attr.topics))
                return cls(topics=[])
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="from_legacy_output",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                f"Expected str, list[str], None, or ModelDispatchResult, "
                f"got {type(output).__name__}",
                context=context,
            )

    @classmethod
    def none(cls) -> ModelDispatchOutcome:
        """Create an outcome representing no output topics.

        Returns:
            ModelDispatchOutcome with empty topics list.

        Example:
            >>> outcome = ModelDispatchOutcome.none()
            >>> outcome.is_empty
            True

        .. versionadded:: 0.6.0
        """
        return cls(topics=[])

    @classmethod
    def single(cls, topic: str) -> ModelDispatchOutcome:
        """Create an outcome with a single output topic.

        Args:
            topic: The output topic.

        Returns:
            ModelDispatchOutcome with single topic.

        Example:
            >>> outcome = ModelDispatchOutcome.single("dev.user.created.v1")
            >>> outcome.single_topic
            'dev.user.created.v1'

        .. versionadded:: 0.6.0
        """
        return cls(topics=[topic])

    @classmethod
    def multiple(cls, topics: list[str]) -> ModelDispatchOutcome:
        """Create an outcome with multiple output topics.

        Args:
            topics: List of output topics.

        Returns:
            ModelDispatchOutcome with multiple topics.

        Example:
            >>> outcome = ModelDispatchOutcome.multiple(["a.v1", "b.v1"])
            >>> len(outcome.topics)
            2

        .. versionadded:: 0.6.0
        """
        return cls(topics=list(topics))

    @property
    def single_topic(self) -> str:
        """Get the single topic if exactly one exists.

        Returns:
            The topic string if exactly one topic, empty string otherwise.
            Use ``has_single_topic`` to check if the outcome has exactly one topic
            before calling this property.

        Note:
            Empty string is used as a sentinel value to indicate that either
            no topics exist or multiple topics exist. This eliminates the
            ``str | None`` union pattern while preserving functionality.

        Example:
            >>> ModelDispatchOutcome(topics=["topic.v1"]).single_topic
            'topic.v1'
            >>> ModelDispatchOutcome(topics=[]).single_topic
            ''
            >>> ModelDispatchOutcome(topics=["a.v1", "b.v1"]).single_topic
            ''

        .. versionadded:: 0.6.0
        .. versionchanged:: 0.6.1
            Changed return type from ``str | None`` to ``str`` with empty string
            sentinel for union reduction (OMN-1002).
        """
        return self.topics[0] if len(self.topics) == 1 else ""

    @property
    def has_single_topic(self) -> bool:
        """Check if exactly one topic was produced.

        This property should be used to check for a single topic before
        accessing ``single_topic`` to ensure meaningful results.

        Returns:
            True if exactly one topic exists, False otherwise.

        Example:
            >>> outcome = ModelDispatchOutcome(topics=["topic.v1"])
            >>> if outcome.has_single_topic:
            ...     print(outcome.single_topic)
            topic.v1
            >>> ModelDispatchOutcome(topics=[]).has_single_topic
            False
            >>> ModelDispatchOutcome(topics=["a.v1", "b.v1"]).has_single_topic
            False

        .. versionadded:: 0.6.1
            Added as part of union reduction (OMN-1002) to complement
            the ``single_topic`` property's change to return empty string.
        """
        return len(self.topics) == 1

    @property
    def is_empty(self) -> bool:
        """Check if no topics were produced.

        Returns:
            True if topics list is empty, False otherwise.

        Example:
            >>> ModelDispatchOutcome().is_empty
            True
            >>> ModelDispatchOutcome(topics=["topic.v1"]).is_empty
            False

        .. versionadded:: 0.6.0
        """
        return len(self.topics) == 0

    @property
    def has_topics(self) -> bool:
        """Check if any topics were produced.

        Returns:
            True if topics list is non-empty, False otherwise.

        Example:
            >>> ModelDispatchOutcome().has_topics
            False
            >>> ModelDispatchOutcome(topics=["topic.v1"]).has_topics
            True

        .. versionadded:: 0.6.0
        """
        return len(self.topics) > 0

    def to_legacy_output(self) -> str | list[str] | None:
        """Convert back to legacy output format.

        This method enables gradual migration by allowing conversion back
        to the original format where needed.

        Returns:
            - None if no topics
            - str if single topic
            - list[str] if multiple topics

        Example:
            >>> ModelDispatchOutcome().to_legacy_output() is None
            True
            >>> ModelDispatchOutcome(topics=["topic.v1"]).to_legacy_output()
            'topic.v1'
            >>> ModelDispatchOutcome(topics=["a.v1", "b.v1"]).to_legacy_output()
            ['a.v1', 'b.v1']

        .. versionadded:: 0.6.0
        """
        if not self.topics:
            return None
        elif len(self.topics) == 1:
            return self.topics[0]
        else:
            return self.topics
