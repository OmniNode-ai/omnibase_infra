# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Metrics policy model for cardinality enforcement.

This is a local implementation that will be replaced by the omnibase_core
version once it's released. See OMN-1367 for the upstream implementation.

The policy controls which labels are allowed/forbidden and how violations
are handled during metrics emission.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass


class EnumMetricsPolicyViolationAction(str, Enum):
    """Action to take when a metrics policy violation is detected.

    Attributes:
        RAISE: Raise an exception on policy violation.
        WARN_AND_DROP: Log a warning and drop the metric entirely.
        DROP_SILENT: Silently drop the metric without logging.
        WARN_AND_STRIP: Log a warning and strip the offending labels.
    """

    RAISE = "raise"
    WARN_AND_DROP = "warn_and_drop"
    DROP_SILENT = "drop_silent"
    WARN_AND_STRIP = "warn_and_strip"


class ModelMetricsPolicy(BaseModel):
    """Policy configuration for metrics cardinality enforcement.

    This model defines the rules for validating metric labels to prevent
    high-cardinality label explosions that can overwhelm metrics backends.

    Attributes:
        forbidden_label_keys: Set of label keys that are never allowed.
            Defaults to high-cardinality identifiers like envelope_id,
            correlation_id, node_id, runtime_id.
        max_label_value_length: Maximum allowed length for label values.
            Values exceeding this are truncated or rejected based on
            on_violation setting. Default: 128.
        on_violation: Action to take when policy is violated.
            Default: WARN_AND_DROP.

    Example:
        >>> policy = ModelMetricsPolicy(
        ...     on_violation=EnumMetricsPolicyViolationAction.RAISE,
        ...     max_label_value_length=64,
        ... )
        >>> result = policy.enforce_labels({"user_id": "12345"})
        >>> print(result)
        {'user_id': '12345'}
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forbidden_label_keys: frozenset[str] = Field(
        default=frozenset({"envelope_id", "correlation_id", "node_id", "runtime_id"}),
        description="Set of label keys that are forbidden due to high cardinality.",
    )
    max_label_value_length: int = Field(
        default=128,
        ge=1,
        description="Maximum allowed length for label values.",
    )
    on_violation: EnumMetricsPolicyViolationAction = Field(
        default=EnumMetricsPolicyViolationAction.WARN_AND_DROP,
        description="Action to take when policy is violated.",
    )

    def enforce_labels(self, labels: dict[str, str]) -> dict[str, str] | None:
        """Enforce policy on the given labels.

        Checks labels against forbidden keys and value length limits.
        Returns the validated/modified labels or None if the metric should
        be dropped.

        Args:
            labels: Label key-value pairs to validate.

        Returns:
            - The (possibly modified) labels if allowed
            - None if metric should be dropped (WARN_AND_DROP or DROP_SILENT)

        Raises:
            OnexError: If on_violation is RAISE and violations are detected.
        """
        import logging

        from omnibase_core.errors import OnexError

        _logger = logging.getLogger(__name__)

        violations: list[str] = []
        result_labels = dict(labels)

        # Check for forbidden keys
        for key in labels:
            if key in self.forbidden_label_keys:
                violations.append(
                    f"Label key '{key}' is forbidden (high-cardinality identifier)"
                )
                if self.on_violation == EnumMetricsPolicyViolationAction.WARN_AND_STRIP:
                    del result_labels[key]

        # Check value lengths
        for key, value in list(result_labels.items()):
            if len(value) > self.max_label_value_length:
                violations.append(
                    f"Label value too long: {key} "
                    f"({len(value)} > {self.max_label_value_length})"
                )
                if self.on_violation == EnumMetricsPolicyViolationAction.WARN_AND_STRIP:
                    result_labels[key] = value[: self.max_label_value_length]

        # No violations - return original labels
        if not violations:
            return result_labels

        # Handle violations based on policy
        violation_msg = "; ".join(violations)

        if self.on_violation == EnumMetricsPolicyViolationAction.RAISE:
            raise OnexError(f"Metrics policy violation(s): {violation_msg}")

        if self.on_violation == EnumMetricsPolicyViolationAction.WARN_AND_DROP:
            _logger.warning(
                "Dropping metric due to policy violation(s): %s", violation_msg
            )
            return None

        if self.on_violation == EnumMetricsPolicyViolationAction.DROP_SILENT:
            return None

        if self.on_violation == EnumMetricsPolicyViolationAction.WARN_AND_STRIP:
            _logger.warning(
                "Stripping invalid labels due to policy violation(s): %s", violation_msg
            )
            return result_labels

        return result_labels


__all__: list[str] = [
    "EnumMetricsPolicyViolationAction",
    "ModelMetricsPolicy",
]
