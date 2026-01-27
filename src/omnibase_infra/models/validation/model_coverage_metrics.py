# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Coverage metrics model for routing coverage validation.

This model replaces the union pattern `dict[str, int | float | list[str]]` that was
used for coverage report returns. By using a structured model with strongly-typed
fields, we eliminate the dict+union pattern while providing better type safety
and IDE support.

.. versionadded:: 0.6.0
    Created as part of Union Reduction Phase 2 (OMN-1002).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class ModelCoverageMetrics(BaseModel):
    """Coverage metrics for routing validation.

    This model normalizes coverage report results into a consistent structure,
    replacing the `dict[str, int | float | list[str]]` pattern with strongly-typed
    fields that provide better type safety and IDE support.

    Attributes:
        total_types: Total number of discovered message types.
        registered_types: Number of types with registered routes.
        unmapped_types: List of type names without routes (sorted alphabetically).
        coverage_percent: Percentage of types with routes (0.0 to 100.0).

    Example:
        >>> # Create metrics from validation results
        >>> metrics = ModelCoverageMetrics.from_counts(
        ...     total=100,
        ...     registered=95,
        ...     unmapped=["TypeA", "TypeB", "TypeC", "TypeD", "TypeE"],
        ... )
        >>> metrics.total_types
        100
        >>> metrics.coverage_percent
        95.0

        >>> # Check if coverage meets threshold
        >>> metrics.coverage_percent >= 90.0
        True

    .. versionadded:: 0.6.0
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        from_attributes=True,
    )

    total_types: int = Field(
        ...,
        ge=0,
        description="Total number of discovered message types.",
    )
    registered_types: int = Field(
        ...,
        ge=0,
        description="Number of types with registered routes.",
    )
    unmapped_types: list[str] = Field(
        default_factory=list,
        description="List of type names without routes (sorted alphabetically).",
    )
    coverage_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of types with routes (0.0 to 100.0).",
    )

    @classmethod
    def from_counts(
        cls,
        total: int,
        registered: int,
        unmapped: Collection[str],
    ) -> ModelCoverageMetrics:
        """Create coverage metrics from raw counts.

        This factory method calculates the coverage percentage and ensures
        the unmapped types list is sorted for consistent output.

        Args:
            total: Total number of discovered message types.
            registered: Number of types with registered routes.
            unmapped: Collection of type names without routes.

        Returns:
            ModelCoverageMetrics with calculated coverage percentage.

        Example:
            >>> metrics = ModelCoverageMetrics.from_counts(
            ...     total=10,
            ...     registered=8,
            ...     unmapped=["ZType", "AType"],
            ... )
            >>> metrics.coverage_percent
            80.0
            >>> metrics.unmapped_types
            ['AType', 'ZType']

        .. versionadded:: 0.6.0
        """
        coverage = (registered / total * 100) if total > 0 else 100.0
        return cls(
            total_types=total,
            registered_types=registered,
            unmapped_types=sorted(unmapped),
            coverage_percent=coverage,
        )

    @classmethod
    def from_discovery(
        cls,
        discovered_types: Mapping[str, object],
        registered_routes: set[str],
    ) -> ModelCoverageMetrics:
        """Create coverage metrics from discovery results.

        This factory method calculates metrics directly from the raw discovery
        data, computing the unmapped types and coverage percentage.

        Args:
            discovered_types: Dictionary of discovered type names to type info.
            registered_routes: Set of type names with registered routes.

        Returns:
            ModelCoverageMetrics with calculated values.

        Example:
            >>> discovered = {"TypeA": ..., "TypeB": ..., "TypeC": ...}
            >>> registered = {"TypeA", "TypeB"}
            >>> metrics = ModelCoverageMetrics.from_discovery(discovered, registered)
            >>> metrics.total_types
            3
            >>> metrics.registered_types
            2
            >>> metrics.unmapped_types
            ['TypeC']

        .. versionadded:: 0.6.0
        """
        discovered_names = set(discovered_types.keys())
        unmapped = discovered_names - registered_routes
        total = len(discovered_types)
        registered = total - len(unmapped)

        return cls.from_counts(
            total=total,
            registered=registered,
            unmapped=unmapped,
        )

    @classmethod
    def from_legacy_dict(cls, report: dict[str, object]) -> ModelCoverageMetrics:
        """Create from legacy dictionary-based coverage report.

        This factory method handles the conversion from the old dict pattern
        to the new model structure. The parameter type uses ``dict[str, object]``
        instead of ``dict[str, int | float | list[str]]`` to reduce union count;
        runtime type assertions ensure the values have the expected types.

        Args:
            report: Legacy coverage report dictionary with keys:
                - "total_types": int
                - "registered_types": int
                - "unmapped_types": list[str]
                - "coverage_percent": float

        Returns:
            ModelCoverageMetrics with equivalent values.

        Raises:
            ProtocolConfigurationError: If any value has an unexpected type.

        Example:
            >>> legacy = {
            ...     "total_types": 100,
            ...     "registered_types": 95,
            ...     "unmapped_types": ["TypeA", "TypeB"],
            ...     "coverage_percent": 95.0,
            ... }
            >>> metrics = ModelCoverageMetrics.from_legacy_dict(legacy)
            >>> metrics.total_types
            100

        .. versionadded:: 0.6.0
        """
        # Type assertions validate the expected structure at runtime
        total_types = report["total_types"]
        registered_types = report["registered_types"]
        unmapped_types = report["unmapped_types"]
        coverage_percent = report["coverage_percent"]

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="from_legacy_dict",
            correlation_id=uuid4(),
        )

        if not isinstance(total_types, int):
            raise ProtocolConfigurationError("total_types must be int", context=context)
        if not isinstance(registered_types, int):
            raise ProtocolConfigurationError(
                "registered_types must be int", context=context
            )
        if not isinstance(unmapped_types, list):
            raise ProtocolConfigurationError(
                "unmapped_types must be list[str]", context=context
            )
        if not isinstance(coverage_percent, int | float):
            raise ProtocolConfigurationError(
                "coverage_percent must be int or float", context=context
            )

        return cls(
            total_types=total_types,
            registered_types=registered_types,
            unmapped_types=unmapped_types,
            coverage_percent=float(coverage_percent),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to legacy dictionary format.

        This method enables gradual migration by allowing conversion back
        to the original format where needed for backwards compatibility.

        The return type uses ``dict[str, object]`` instead of the union
        ``dict[str, int | float | list[str]]`` to reduce union count while
        maintaining structural compatibility. Callers should use the typed
        model fields for type-safe access; this method is primarily for
        serialization and legacy API compatibility.

        Returns:
            Dictionary with keys: total_types (int), registered_types (int),
            unmapped_types (list[str]), coverage_percent (float).

        Example:
            >>> metrics = ModelCoverageMetrics.from_counts(
            ...     total=100, registered=95, unmapped=["TypeA"]
            ... )
            >>> report = metrics.to_dict()
            >>> report["total_types"]
            100
            >>> report["coverage_percent"]
            95.0

        .. versionadded:: 0.6.0
        """
        return {
            "total_types": self.total_types,
            "registered_types": self.registered_types,
            "unmapped_types": list(self.unmapped_types),  # defensive copy
            "coverage_percent": self.coverage_percent,
        }

    @property
    def is_fully_covered(self) -> bool:
        """Check if all types have routes.

        Returns:
            True if coverage is 100%, False otherwise.

        Example:
            >>> metrics = ModelCoverageMetrics.from_counts(
            ...     total=10, registered=10, unmapped=[]
            ... )
            >>> metrics.is_fully_covered
            True

        .. versionadded:: 0.6.0
        """
        return len(self.unmapped_types) == 0

    def meets_threshold(self, threshold_percent: float) -> bool:
        """Check if coverage meets a minimum threshold.

        Args:
            threshold_percent: Minimum required coverage percentage (0.0 to 100.0).

        Returns:
            True if coverage_percent >= threshold_percent.

        Example:
            >>> metrics = ModelCoverageMetrics.from_counts(
            ...     total=100, registered=95, unmapped=["TypeA"] * 5
            ... )
            >>> metrics.meets_threshold(90.0)
            True
            >>> metrics.meets_threshold(99.0)
            False

        .. versionadded:: 0.6.0
        """
        return self.coverage_percent >= threshold_percent


__all__ = ["ModelCoverageMetrics"]
