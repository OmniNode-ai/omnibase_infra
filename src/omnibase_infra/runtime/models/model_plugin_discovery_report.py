# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Plugin discovery report model.

This module provides the ModelPluginDiscoveryReport dataclass for
structured reporting of plugin discovery outcomes across an entire
entry-point group.

Design Pattern:
    The report aggregates all ``ModelPluginDiscoveryEntry`` results
    produced while scanning one entry-point group, providing quick-access
    properties for filtering rejected entries and detecting errors.

Thread Safety:
    The dataclass uses ``frozen=True`` to prevent attribute reassignment
    after construction. Collection fields use tuples for immutability.

Example:
    >>> from omnibase_infra.runtime.models import (
    ...     ModelPluginDiscoveryReport,
    ...     ModelPluginDiscoveryEntry,
    ... )
    >>>
    >>> entries = (
    ...     ModelPluginDiscoveryEntry(
    ...         entry_point_name="my_plugin",
    ...         module_path="myapp.plugins.my_plugin",
    ...         status="accepted",
    ...         plugin_id="my_plugin",
    ...     ),
    ...     ModelPluginDiscoveryEntry(
    ...         entry_point_name="bad_plugin",
    ...         module_path="myapp.plugins.bad",
    ...         status="import_error",
    ...         reason="ModuleNotFoundError: No module named 'myapp.plugins.bad'",
    ...     ),
    ... )
    >>> report = ModelPluginDiscoveryReport(
    ...     group="omnibase_infra.projectors",
    ...     discovered_count=2,
    ...     accepted=("my_plugin",),
    ...     entries=entries,
    ... )
    >>> report.has_errors
    True
    >>> len(report.rejected)
    1

Related:
    - OMN-2012: Create ModelPluginDiscoveryReport + ModelPluginDiscoveryEntry
    - OMN-1346: Registration Code Extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibase_infra.runtime.models.model_plugin_discovery_entry import (
    ModelPluginDiscoveryEntry,
)


@dataclass(frozen=True)
class ModelPluginDiscoveryReport:
    """Structured report for a single entry-point group discovery pass.

    Aggregates all ``ModelPluginDiscoveryEntry`` results produced while scanning
    one entry-point group (e.g. ``omnibase_infra.projectors``). Provides
    quick-access properties for filtering rejected entries and detecting
    errors.

    Attributes:
        group: Entry-point group name that was scanned
            (e.g. ``"omnibase_infra.projectors"``).
        discovered_count: Total number of entry-points found in the group
            before any filtering.
        accepted: Plugin IDs that were successfully registered, in
            registration order.
        entries: Complete tuple of ``ModelPluginDiscoveryEntry`` results for
            every entry-point in the group.

    Example:
        >>> entries = (
        ...     ModelPluginDiscoveryEntry(
        ...         entry_point_name="good",
        ...         module_path="pkg.good",
        ...         status="accepted",
        ...         plugin_id="good",
        ...     ),
        ...     ModelPluginDiscoveryEntry(
        ...         entry_point_name="bad",
        ...         module_path="pkg.bad",
        ...         status="import_error",
        ...         reason="No module named 'pkg.bad'",
        ...     ),
        ... )
        >>> report = ModelPluginDiscoveryReport(
        ...     group="my.plugins",
        ...     discovered_count=2,
        ...     accepted=("good",),
        ...     entries=entries,
        ... )
        >>> report.has_errors
        True
        >>> [e.entry_point_name for e in report.rejected]
        ['bad']
    """

    group: str
    discovered_count: int
    accepted: tuple[str, ...] = field(default_factory=tuple)
    entries: tuple[ModelPluginDiscoveryEntry, ...] = field(default_factory=tuple)

    @property
    def rejected(self) -> list[ModelPluginDiscoveryEntry]:
        """Return entries whose status is not ``"accepted"``.

        Returns:
            List of non-accepted entries preserving discovery order.

        Example:
            >>> report = ModelPluginDiscoveryReport(
            ...     group="g",
            ...     discovered_count=1,
            ...     entries=(
            ...         ModelPluginDiscoveryEntry(
            ...             entry_point_name="x",
            ...             module_path="m",
            ...             status="namespace_rejected",
            ...             reason="blocked",
            ...         ),
            ...     ),
            ... )
            >>> len(report.rejected)
            1
        """
        return [e for e in self.entries if e.status != "accepted"]

    @property
    def has_errors(self) -> bool:
        """Detect whether any entry suffered an import or instantiation failure.

        Only ``"import_error"`` and ``"instantiation_error"`` are considered
        errors. Policy rejections (``"namespace_rejected"``,
        ``"protocol_invalid"``, ``"duplicate_skipped"``) are not errors --
        they are expected outcomes of the filtering pipeline.

        Returns:
            True if at least one entry has status ``"import_error"`` or
            ``"instantiation_error"``.

        Example:
            >>> report = ModelPluginDiscoveryReport(
            ...     group="g",
            ...     discovered_count=1,
            ...     entries=(
            ...         ModelPluginDiscoveryEntry(
            ...             entry_point_name="x",
            ...             module_path="m",
            ...             status="namespace_rejected",
            ...             reason="blocked",
            ...         ),
            ...     ),
            ... )
            >>> report.has_errors
            False
        """
        error_statuses = {"import_error", "instantiation_error"}
        return any(e.status in error_statuses for e in self.entries)


__all__ = ["ModelPluginDiscoveryReport"]
