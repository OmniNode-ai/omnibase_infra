# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Plugin discovery report model.

This module provides the ModelPluginDiscoveryReport and PluginDiscoveryEntry
dataclasses for structured reporting of plugin discovery outcomes.

Design Pattern:
    Discovery report models provide detailed per-entry diagnostics for
    plugin loading, making "why didn't my plugin load?" a 10-second
    debugging problem. Each entry records the disposition of a single
    entry-point discovered via ``importlib.metadata``.

Thread Safety:
    Both dataclasses are immutable by convention (no mutating methods).
    Lists are populated at construction time and should not be modified
    after creation.

Example:
    >>> from omnibase_infra.runtime.models import (
    ...     ModelPluginDiscoveryReport,
    ...     PluginDiscoveryEntry,
    ... )
    >>>
    >>> entries = [
    ...     PluginDiscoveryEntry(
    ...         entry_point_name="my_plugin",
    ...         module_path="myapp.plugins.my_plugin",
    ...         status="accepted",
    ...         plugin_id="my_plugin",
    ...     ),
    ...     PluginDiscoveryEntry(
    ...         entry_point_name="bad_plugin",
    ...         module_path="myapp.plugins.bad",
    ...         status="import_error",
    ...         reason="ModuleNotFoundError: No module named 'myapp.plugins.bad'",
    ...     ),
    ... ]
    >>> report = ModelPluginDiscoveryReport(
    ...     group="omnibase_infra.projectors",
    ...     discovered_count=2,
    ...     accepted=["my_plugin"],
    ...     entries=entries,
    ... )
    >>> report.has_errors
    True
    >>> len(report.rejected)
    1

Related:
    - OMN-2012: Create ModelPluginDiscoveryReport + PluginDiscoveryEntry
    - OMN-1346: Registration Code Extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PluginDiscoveryEntry:
    """A single entry-point discovery result.

    Records the outcome of attempting to load one plugin entry-point,
    including its final disposition and any diagnostic information.

    Attributes:
        entry_point_name: Name of the entry-point as declared in
            ``pyproject.toml`` or ``setup.cfg``.
        module_path: Dotted module path the entry-point resolves to.
        status: Disposition of this entry-point. One of:
            ``"accepted"`` -- successfully loaded and registered.
            ``"namespace_rejected"`` -- blocked by namespace allowlist.
            ``"import_error"`` -- ``importlib`` could not load the module.
            ``"instantiation_error"`` -- class loaded but constructor failed.
            ``"protocol_invalid"`` -- class does not satisfy required protocol.
            ``"duplicate_skipped"`` -- a plugin with the same ID was already
            registered.
        reason: Human-readable explanation. Empty string for accepted entries.
        plugin_id: Plugin identifier. Set only for ``"accepted"`` and
            ``"duplicate_skipped"`` entries; ``None`` otherwise.

    Example:
        >>> entry = PluginDiscoveryEntry(
        ...     entry_point_name="my_plugin",
        ...     module_path="myapp.plugins.my_plugin",
        ...     status="accepted",
        ...     plugin_id="my_plugin",
        ... )
        >>> entry.status
        'accepted'
        >>> entry.reason
        ''
    """

    entry_point_name: str
    module_path: str
    status: str
    reason: str = ""
    plugin_id: str | None = None


@dataclass
class ModelPluginDiscoveryReport:
    """Structured report for a single entry-point group discovery pass.

    Aggregates all ``PluginDiscoveryEntry`` results produced while scanning
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
        entries: Complete list of ``PluginDiscoveryEntry`` results for
            every entry-point in the group.

    Example:
        >>> entries = [
        ...     PluginDiscoveryEntry(
        ...         entry_point_name="good",
        ...         module_path="pkg.good",
        ...         status="accepted",
        ...         plugin_id="good",
        ...     ),
        ...     PluginDiscoveryEntry(
        ...         entry_point_name="bad",
        ...         module_path="pkg.bad",
        ...         status="import_error",
        ...         reason="No module named 'pkg.bad'",
        ...     ),
        ... ]
        >>> report = ModelPluginDiscoveryReport(
        ...     group="my.plugins",
        ...     discovered_count=2,
        ...     accepted=["good"],
        ...     entries=entries,
        ... )
        >>> report.has_errors
        True
        >>> [e.entry_point_name for e in report.rejected]
        ['bad']
    """

    group: str
    discovered_count: int
    accepted: list[str] = field(default_factory=list)
    entries: list[PluginDiscoveryEntry] = field(default_factory=list)

    @property
    def rejected(self) -> list[PluginDiscoveryEntry]:
        """Return entries whose status is not ``"accepted"``.

        Returns:
            List of non-accepted entries preserving discovery order.

        Example:
            >>> report = ModelPluginDiscoveryReport(
            ...     group="g",
            ...     discovered_count=1,
            ...     entries=[
            ...         PluginDiscoveryEntry(
            ...             entry_point_name="x",
            ...             module_path="m",
            ...             status="namespace_rejected",
            ...             reason="blocked",
            ...         ),
            ...     ],
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
            ...     entries=[
            ...         PluginDiscoveryEntry(
            ...             entry_point_name="x",
            ...             module_path="m",
            ...             status="namespace_rejected",
            ...             reason="blocked",
            ...         ),
            ...     ],
            ... )
            >>> report.has_errors
            False
        """
        error_statuses = {"import_error", "instantiation_error"}
        return any(e.status in error_statuses for e in self.entries)


__all__ = ["ModelPluginDiscoveryReport", "PluginDiscoveryEntry"]
