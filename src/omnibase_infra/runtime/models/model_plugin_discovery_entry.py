# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Plugin discovery entry model.

This module provides the ModelPluginDiscoveryEntry dataclass and the
PluginDiscoveryStatus Literal type for structured diagnostics of
individual plugin entry-point discovery outcomes.

Design Pattern:
    Each entry records the disposition of a single entry-point discovered
    via ``importlib.metadata``, making "why didn't my plugin load?" a
    10-second debugging problem.

Thread Safety:
    The dataclass uses ``frozen=True`` to prevent attribute reassignment
    after construction.

Example:
    >>> from omnibase_infra.runtime.models import ModelPluginDiscoveryEntry
    >>>
    >>> entry = ModelPluginDiscoveryEntry(
    ...     entry_point_name="my_plugin",
    ...     module_path="myapp.plugins.my_plugin",
    ...     status="accepted",
    ...     plugin_id="my_plugin",
    ... )
    >>> entry.status
    'accepted'

Related:
    - OMN-2012: Create ModelPluginDiscoveryReport + ModelPluginDiscoveryEntry
    - OMN-1346: Registration Code Extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PluginDiscoveryStatus = Literal[
    "accepted",
    "namespace_rejected",
    "import_error",
    "instantiation_error",
    "protocol_invalid",
    "duplicate_skipped",
]


@dataclass(frozen=True)
class ModelPluginDiscoveryEntry:
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
        >>> entry = ModelPluginDiscoveryEntry(
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
    status: PluginDiscoveryStatus
    reason: str = ""
    plugin_id: str | None = None


__all__ = ["ModelPluginDiscoveryEntry", "PluginDiscoveryStatus"]
