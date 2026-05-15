# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Helpers for constraining the active runtime package surface.

This module centralizes the allowlist used to narrow runtime activation,
contract discovery, and topic provisioning to the packages an operator
explicitly wants active in a deployment.
"""

from __future__ import annotations

import os

ENV_ACTIVE_RUNTIME_PACKAGES = "ONEX_ACTIVE_RUNTIME_PACKAGES"
_CONDITIONALLY_OWNED_TOPIC_PRODUCERS = frozenset(
    {"omniclaude", "omniintelligence", "omnimemory"}
)


def normalize_runtime_package_name(name: str) -> str:
    """Normalize package names for comparisons.

    Python distributions may appear with either hyphens or underscores
    depending on the source (entry points, metadata, or operator config).
    """
    return name.strip().lower().replace("-", "_")


def get_active_runtime_packages(
    raw_value: str | None = None,
) -> frozenset[str] | None:
    """Return the normalized active runtime package allowlist.

    When the env var is unset or blank, no filtering is applied and ``None``
    is returned for backwards-compatible behavior.
    """
    if raw_value is None:
        raw_value = os.environ.get(ENV_ACTIVE_RUNTIME_PACKAGES)
    if raw_value is None or not raw_value.strip():
        return None

    packages = {
        normalize_runtime_package_name(token)
        for token in raw_value.split(",")
        if token.strip()
    }
    return frozenset(packages) if packages else None


def is_runtime_package_active(
    package_name: str,
    active_packages: frozenset[str] | None = None,
) -> bool:
    """Return True when *package_name* is within the active runtime surface."""
    if active_packages is None:
        active_packages = get_active_runtime_packages()
    if active_packages is None:
        return True
    return normalize_runtime_package_name(package_name) in active_packages


def is_runtime_topic_active(
    topic: str,
    active_packages: frozenset[str] | None = None,
) -> bool:
    """Return True when a topic belongs to an active runtime-owned package domain."""
    parts = topic.split(".")
    if len(parts) != 5:
        return True
    producer = normalize_runtime_package_name(parts[2])
    if producer not in _CONDITIONALLY_OWNED_TOPIC_PRODUCERS:
        return True
    return is_runtime_package_active(producer, active_packages)


__all__ = [
    "ENV_ACTIVE_RUNTIME_PACKAGES",
    "get_active_runtime_packages",
    "is_runtime_package_active",
    "is_runtime_topic_active",
    "normalize_runtime_package_name",
]
