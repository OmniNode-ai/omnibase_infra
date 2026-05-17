# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for topic suffix export completeness [OMN-8605]."""

from __future__ import annotations

import pytest

import omnibase_infra.topics as topics_init
from omnibase_infra.topics import platform_topic_suffixes


def _get_suffix_constants_from_module(module: object) -> set[str]:
    return {
        name
        for name in dir(module)
        if name.startswith("SUFFIX_") and isinstance(getattr(module, name), str)
    }


@pytest.mark.integration
def test_all_suffix_constants_exported_in_init() -> None:
    """Every SUFFIX_* constant in platform_topic_suffixes must appear in topics __all__."""
    defined = _get_suffix_constants_from_module(platform_topic_suffixes)
    exported = set(topics_init.__all__)
    missing = defined - exported
    assert not missing, (
        f"{len(missing)} SUFFIX_* constant(s) defined but not in topics/__init__.py __all__: "
        + ", ".join(sorted(missing))
    )


@pytest.mark.integration
def test_all_suffix_constants_importable_from_topics() -> None:
    """Every SUFFIX_* constant in __all__ must be importable from omnibase_infra.topics."""
    import omnibase_infra.topics as topics_pkg

    for name in topics_pkg.__all__:
        if name.startswith("SUFFIX_"):
            assert hasattr(topics_pkg, name), (
                f"{name} listed in __all__ but not importable"
            )


@pytest.mark.integration
def test_registration_snapshots_topic_follows_onex_grammar() -> None:
    """SUFFIX_REGISTRATION_SNAPSHOTS must use onex.evt.* grammar, not onex.snapshot.* (OMN-9211)."""
    from omnibase_infra.topics.platform_topic_suffixes import (
        SUFFIX_REGISTRATION_SNAPSHOTS,
    )

    assert SUFFIX_REGISTRATION_SNAPSHOTS.startswith("onex.evt."), (
        f"Registration snapshots topic must use onex.evt.* kind, got: {SUFFIX_REGISTRATION_SNAPSHOTS}"
    )
    parts = SUFFIX_REGISTRATION_SNAPSHOTS.split(".")
    assert len(parts) == 5, (
        f"Topic must have 5 segments (onex.<kind>.<producer>.<event>.v<N>), got: {SUFFIX_REGISTRATION_SNAPSHOTS}"
    )
    assert (
        SUFFIX_REGISTRATION_SNAPSHOTS == "onex.evt.platform.registration-snapshots.v1"
    ), f"Unexpected topic value: {SUFFIX_REGISTRATION_SNAPSHOTS}"
