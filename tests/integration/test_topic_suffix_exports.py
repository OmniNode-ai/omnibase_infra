# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for topic suffix export completeness [OMN-8605]."""

from __future__ import annotations

import pytest

from omnibase_infra.topics import __init__ as topics_init
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
