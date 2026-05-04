# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration sentinel for the omnibase-spi 0.20.6 runtime protocol floor.

OMN-10169 raises the dependency floor because runtime auto-wiring imports
``omnibase_spi.protocols.runtime``. The fallback compatibility matrix must
carry the same floor for installed-package environments where ``pyproject.toml``
is not present.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from omnibase_infra.runtime.version_compatibility import (
    _FALLBACK_MATRIX,
    VERSION_MATRIX,
    VersionConstraint,
)


def _minimum_for(package: str, matrix: Sequence[VersionConstraint]) -> str:
    for constraint in matrix:
        if constraint.package == package:
            return constraint.min_version
    raise AssertionError(f"{package} missing from compatibility matrix")


@pytest.mark.integration
def test_omnibase_spi_runtime_protocols_match_0206_floor() -> None:
    """The declared and fallback SPI floors both expose runtime protocols."""
    from omnibase_spi.protocols.runtime.protocol_handler_ownership_query import (
        ProtocolHandlerOwnershipQuery,
    )
    from omnibase_spi.protocols.runtime.protocol_handler_resolver import (
        ProtocolHandlerResolver,
    )

    assert _minimum_for("omnibase_spi", VERSION_MATRIX) == "0.20.6"
    assert _minimum_for("omnibase_spi", _FALLBACK_MATRIX) == "0.20.6"
    assert hasattr(ProtocolHandlerResolver, "__class_getitem__")
    assert hasattr(ProtocolHandlerOwnershipQuery, "__class_getitem__")
