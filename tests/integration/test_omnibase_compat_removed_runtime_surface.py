# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard the runtime identity stamp and topic graph against the retired package.

Imports the real modules and reads the values the runtime uses.
"""

from __future__ import annotations

from omnibase_infra.runtime_identity import STAMPED_PACKAGES
from omnibase_infra.validators.contract_topic_graph import INSTALLED_PACKAGES


def test_stamped_packages_exclude_omnibase_compat() -> None:
    assert "omnibase_compat" not in STAMPED_PACKAGES
    assert {"omnibase_core", "omnibase_infra", "omnibase_spi"} <= set(STAMPED_PACKAGES)


def test_installed_packages_exclude_omnibase_compat() -> None:
    assert "omnibase_compat" not in INSTALLED_PACKAGES
    assert {"omnibase_core", "omnibase_infra", "omnibase_spi"} <= set(
        INSTALLED_PACKAGES
    )
