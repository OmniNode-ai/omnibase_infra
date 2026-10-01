# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for exemption validation against the default source tree."""

import pytest

from omnibase_infra.validation.infra_validators import validate_infra_unused_exemptions


@pytest.mark.integration
def test_unused_exemptions_against_default_source_tree() -> None:
    """Verify every configured exemption matches a real source-tree violation."""
    result = validate_infra_unused_exemptions()

    assert result.is_valid, result.errors
    assert not result.errors
