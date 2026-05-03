# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for runtime security namespace defaults."""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.models.model_security_config import ModelSecurityConfig

pytestmark = pytest.mark.integration


def test_default_security_config_trusts_omnimarket_runtime_namespaces() -> None:
    """The default runtime allowlists accept first-party omnimarket code."""
    config = ModelSecurityConfig()

    assert "omnimarket." in config.get_effective_namespaces()
    assert "omnimarket." in config.get_effective_plugin_namespaces()
