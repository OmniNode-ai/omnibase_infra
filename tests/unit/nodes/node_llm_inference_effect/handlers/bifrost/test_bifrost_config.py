# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for bifrost configuration model validation.

Tests cover field constraints on ModelBifrostBackendConfig, including
the max_length constraint on hmac_secret (OMN-1467).

Related:
    - OMN-1467: Add max_length constraint to webhook secret field
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_config import (
    ModelBifrostBackendConfig,
)


class TestModelBifrostBackendConfigHmacSecret:
    """Tests for hmac_secret field constraints on ModelBifrostBackendConfig."""

    def test_hmac_secret_none_is_valid(self) -> None:
        """hmac_secret defaults to None (HMAC disabled)."""
        config = ModelBifrostBackendConfig(
            backend_id="test-backend",
            base_url="http://localhost:8000",
        )
        assert config.hmac_secret is None

    def test_hmac_secret_within_max_length_is_valid(self) -> None:
        """hmac_secret within 256 characters is accepted."""
        secret = "a" * 256
        config = ModelBifrostBackendConfig(
            backend_id="test-backend",
            base_url="http://localhost:8000",
            hmac_secret=secret,
        )
        assert config.hmac_secret == secret

    def test_hmac_secret_exceeding_max_length_is_rejected(self) -> None:
        """hmac_secret exceeding 256 characters raises ValidationError."""
        secret = "a" * 257
        with pytest.raises(ValidationError, match="hmac_secret"):
            ModelBifrostBackendConfig(
                backend_id="test-backend",
                base_url="http://localhost:8000",
                hmac_secret=secret,
            )

    def test_hmac_secret_typical_length_is_valid(self) -> None:
        """A typical 64-character hex secret is accepted."""
        secret = "abcdef0123456789" * 4  # 64 chars
        config = ModelBifrostBackendConfig(
            backend_id="test-backend",
            base_url="http://localhost:8000",
            hmac_secret=secret,
        )
        assert config.hmac_secret == secret

    def test_hmac_secret_not_in_repr(self) -> None:
        """hmac_secret is excluded from repr for security."""
        config = ModelBifrostBackendConfig(
            backend_id="test-backend",
            base_url="http://localhost:8000",
            hmac_secret="super-secret-key",
        )
        assert "super-secret-key" not in repr(config)
