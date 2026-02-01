# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for contract publisher configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.services.contract_publisher import ModelContractPublisherConfig


class TestModelContractPublisherConfig:
    """Tests for ModelContractPublisherConfig validation and behavior."""

    def test_filesystem_mode_valid(self) -> None:
        """Test filesystem mode with valid config."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/app/contracts"),
        )
        assert config.mode == "filesystem"
        assert config.filesystem_root == Path("/app/contracts")

    def test_filesystem_mode_missing_root_raises(self) -> None:
        """Test filesystem mode without root raises ValueError."""
        with pytest.raises(
            ValidationError, match="filesystem mode requires filesystem_root"
        ):
            ModelContractPublisherConfig(mode="filesystem")

    def test_package_mode_valid(self) -> None:
        """Test package mode with valid config."""
        config = ModelContractPublisherConfig(
            mode="package",
            package_module="myapp.contracts",
        )
        assert config.mode == "package"
        assert config.package_module == "myapp.contracts"

    def test_package_mode_missing_module_raises(self) -> None:
        """Test package mode without module raises ValueError."""
        with pytest.raises(
            ValidationError, match="package mode requires package_module"
        ):
            ModelContractPublisherConfig(mode="package")

    def test_composite_mode_with_filesystem(self) -> None:
        """Test composite mode with only filesystem."""
        config = ModelContractPublisherConfig(
            mode="composite",
            filesystem_root=Path("/app/contracts"),
        )
        assert config.mode == "composite"
        assert config.filesystem_root is not None

    def test_composite_mode_with_package(self) -> None:
        """Test composite mode with only package."""
        config = ModelContractPublisherConfig(
            mode="composite",
            package_module="myapp.contracts",
        )
        assert config.mode == "composite"
        assert config.package_module is not None

    def test_composite_mode_with_both(self) -> None:
        """Test composite mode with both sources."""
        config = ModelContractPublisherConfig(
            mode="composite",
            filesystem_root=Path("/app/contracts"),
            package_module="myapp.contracts",
        )
        assert config.mode == "composite"
        assert config.filesystem_root is not None
        assert config.package_module is not None

    def test_composite_mode_no_sources_raises(self) -> None:
        """Test composite mode without any source raises ValueError."""
        with pytest.raises(
            ValidationError, match="composite mode requires at least one source"
        ):
            ModelContractPublisherConfig(mode="composite")

    def test_defaults(self) -> None:
        """Test default values."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/app"),
        )
        assert config.fail_fast is True
        assert config.allow_zero_contracts is False
        assert config.environment is None

    def test_resolve_environment_from_config(self) -> None:
        """Test environment resolution from config."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/app"),
            environment="prod",
        )
        assert config.resolve_environment() == "prod"

    def test_resolve_environment_from_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment resolution from env var."""
        monkeypatch.setenv("ONEX_ENV", "staging")
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/app"),
        )
        assert config.resolve_environment() == "staging"

    def test_resolve_environment_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment resolution default."""
        monkeypatch.delenv("ONEX_ENV", raising=False)
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/app"),
        )
        assert config.resolve_environment() == "dev"

    def test_resolve_environment_config_wins_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test config environment takes precedence over env var."""
        monkeypatch.setenv("ONEX_ENV", "staging")
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/app"),
            environment="prod",
        )
        assert config.resolve_environment() == "prod"

    def test_resolve_environment_strips_whitespace(self) -> None:
        """Test environment normalization strips whitespace."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/app"),
            environment="  prod  ",
        )
        assert config.resolve_environment() == "prod"

    def test_resolve_environment_strips_trailing_dot(self) -> None:
        """Test environment normalization strips trailing dot."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/app"),
            environment="prod.",
        )
        assert config.resolve_environment() == "prod"

    def test_config_is_frozen(self) -> None:
        """Test config model is frozen."""
        config = ModelContractPublisherConfig(
            mode="filesystem",
            filesystem_root=Path("/app"),
        )
        with pytest.raises(ValidationError):
            config.mode = "package"  # type: ignore[misc]
