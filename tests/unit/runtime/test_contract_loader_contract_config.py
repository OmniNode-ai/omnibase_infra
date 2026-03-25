# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test RuntimeContractConfigLoader respects contract-declared security config."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.errors import ProtocolConfigurationError


@pytest.mark.unit
def test_loader_accepts_valid_path() -> None:
    """Loader must accept paths not in deny list."""
    from omnibase_infra.runtime.runtime_contract_config_loader import (
        RuntimeContractConfigLoader,
    )

    loader = RuntimeContractConfigLoader(
        scan_deny_paths=("..", "/etc", "/var"),
    )
    # Should not raise
    loader.validate_path(Path("src/nodes/my_node/contract.yaml"))


@pytest.mark.unit
def test_loader_validates_path_security() -> None:
    """Loader must reject paths matching deny patterns from contract."""
    from omnibase_infra.runtime.runtime_contract_config_loader import (
        RuntimeContractConfigLoader,
    )

    loader = RuntimeContractConfigLoader(
        scan_deny_paths=("..", "/etc", "/var"),
    )
    with pytest.raises(ProtocolConfigurationError, match="denied"):
        loader.validate_path(Path("/etc/passwd"))


@pytest.mark.unit
def test_loader_path_traversal_denied() -> None:
    """Loader must reject path traversal attempts."""
    from omnibase_infra.runtime.runtime_contract_config_loader import (
        RuntimeContractConfigLoader,
    )

    loader = RuntimeContractConfigLoader(
        scan_deny_paths=("..",),
    )
    with pytest.raises(ProtocolConfigurationError, match="denied"):
        loader.validate_path(Path("../../secrets/keys.yaml"))


@pytest.mark.unit
def test_loader_accepts_exclude_patterns() -> None:
    """Loader must accept exclude patterns from config."""
    from omnibase_infra.runtime.runtime_contract_config_loader import (
        RuntimeContractConfigLoader,
    )

    loader = RuntimeContractConfigLoader(
        scan_exclude_patterns=("**/runtime/**", "**/_*.yaml"),
    )
    assert loader._scan_exclude_patterns == ("**/runtime/**", "**/_*.yaml")


@pytest.mark.unit
def test_loader_backward_compatible() -> None:
    """Loader must work with no args (backward compatible)."""
    from omnibase_infra.runtime.runtime_contract_config_loader import (
        RuntimeContractConfigLoader,
    )

    loader = RuntimeContractConfigLoader()
    assert loader._scan_exclude_patterns == ()
    assert loader._scan_deny_paths == ()
