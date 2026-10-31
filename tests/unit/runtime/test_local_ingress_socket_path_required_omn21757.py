# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The local ingress socket path is a required setting, not a code default (OMN-21757)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_local_runtime_ingress_config import (
    ModelLocalRuntimeIngressConfig,
)


@pytest.mark.unit
def test_disabled_ingress_needs_no_socket_path() -> None:
    config = ModelLocalRuntimeIngressConfig()
    assert config.enabled is False
    assert config.socket_path == ""


@pytest.mark.unit
def test_enabled_ingress_without_socket_path_is_refused() -> None:
    with pytest.raises(ValidationError, match="socket_path"):
        ModelLocalRuntimeIngressConfig(enabled=True)


@pytest.mark.unit
def test_enabled_ingress_takes_the_supplied_socket_path(tmp_path: object) -> None:
    config = ModelLocalRuntimeIngressConfig(
        enabled=True, socket_path=f"  {tmp_path}/ingress.sock "
    )
    assert config.socket_path == f"{tmp_path}/ingress.sock"
