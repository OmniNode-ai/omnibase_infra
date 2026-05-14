# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for delegation runtime profile config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.delegation_profile_config_loader import (
    DelegationProfileConfigLoader,
)


@pytest.mark.integration
def test_delegation_profile_loader_projects_contract_runtime_config() -> None:
    pytest.importorskip(
        "omnibase_compat.contracts.delegation.model_delegation_runtime_profile",
        reason="delegation contracts require omnibase_compat PR #87",
    )
    fixture = (
        Path(__file__).parents[2] / "fixtures" / "delegation-runtime-profile-test.yaml"
    )

    loader = DelegationProfileConfigLoader(contract_path=fixture)
    profile = loader.load()

    assert profile.runtime_profile == "main"
    assert loader.event_bus_config().bootstrap_servers == ["redpanda:9092"]
    assert (
        loader.llm_backend_config()["default"].bifrost_endpoint_ref == "local-bifrost"
    )
