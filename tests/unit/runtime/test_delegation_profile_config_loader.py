# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for DelegationProfileConfigLoader (OMN-10923).

Requires omnibase-core PR #1072 (jonah/omn-10919-delegation-runtime-profile-contract)
to be merged before the published package includes the delegation models.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURE = (
    Path(__file__).parent.parent.parent
    / "fixtures"
    / "delegation-runtime-profile-test.yaml"
)


@pytest.mark.unit
def test_loads_typed_profile_from_yaml() -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    profile = loader.load()
    assert profile.name == "delegation-runtime-profile"
    assert profile.version == 1
    assert profile.runtime_profile == "main"
    assert profile.event_bus.bootstrap_servers == ["redpanda:9092"]
    assert profile.llm_backends["default"].max_tokens_default == 2048


@pytest.mark.unit
def test_caches_profile_on_repeated_load() -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    profile1 = loader.load()
    profile2 = loader.load()
    assert profile1 is profile2


@pytest.mark.unit
def test_returns_event_bus_config() -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    bus_config = loader.event_bus_config()
    assert bus_config.bootstrap_servers == ["redpanda:9092"]
    assert bus_config.provider == "kafka"
    assert bus_config.topic_policy_ref == "delegation-topic-policy-v1"


@pytest.mark.unit
def test_returns_llm_backend_config() -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    llm_backends = loader.llm_backend_config()
    assert "default" in llm_backends
    default = llm_backends["default"]
    assert default.max_tokens_default == 2048
    assert default.bifrost_endpoint_ref == "local-bifrost"
    assert default.task_model_overrides["reasoning"] == "deepseek-r1"


@pytest.mark.unit
def test_raises_on_missing_file() -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
        DelegationProfileNotFoundError,
    )

    loader = DelegationProfileConfigLoader(contract_path=Path("/nonexistent/path.yaml"))
    with pytest.raises(DelegationProfileNotFoundError, match="not found"):
        loader.load()


@pytest.mark.unit
def test_raises_on_invalid_yaml_schema() -> None:
    """Validation error on YAML that exists but fails model validation."""
    import tempfile

    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
        DelegationProfileNotFoundError,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("name: bad\nversion: -1\nruntime_profile: test\n")
        tmp_path = Path(f.name)

    try:
        loader = DelegationProfileConfigLoader(contract_path=tmp_path)
        with pytest.raises(DelegationProfileNotFoundError, match="Invalid"):
            loader.load()
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.mark.unit
def test_no_delegation_env_var_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loader must not read delegation-specific env vars during load."""
    env_reads: list[str] = []
    original_getenv = os.getenv

    def tracking_getenv(key: str, default: str | None = None) -> str | None:
        env_reads.append(key)
        return original_getenv(key, default)

    monkeypatch.setattr(os, "getenv", tracking_getenv)

    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    loader.load()

    delegation_prefixes = ("KAFKA_", "LLM_", "BIFROST_", "DELEGATION_", "REDPANDA_")
    delegation_reads = [k for k in env_reads if k.startswith(delegation_prefixes)]
    assert delegation_reads == [], (
        f"Loader read delegation env vars: {delegation_reads}"
    )
