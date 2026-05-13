# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Prove delegation runtime boots from contract config with zero delegation env vars."""

import os
from pathlib import Path

import pytest

FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "delegation-runtime-profile-test.yaml"
)

DELEGATION_ENV_VARS = [
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_BROKER_ALLOWLIST",
    "LLM_CODER_URL",
    "LLM_CODER_FAST_URL",
    "LLM_CODER_MODEL_NAME",
    "LLM_CODER_FAST_MODEL_NAME",
    "BIFROST_CONTRACT_PATH",
    "TASK_CLASS_CONTRACT_PATH",
    "LLM_ENDPOINT_CIDR_ALLOWLIST",
    "LOCAL_LLM_SHARED_SECRET",
]


@pytest.fixture
def clean_delegation_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in DELEGATION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.integration
def test_fixture_exists() -> None:
    assert FIXTURE.exists(), f"Test fixture not found: {FIXTURE}"


@pytest.mark.integration
def test_contract_loads_with_no_delegation_env_vars(clean_delegation_env: None) -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    profile = loader.load()
    assert profile.name == "delegation-runtime-profile"
    assert profile.version == 1


@pytest.mark.integration
def test_event_bus_config_accessor(clean_delegation_env: None) -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    bus = loader.event_bus_config()
    assert bus.provider == "kafka"
    assert len(bus.bootstrap_servers) > 0


@pytest.mark.integration
def test_llm_backend_config_accessor(clean_delegation_env: None) -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    llm = loader.llm_backend_config()
    assert llm is not None


@pytest.mark.integration
def test_security_config_accessor(clean_delegation_env: None) -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    security = loader.security_config()
    assert security is not None


@pytest.mark.integration
def test_no_delegation_env_reads_during_load(
    clean_delegation_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critical: prove the loader does not read any delegation env vars."""
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

    delegation_reads = [k for k in env_reads if k in DELEGATION_ENV_VARS]
    assert delegation_reads == [], (
        f"Loader read delegation env vars: {delegation_reads}"
    )


@pytest.mark.integration
def test_missing_file_raises_not_found_error() -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
        DelegationProfileNotFoundError,
    )

    loader = DelegationProfileConfigLoader(contract_path=Path("/nonexistent/path.yaml"))
    with pytest.raises(DelegationProfileNotFoundError):
        loader.load()


@pytest.mark.integration
def test_loader_caches_after_first_load(clean_delegation_env: None) -> None:
    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    profile1 = loader.load()
    profile2 = loader.load()
    assert profile1 is profile2
