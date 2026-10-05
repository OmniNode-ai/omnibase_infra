# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline branch coverage for the delegation identity doctor check."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.doctor.checks import check_delegation_identity as module
from omnibase_infra.doctor.checks.check_delegation_identity import (
    CheckDelegationIdentity,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import EnumDelegationDoctorFault


@pytest.mark.parametrize(
    "config",
    [
        None,
        "{}",
        "gateway: null",
        "gateway: {}",
        "gateway:\n  tenant_slug: acme",
        "gateway:\n  base_url: https://gateway.invalid",
        "gateway:\n  tenant_slug: ' '\n  base_url: https://gateway.invalid",
        "gateway:\n  tenant_slug: acme\n  base_url: 42",
    ],
)
def test_missing_identity_has_login_remediation(
    tmp_path: Path, config: str | None
) -> None:
    if config is not None:
        (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    check = CheckDelegationIdentity(onex_home=tmp_path)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is EnumDelegationDoctorFault.NO_IDENTITY
    assert diagnosis.fix == (
        "pbpaste | onex auth login --tenant-slug <slug> --base-url <origin> --api-key-stdin"
    )
    assert result.status is EnumHealthStatusValue.UNHEALTHY
    assert "[no_identity]" in result.message
    assert diagnosis.fix in result.message


@pytest.mark.parametrize("config", ["[]", "gateway: []", "gateway: [", ""])
def test_invalid_config_is_unknown(tmp_path: Path, config: str) -> None:
    (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    check = CheckDelegationIdentity(onex_home=tmp_path)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is None
    assert diagnosis.fix == ""
    assert "config.yaml is unreadable or invalid" in diagnosis.detail
    assert result.status is EnumHealthStatusValue.UNKNOWN
    assert result.message == diagnosis.detail


def test_unreadable_config_is_unknown(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").mkdir()
    check = CheckDelegationIdentity(onex_home=tmp_path)

    assert check.diagnose().fault is None
    assert check.run().status is EnumHealthStatusValue.UNKNOWN


def test_identity_is_trimmed_and_needs_no_key(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  tenant_slug: ' acme '\n  base_url: ' https://gateway.invalid '\n",
        encoding="utf-8",
    )
    check = CheckDelegationIdentity(onex_home=tmp_path)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is None
    assert diagnosis.fix == ""
    assert diagnosis.detail == (
        "Delegation identity 'acme' is configured for https://gateway.invalid."
    )
    assert result.status is EnumHealthStatusValue.HEALTHY
    assert result.message == diagnosis.detail


def test_default_home_uses_injected_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    (tmp_path / ".onex").mkdir()
    (tmp_path / ".onex" / "config.yaml").write_text(
        "gateway:\n  tenant_slug: acme\n  base_url: https://gateway.invalid\n",
        encoding="utf-8",
    )

    assert CheckDelegationIdentity().run().status is EnumHealthStatusValue.HEALTHY


def test_unexpected_reader_failure_is_total_and_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sensitive = "identity-reader-private-detail"
    monkeypatch.setattr(
        module, "read_gateway_block", Mock(side_effect=RuntimeError(sensitive))
    )
    check = CheckDelegationIdentity(onex_home=tmp_path)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is None
    assert diagnosis.fix == ""
    assert "check failed safely" in diagnosis.detail
    assert result.status is EnumHealthStatusValue.UNKNOWN
    assert "check failed safely" in result.message
    assert sensitive not in diagnosis.detail + result.message
