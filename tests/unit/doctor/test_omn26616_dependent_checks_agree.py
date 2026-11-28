# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26616 AC2: every delegation doctor line agrees on whether an identity exists.

The gateway, API key and quota checks used to decide "is an identity configured"
from the config.yaml gateway block while the identity check asked the local
tenant identity. After ``onex local init`` that put "Delegation identity is
configured." beside "was not judged because delegation_identity is not
configured". These tests drive the real seam, the omnimarket identity reader,
and assert on the rendered doctor lines.
"""

import sys
import types
import uuid
from pathlib import Path

import pytest

from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.doctor.checks.check_delegation_gateway import (
    CheckDelegationGateway,
)
from omnibase_infra.doctor.checks.check_delegation_identity import (
    CheckDelegationIdentity,
)
from omnibase_infra.doctor.checks.check_delegation_key import CheckDelegationKey
from omnibase_infra.doctor.checks.check_delegation_quota import CheckDelegationQuota
from omnibase_infra.doctor.enum_delegation_doctor_fault import EnumDelegationDoctorFault

_MODULE = "omnimarket.local_deployment.tenant_identity"
_DEPENDENTS = [CheckDelegationGateway, CheckDelegationKey, CheckDelegationQuota]
_GATEWAY_BLOCK = "gateway:\n  tenant_slug: acme\n  base_url: https://gateway.invalid\n"


class _IdentityError(Exception):
    pass


def _install(monkeypatch: pytest.MonkeyPatch, reader: object) -> None:
    fake = types.ModuleType(_MODULE)
    fake.read_local_tenant_identity = reader
    fake.LocalTenantIdentityError = _IdentityError
    monkeypatch.setitem(sys.modules, _MODULE, fake)


def _with_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant = uuid.uuid4()
    _install(monkeypatch, lambda: types.SimpleNamespace(tenant_uuid=tenant))


@pytest.mark.parametrize("check_class", _DEPENDENTS)
def test_dependents_after_local_init_do_not_contradict_the_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_class: type
) -> None:
    _with_identity(monkeypatch)

    identity = CheckDelegationIdentity(onex_home=tmp_path).run()
    dependent = check_class(onex_home=tmp_path).run()

    assert identity.status is EnumHealthStatusValue.HEALTHY
    assert "not configured" not in dependent.message
    assert "no delegation gateway address is saved" in dependent.message
    assert dependent.status is EnumHealthStatusValue.UNKNOWN


@pytest.mark.parametrize("check_class", _DEPENDENTS)
def test_no_doctor_line_shows_the_internal_identity_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_class: type
) -> None:
    for install in (
        lambda: _install(monkeypatch, lambda: None),
        lambda: _with_identity(monkeypatch),
    ):
        install()
        assert (
            "delegation_identity" not in check_class(onex_home=tmp_path).run().message
        )
        assert (
            "delegation_identity"
            not in check_class(onex_home=tmp_path).diagnose().detail
        )
    (tmp_path / "config.yaml").write_text("gateway:\n  tenant_slug: acme\n")
    assert "delegation_identity" not in check_class(onex_home=tmp_path).run().message


@pytest.mark.parametrize("check_class", _DEPENDENTS)
def test_dependents_without_any_identity_point_at_local_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_class: type
) -> None:
    _install(monkeypatch, lambda: None)

    result = check_class(onex_home=tmp_path).run()

    assert "onex local init" in result.message
    assert result.status is EnumHealthStatusValue.UNKNOWN


def test_gateway_block_without_local_identity_is_not_reported_as_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, lambda: None)
    (tmp_path / "config.yaml").write_text(_GATEWAY_BLOCK, encoding="utf-8")
    check = CheckDelegationIdentity(onex_home=tmp_path)

    diagnosis = check.diagnose()

    assert diagnosis.fault is EnumDelegationDoctorFault.NO_IDENTITY
    assert diagnosis.fix == "onex local init"
    assert check.run().status is EnumHealthStatusValue.UNHEALTHY


def test_control_gateway_block_with_local_identity_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_identity(monkeypatch)
    (tmp_path / "config.yaml").write_text(_GATEWAY_BLOCK, encoding="utf-8")

    result = CheckDelegationIdentity(onex_home=tmp_path).run()

    assert result.status is EnumHealthStatusValue.HEALTHY
    assert result.message == "Delegation identity is configured."


@pytest.mark.parametrize("check_class", _DEPENDENTS)
def test_unreadable_recorded_identity_is_reported_not_called_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check_class: type
) -> None:
    def reader() -> None:
        raise _IdentityError("corrupt")

    _install(monkeypatch, reader)

    message = check_class(onex_home=tmp_path).run().message

    assert "could not be read" in message
