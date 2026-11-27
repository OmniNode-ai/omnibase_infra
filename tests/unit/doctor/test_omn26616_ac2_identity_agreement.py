# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26616 AC2: the doctor identity check agrees with `onex local identity`.

Both read the identity through ``omnimarket.local_deployment.tenant_identity``.
The tests stand in for that module so the whole seam is exercised.
"""

import sys
import types
import uuid
from pathlib import Path

import pytest

from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.doctor.checks.check_delegation_identity import (
    CheckDelegationIdentity,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import EnumDelegationDoctorFault

_MODULE = "omnimarket.local_deployment.tenant_identity"


class _IdentityError(Exception):
    pass


def _install(monkeypatch: pytest.MonkeyPatch, reader: object) -> None:
    fake = types.ModuleType(_MODULE)
    fake.read_local_tenant_identity = reader
    fake.LocalTenantIdentityError = _IdentityError
    monkeypatch.setitem(sys.modules, _MODULE, fake)


def test_local_identity_present_is_not_no_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tenant = uuid.uuid4()
    _install(monkeypatch, lambda: types.SimpleNamespace(tenant_uuid=tenant))
    check = CheckDelegationIdentity(onex_home=tmp_path)

    diagnosis = check.diagnose()

    assert diagnosis.fault is None
    assert check.run().status is EnumHealthStatusValue.HEALTHY


def test_no_local_identity_says_how_to_create_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, lambda: None)

    diagnosis = CheckDelegationIdentity(onex_home=tmp_path).diagnose()

    assert diagnosis.fault is EnumDelegationDoctorFault.NO_IDENTITY
    assert diagnosis.fix == "onex local init"


def test_unreadable_recorded_identity_is_a_refusal_not_no_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reader() -> None:
        raise _IdentityError("the recorded tenant identity is corrupt")

    _install(monkeypatch, reader)
    check = CheckDelegationIdentity(onex_home=tmp_path)

    diagnosis = check.diagnose()

    assert diagnosis.fault is not EnumDelegationDoctorFault.NO_IDENTITY
    assert "could not be read" in diagnosis.detail
    assert check.run().status is not EnumHealthStatusValue.HEALTHY


def test_omnimarket_not_importable_is_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, _MODULE, None)  # import raises ImportError

    diagnosis = CheckDelegationIdentity(onex_home=tmp_path).diagnose()

    assert diagnosis.fault is EnumDelegationDoctorFault.NO_IDENTITY
