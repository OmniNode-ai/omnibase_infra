# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26616 AC3: a refused local identity is reported in fixed plain words.

The upstream identity reader's own message can carry a file path or parser
text. The doctor must not echo it, and must say how to fix the identity.
"""

import sys
import types
from pathlib import Path

import pytest

from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.doctor.checks.check_delegation_identity import (
    CheckDelegationIdentity,
)

_MODULE = "omnimarket.local_deployment.tenant_identity"
_UPSTREAM_TEXT = "cannot parse /Users/someone/.onex/identity.yaml: line 3 column 9"


class _IdentityError(Exception):
    pass


@pytest.fixture
def refused_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    def reader() -> None:
        raise _IdentityError(_UPSTREAM_TEXT)

    fake = types.ModuleType(_MODULE)
    fake.read_local_tenant_identity = reader
    fake.LocalTenantIdentityError = _IdentityError
    monkeypatch.setitem(sys.modules, _MODULE, fake)


def test_refusal_text_is_not_echoed_and_a_fix_is_named(
    tmp_path: Path, refused_identity: None
) -> None:
    check = CheckDelegationIdentity(onex_home=tmp_path)

    diagnosis = check.diagnose()
    result = check.run()

    for text in (diagnosis.detail, result.message):
        assert "/Users/someone" not in text
        assert "line 3" not in text
        assert "identity.yaml" not in text
    assert "Fix: onex local init" in diagnosis.detail
    assert "Fix: onex local init" in result.message
    assert result.status is not EnumHealthStatusValue.HEALTHY
