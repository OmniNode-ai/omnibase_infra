# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20827 AC2: a crash between the two ~/.onex writes is repaired by a rerun of lane-login.

``StoreLaneCredential.save`` writes ``config.yaml`` and then ``credentials.json``.
A crash after the first and before the second leaves either a config naming a
secret the 0600 file does not hold yet (a first login), which ``load`` refuses
naming ``lane-login``, or the previous secret still filed under the lane's fixed
reference (a rotation), which is readable and stale. These tests drive the real
``onex auth lane-login`` command against a real directory, crash the store
between the two writes, and check that a rerun of the command repairs both.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.cli_auth import auth_group
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from omnibase_infra.cli.store_onex_home_files import StoreOnexHomeFiles

pytestmark = pytest.mark.integration

_LANE = "local"
_PASSWORD_REF = StoreLaneCredential.password_ref(_LANE)


class _InjectedCrashError(RuntimeError):
    """Stands in for the process dying between the two writes."""


@pytest.fixture
def onex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``lane-login`` at a real directory instead of the operator's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home / ".onex"


def _lane_login(username: str, password: str) -> None:
    result = CliRunner().invoke(
        auth_group,
        [
            "lane-login",
            "--lane",
            _LANE,
            "--sasl-username",
            username,
            "--sasl-password-stdin",
        ],
        input=f"{password}\n",
    )
    assert result.exit_code == 0, result.output


def _leftovers(directory: Path) -> list[str]:
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.name not in {"config.yaml", "credentials.json"}
    )


def test_a_crash_between_the_two_writes_on_a_first_login_is_refused_then_repaired(
    onex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            StoreOnexHomeFiles,
            "write_secrets",
            lambda self, secrets: (_ for _ in ()).throw(_InjectedCrashError("secrets")),
        )
        with pytest.raises(_InjectedCrashError):
            StoreLaneCredential(onex_home=onex_home).save(
                lane=_LANE,
                sasl_username="local-developer",
                sasl_password="first-secret",
            )

    store = StoreLaneCredential(onex_home=onex_home)
    assert store.declared_lanes() == (_LANE,)
    with pytest.raises(ModelOnexError) as caught:
        store.load(_LANE)
    assert caught.value.error_code == EnumCoreErrorCode.CONFIGURATION_NOT_FOUND
    assert "lane-login" in str(caught.value)

    _lane_login("local-developer", "first-secret")
    credential = StoreLaneCredential(onex_home=onex_home).load(_LANE)
    assert credential.sasl_password.get_secret_value() == "first-secret"
    assert _leftovers(onex_home) == []


def test_a_crash_between_the_two_writes_on_a_rotation_keeps_the_previous_secret(
    onex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lane_login("local-developer", "first-secret")
    secrets_before = (onex_home / "credentials.json").read_bytes()

    with monkeypatch.context() as patch:
        patch.setattr(
            StoreOnexHomeFiles,
            "write_secrets",
            lambda self, secrets: (_ for _ in ()).throw(_InjectedCrashError("secrets")),
        )
        with pytest.raises(_InjectedCrashError):
            StoreLaneCredential(onex_home=onex_home).save(
                lane=_LANE,
                sasl_username="local-developer",
                sasl_password="second-secret",
            )

    # The fixed reference still holds the previous secret: readable, and stale.
    assert (onex_home / "credentials.json").read_bytes() == secrets_before
    stale = StoreLaneCredential(onex_home=onex_home).load(_LANE)
    assert stale.sasl_password_ref == _PASSWORD_REF
    assert stale.sasl_password.get_secret_value() == "first-secret"

    _lane_login("local-developer", "second-secret")
    rotated = StoreLaneCredential(onex_home=onex_home).load(_LANE)
    assert rotated.sasl_password.get_secret_value() == "second-secret"
    assert _leftovers(onex_home) == []
