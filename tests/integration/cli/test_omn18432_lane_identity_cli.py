# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for the lane bus identity (OMN-18432).

The unit modules drive the store, the transport model and the resolution order
directly. This one goes through ``click`` twice -- the real ``onex auth
lane-login`` writing real files under a real ``~/.onex`` root, then the real
``onex delegate`` reading them back -- because the two halves are only connected
by those two files and a unit test that constructs the store itself would never
notice the command writing a different shape than the reader expects.

That is not a hypothetical failure: it is exactly OMN-18422, where two live
commands wrote the SAME credential kind into the SAME config file under two
different block names and neither observed the other, so a machine holding a
key reported that it held none.

Dispatch itself needs a co-installed omnimarket and a live broker, neither of
which belongs in this gate. Everything under test happens in front of
dispatch: the login write, the declaration read, the credential resolution and
the refusal -- and, for the success path, exactly what the dispatch would have
been handed.

Every test exports an ambient SASL environment that names a DIFFERENT
principal, so a silent fall-through to the environment on a machine that holds
its own identity is visible as a test resolving the wrong name rather than as
a test that quietly still passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_auth import auth_group
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.delegate_lane import LANE_DECLARATION_RELATIVE_PATH
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bound_lane_client_transport,
)

pytestmark = pytest.mark.integration

DECLARED_DEV_BROKER = "declared-dev.example:19092"
STORED_PRINCIPAL = "dev-cli-under-test"
AMBIENT_PRINCIPAL = "ambient-principal-that-must-not-win"

_DECLARATION = f"""
lanes:
  dev:
    broker: "{DECLARED_DEV_BROKER}"
    security_protocol: SASL_PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
"""


@pytest.fixture
def onex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real ``~/.onex`` root, in a temp directory, for both commands."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setenv("KAFKA_SASL_USERNAME", AMBIENT_PRINCIPAL)
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", "ambient-value")
    # The delegate orchestrator contract ships in omnimarket, which repo
    # layering forbids depending on here. Without this every delegation would
    # refuse on the missing node rather than on its identity -- the same
    # stand-in reason the task-class vocabulary above carries.
    monkeypatch.setattr(
        cli_delegate,
        "_resolve_packaged_contract",
        lambda _name: tmp_path / "contract.yaml",
    )
    return home / ".onex"


@pytest.fixture
def captured_dispatch(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """What the dispatch would have been handed, and what was bound to it."""
    captured: dict[str, object] = {}

    def _fake_run_receipt_mode(**kwargs: object) -> int:
        captured.update(kwargs)
        # The binding is scoped to the dispatch and is already gone by the
        # time the command returns, so it is read from inside.
        captured["lane_transport"] = bound_lane_client_transport()
        return 0

    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)
    return captured


def _workspace(root: Path) -> Path:
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(_DECLARATION, encoding="utf-8")
    return root


def _store_identity_through_the_real_command() -> object:
    return CliRunner().invoke(
        auth_group,
        [
            "lane-login",
            "--lane",
            "dev",
            "--sasl-username",
            STORED_PRINCIPAL,
            "--sasl-password-stdin",
        ],
        input="stored-value\n",
        catch_exceptions=False,
    )


def _delegate(tmp_path: Path) -> object:
    return CliRunner().invoke(
        delegate_command,
        [
            "document the router",
            "--task-type",
            "document",
            "--bus",
            "kafka",
            # The subject is identity, so the locus is pinned in-process --
            # otherwise this would also require a live consumer group.
            "--locus",
            "in-process",
            "--lane",
            "dev",
            "--omnibase-path",
            str(_workspace(tmp_path / "workspace")),
            "--state-root",
            str(tmp_path / "state"),
        ],
        catch_exceptions=False,
    )


class TestAMachineWithNoIdentityIsRefused:
    def test_the_refusal_names_the_lane_the_declaration_and_the_remedy(
        self, tmp_path: Path, onex_home: Path, captured_dispatch: dict[str, object]
    ) -> None:
        """RED before the fix: this opened an anonymous connect and hung."""
        # The ambient half-credential is deliberately incomplete here, so the
        # test proves the refusal rather than an environment fall-through.
        result = _delegate(tmp_path)

        assert result.exit_code != 0
        output = str(result.output)
        assert "dev" in output
        assert "ci_bus_lanes.yaml" in output
        assert "lane-login" in output
        assert captured_dispatch == {}, "nothing may be dispatched on a refusal"

    @pytest.fixture(autouse=True)
    def _no_ambient_credential(
        self, onex_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requests ``onex_home`` so it runs AFTER the fixture that sets them.

        Ordering matters and is stated rather than assumed: without the
        dependency this fixture can run first and the ambient variables are
        then re-set underneath it, which would turn the refusal assertion into
        an environment-fallback assertion that still passes.
        """
        monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
        monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)


class TestTheStoredIdentityIsWhatTheDispatchIsHanded:
    def test_login_then_delegate_carries_the_declared_transport(
        self, tmp_path: Path, onex_home: Path, captured_dispatch: dict[str, object]
    ) -> None:
        login = _store_identity_through_the_real_command()
        assert login.exit_code == 0, login.output

        result = _delegate(tmp_path)
        assert result.exit_code == 0, result.output

        transport = captured_dispatch["lane_transport"]
        assert transport is not None
        assert transport.lane == "dev"
        assert transport.bootstrap_servers == DECLARED_DEV_BROKER
        assert transport.security_protocol == "SASL_PLAINTEXT"
        assert transport.sasl_mechanism == "SCRAM-SHA-256"
        assert transport.sasl_username == STORED_PRINCIPAL
        assert transport.sasl_password is not None
        assert transport.sasl_password.get_secret_value() == "stored-value"

    def test_the_stored_identity_beats_the_ambient_environment(
        self, tmp_path: Path, onex_home: Path, captured_dispatch: dict[str, object]
    ) -> None:
        """The ambient variables are set by the fixture and must not win."""
        _store_identity_through_the_real_command()

        assert _delegate(tmp_path).exit_code == 0

        transport = captured_dispatch["lane_transport"]
        assert transport is not None
        assert transport.sasl_username == STORED_PRINCIPAL
        assert transport.sasl_username != AMBIENT_PRINCIPAL

    def test_the_value_reaches_no_command_output(
        self, tmp_path: Path, onex_home: Path, captured_dispatch: dict[str, object]
    ) -> None:
        login = _store_identity_through_the_real_command()
        delegated = _delegate(tmp_path)

        assert "stored-value" not in str(login.output)
        assert "stored-value" not in str(delegated.output)

    def test_the_address_still_travels_in_the_backend_overrides(
        self, tmp_path: Path, onex_home: Path, captured_dispatch: dict[str, object]
    ) -> None:
        """The core seam is unchanged: one address key, nothing more."""
        _store_identity_through_the_real_command()
        assert _delegate(tmp_path).exit_code == 0

        assert captured_dispatch["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": DECLARED_DEV_BROKER,
        }

    def test_the_binding_does_not_outlive_the_dispatch(
        self, tmp_path: Path, onex_home: Path, captured_dispatch: dict[str, object]
    ) -> None:
        """A credential that survives the command would reach the next one."""
        _store_identity_through_the_real_command()
        assert _delegate(tmp_path).exit_code == 0

        assert captured_dispatch["lane_transport"] is not None
        assert bound_lane_client_transport() is None

    def test_logout_returns_the_machine_to_the_refusal(
        self, tmp_path: Path, onex_home: Path, captured_dispatch: dict[str, object]
    ) -> None:
        """The positive control for the refusal: same machine, one command apart."""
        _store_identity_through_the_real_command()
        assert _delegate(tmp_path).exit_code == 0

        CliRunner().invoke(
            auth_group, ["lane-logout", "--lane", "dev"], catch_exceptions=False
        )
        captured_dispatch.clear()

        # The ambient environment is still set by the fixture, so this is also
        # the assertion that a full ambient credential DOES answer once the
        # machine holds no identity of its own -- the container and CI path.
        assert _delegate(tmp_path).exit_code == 0
        assert captured_dispatch["lane_transport"] is None
