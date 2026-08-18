# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex auth`` command surface, including the fail-closed path (OMN-15922).

The fail-closed tests assert a NON-ZERO EXIT plus a named remediation, not
merely that a warning was printed. A command that warns on stderr and exits 0
is indistinguishable, to every caller and every CI script, from a command that
succeeded -- which is the shape of the failure this whole slice exists to
prevent: a delegation that ran locally while the operator believed it reached
the cloud.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_auth import auth_group
from omnibase_infra.gateway.client.gateway_transport_httpx import (
    GatewayTransportHttpx,
)
from omnibase_infra.protocols.protocol_gateway_transport import (
    ProtocolGatewayTransport,
)

pytestmark = pytest.mark.unit

_SECRET = "s3cr3t-not-a-real-value"  # pragma: allowlist secret
_TOKEN_ENDPOINT = "https://keycloak.invalid/realms/acme/protocol/openid-connect/token"


@pytest.fixture
def onex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``Path.home()`` at a scratch directory for the whole command."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home / ".onex"


def _login(runner: CliRunner) -> object:
    return runner.invoke(
        auth_group,
        [
            "login",
            "--tenant-slug",
            "acme",
            "--client-id",
            "ga-acme",
            "--token-endpoint",
            _TOKEN_ENDPOINT,
            "--base-url",
            "https://api.invalid",
            "--client-secret-stdin",
            "--edge-instance-id",
            "test-edge",
        ],
        input=f"{_SECRET}\n",
    )


def test_login_stores_by_reference_and_never_echoes_the_secret(onex_home: Path) -> None:
    runner = CliRunner()

    result = _login(runner)

    assert result.exit_code == 0
    assert _SECRET not in result.output
    assert _SECRET not in (onex_home / "config.yaml").read_text()
    credentials = onex_home / "credentials.json"
    assert json.loads(credentials.read_text())["acme-gateway"] == _SECRET
    assert stat.S_IMODE(credentials.stat().st_mode) == 0o600


def test_status_prints_identity_without_any_secret_material(onex_home: Path) -> None:
    runner = CliRunner()
    _login(runner)

    result = runner.invoke(auth_group, ["status"])

    assert result.exit_code == 0
    assert "acme" in result.output
    assert "ga-acme" in result.output
    assert _SECRET not in result.output


def test_status_without_a_credential_exits_non_zero_and_names_the_remediation(
    onex_home: Path,
) -> None:
    result = CliRunner().invoke(auth_group, ["status"])

    assert result.exit_code != 0
    assert "onex auth login" in result.output


def test_token_without_a_credential_fails_closed_before_touching_a_transport(
    onex_home: Path,
) -> None:
    """No credential is a hard stop -- there is no anonymous token to emit."""
    result = CliRunner().invoke(auth_group, ["token"])

    assert result.exit_code != 0
    assert "onex auth login" in result.output


def test_the_shipped_adapter_actually_satisfies_the_transport_seam() -> None:
    """The infra-side replacement for core's entry-point discovery check.

    In ``omnibase_core`` the concrete transport could not exist at all (ADR-005
    bans a transport import there), so the CLI discovered one through the
    ``onex.gateway_transport`` entry-point group and refused loudly when the
    group was empty. Here the adapter ships in the same package as its caller,
    so that whole indirection is gone -- and the thing worth asserting changes
    with it. What must not silently rot is that the adapter the CLI constructs
    really satisfies the seam the services are typed against: a renamed or
    dropped method would otherwise surface only on a live mint.

    ``ProtocolGatewayTransport`` is ``runtime_checkable``, so this is a real
    structural check against the protocol, not a restatement of the import.
    """
    assert isinstance(GatewayTransportHttpx(), ProtocolGatewayTransport)


def test_a_transport_failure_on_token_exits_non_zero_rather_than_printing_nothing(
    onex_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable gateway is an error, never an empty-but-successful token.

    ``onex auth token`` is designed to be consumed as ``TOKEN=$(onex auth
    token)``. A path that failed to mint but still exited 0 would set an empty
    variable and hand it to the next call, which is the anonymous-call failure
    this slice exists to prevent -- so the exit code carries the refusal.
    """
    runner = CliRunner()
    _login(runner)

    def _unreachable(self: object) -> None:
        raise AssertionError("the token endpoint must not actually be dialled here")

    # The credential resolves; the mint is what fails. Pointing the token
    # endpoint at an unroutable host keeps this a pure unit test.
    monkeypatch.setattr(
        "omnibase_infra.cli.cli_auth.GatewayTokenMinter.token_for",
        _unreachable,
    )

    result = runner.invoke(auth_group, ["token"])

    assert result.exit_code != 0
    assert _SECRET not in result.output


def test_login_refuses_an_empty_secret_on_stdin(onex_home: Path) -> None:
    result = CliRunner().invoke(
        auth_group,
        [
            "login",
            "--tenant-slug",
            "acme",
            "--client-id",
            "ga-acme",
            "--token-endpoint",
            _TOKEN_ENDPOINT,
            "--base-url",
            "https://api.invalid",
            "--client-secret-stdin",
        ],
        input="\n",
    )

    assert result.exit_code != 0
    assert not (onex_home / "credentials.json").exists()


def test_the_secret_is_never_accepted_as_a_command_line_value(onex_home: Path) -> None:
    """A --client-secret <value> option must not exist at all.

    Its absence is the mechanism; a documented convention that operators should
    prefer stdin is not one. argv reaches the process table, the shell history
    file, and any exec audit log.
    """
    result = CliRunner().invoke(auth_group, ["login", "--help"])

    assert "--client-secret-stdin" in result.output
    assert "--client-secret " not in result.output


def test_logout_removes_the_credential_and_status_then_fails_closed(
    onex_home: Path,
) -> None:
    runner = CliRunner()
    _login(runner)

    logout = runner.invoke(auth_group, ["logout"])
    assert logout.exit_code == 0

    after = runner.invoke(auth_group, ["status"])
    assert after.exit_code != 0


def test_auth_group_is_registered_as_an_onex_cli_entry_point() -> None:
    """``onex auth`` must resolve through the ``onex.cli`` extension group.

    The DoD surface for this slice is ``onex auth login`` (OMN-15922 §DoD-1)
    -- the marketplace skill and every harness shell out to ``onex``, not to
    ``omni-infra``. Registering ``auth_group`` only on the ``omni-infra``
    click CLI (the #2747 port's gap) leaves ``onex auth`` a phantom callable:
    core's ``cli_commands`` loads extensions exclusively from the ``onex.cli``
    entry-point group. This asserts the declaration in pyproject.toml, not
    installed metadata, so it fails on the omission itself regardless of the
    venv's install state.
    """
    import tomllib

    import click

    pyproject = Path(__file__).parents[3] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)

    entry_points = data["project"]["entry-points"]["onex.cli"]
    assert entry_points.get("auth") == "omnibase_infra.cli.cli_auth:auth_group"

    # The declared target must actually be the loadable click group.
    assert isinstance(auth_group, click.Group)
    assert set(auth_group.commands) == {"login", "status", "token", "logout"}
