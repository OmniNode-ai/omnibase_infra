# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mint through the new target, then forget the login and exercise the real CLI.

The refusal/fallback guard predates OMN-20823. These cases add the new mint-local
entrypoint to that journey; their merge-base RED is the absent mint target, not
a claim that this PR introduced the existing credential guard. Docker, broker
and the mint probe use the recording fixture. Login, logout, credential storage
and delegate validation use their real production implementations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_auth import auth_group
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)
from tests.unit.cli.test_delegate_lane_local_omn20823 import _Machine
from tests.unit.infra.test_make_mint_local import _Stack

pytestmark = pytest.mark.unit

# The fixture records the real target's login and broker probe. Keep that
# sidecar, but first execute the real auth command with the same stdin secret.
_REAL_LOGIN = r"""
    from click.testing import CliRunner
    from omnibase_infra.cli.cli_auth import auth_group
    real_login = CliRunner().invoke(auth_group, argv[1:], input=password + "\n")
    if real_login.exit_code != 0:
        print(real_login.output, file=sys.stderr)
        sys.exit(real_login.exit_code)
"""


@pytest.mark.parametrize("ambient_pair", [False, True], ids=["refusal", "ambient"])
def test_mint_local_logout_journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ambient_pair: bool
) -> None:
    machine = _Machine(tmp_path, monkeypatch)
    stack = _Stack(tmp_path)
    onex = stack.bin / "onex"
    body = onex.read_text(encoding="utf-8")
    marker = '    password = sys.stdin.readline().rstrip("\\n")\n'
    assert body.count(marker) == 1
    onex.write_text(body.replace(marker, marker + _REAL_LOGIN), encoding="utf-8")

    minted = stack.run()
    assert minted.returncode == 0, minted.stdout + minted.stderr
    assert "mint-local: verified" in minted.stdout
    held = machine.store().load("local")
    assert held.sasl_username == stack.stored()["username"]
    assert held.sasl_password.get_secret_value() == stack.stored()["password"]

    # Positive control: the actual delegate surface binds the minted login.
    result = machine.delegate()
    assert result.exit_code == 0, result.output
    transport = machine.captured["lane_transport"]
    assert isinstance(transport, ModelLaneClientTransport)
    assert transport.sasl_username == held.sasl_username
    machine.captured.clear()

    logged_out = CliRunner().invoke(auth_group, ["lane-logout", "--lane", "local"])
    assert logged_out.exit_code == 0, logged_out.output
    assert "local" not in machine.store().declared_lanes()
    if ambient_pair:
        monkeypatch.setenv("KAFKA_SASL_USERNAME", "ambient-user")
        monkeypatch.setenv("KAFKA_SASL_PASSWORD", "ambient-value")

    result = machine.delegate()
    if ambient_pair:
        assert result.exit_code == 0, result.output
        assert machine.captured["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": "localhost:19092",
        }
        assert machine.captured["lane_transport"] is None
    else:
        assert result.exit_code != 0, result.output
        output = " ".join(result.output.split())
        assert "onex auth lane-login --lane local" in output
        assert "SASL_PLAINTEXT" in output
        assert machine.captured == {}, "the run was dispatched after logout"
