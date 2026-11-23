# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The C28 probe's rpk calls authenticate as the probe's own broker login (OMN-19812).

Since the stability broker required SASL (OMN-26121) the probe's rpk wrapper, which
set a login only from the broker container's own ``DEV_KAFKA_SASL_USERNAME``, ran
unauthenticated and every scheduled run graded INDETERMINATE ("SASL is required but
not provided"). The probe now runs as its own SCRAM login: the user and mechanism ride
``docker exec -e``, the password rides ``--env-file`` (a mode-600 file the workflow
wrote from the dev secret store), and a login that cannot be read is INDETERMINATE
before any rpk call, never an unauthenticated call.
"""

from __future__ import annotations

import asyncio
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_board_probe_effect.handlers import (
    handler_docker_consumer_flow_target as target_module,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
    ConsumerFlowLane,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_consumer_flow_target import (
    HandlerDockerConsumerFlowTarget,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_lane import (
    ModelConsumerFlowLane,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)

pytestmark = pytest.mark.unit

PRINCIPAL = "stability-c28-probe"
SENTINEL = "sentinel-password-never-in-argv-or-errors"


def _login_file(tmp_path: Path, *, mode: int = 0o600, body: str | None = None) -> Path:
    path = tmp_path / "c28-broker-login.env"
    path.write_text(body if body is not None else f"RPK_PASS={SENTINEL}\n")
    path.chmod(mode)
    return path


def _lane(
    runner: Any, *, principal: str | None, env_file: Path | None
) -> ConsumerFlowLane:
    return ConsumerFlowLane(
        containers=ModelConsumerFlowLane.from_compose_project("omnibase-infra-c28lane"),
        docker="fake-docker",
        base_url="http://projection.test",
        runner=runner,
        broker_principal=principal,
        broker_password_env_file=env_file,
    )


def _request(tmp_path: Path, **extra: Any) -> ModelConsumerFlowRequest:
    return ModelConsumerFlowRequest(
        subject_lane="c28lane",
        compose_project="omnibase-infra-c28lane",
        scratch=tmp_path,
        **extra,
    )


def test_rpk_runs_as_the_probe_login_with_the_password_in_no_argv(
    tmp_path: Path,
) -> None:
    env_file = _login_file(tmp_path)
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    lane = _lane(runner, principal=PRINCIPAL, env_file=env_file)

    assert lane.rpk("topic", "describe", "-p", "t") == "ok"
    argv = calls[0]
    assert argv[:3] == ["fake-docker", "exec", "-i"]
    pairs = list(pairwise(argv))
    assert ("-e", f"RPK_USER={PRINCIPAL}") in pairs
    assert ("-e", "RPK_SASL_MECHANISM=SCRAM-SHA-256") in pairs
    assert ("--env-file", str(env_file)) in pairs
    broker = argv.index("omnibase-infra-c28lane-redpanda")
    assert argv[broker + 1 : broker + 3] == ["sh", "-c"]
    assert argv[broker + 4 :] == ["rpk", "topic", "describe", "-p", "t"]
    script = argv[broker + 3]
    assert 'rpk "$@" -X brokers="redpanda:9092"' in script
    # No fallback to the container's own credential, and no password in any argv.
    assert "DEV_KAFKA_SASL" not in script
    assert not any(SENTINEL in part for part in argv)
    assert not any(part.startswith("RPK_PASS") for part in argv)


def test_rpk_without_a_probe_login_keeps_the_container_credential_form() -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    _lane(runner, principal=None, env_file=None).rpk("topic", "list")
    assert "--env-file" not in calls[0]
    assert calls[0][6].startswith('if [ -n "${DEV_KAFKA_SASL_USERNAME:-}" ]; then ')


def test_request_names_the_principal_and_the_file_together_or_not_at_all(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="together or not at all"):
        _request(tmp_path, broker_principal=PRINCIPAL)
    with pytest.raises(ValidationError, match="together or not at all"):
        _request(tmp_path, broker_password_env_file=tmp_path / "x.env")
    with pytest.raises(ValidationError):
        _request(
            tmp_path,
            broker_principal="Bad Principal; rm -rf",
            broker_password_env_file=tmp_path / "x.env",
        )


def _observe_with(tmp_path: Path, env_file: Path, calls: list[list[str]]) -> Any:
    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    handler = HandlerDockerConsumerFlowTarget(runner=runner, repo_root=tmp_path)
    return asyncio.run(
        handler.observe(
            _request(
                tmp_path,
                broker_principal=PRINCIPAL,
                broker_password_env_file=env_file,
            )
        )
    )


def test_a_login_that_was_not_read_is_unreadable_and_makes_no_call(
    tmp_path: Path,
) -> None:
    """The workflow leaves no file and a reason when the dev store read fails."""
    env_file = tmp_path / "c28-broker-login.env"
    (tmp_path / "c28-broker-login.env.reason").write_text(
        "INFISICAL_CLIENT_ID is not set in the runner environment\n"
    )
    calls: list[list[str]] = []

    obs = _observe_with(tmp_path, env_file, calls)

    assert obs.read_ok is False
    assert PRINCIPAL in obs.read_error
    assert "INFISICAL_CLIENT_ID is not set in the runner environment" in obs.read_error
    assert calls == []  # not one docker call, so never an unauthenticated rpk


@pytest.mark.parametrize(
    ("mode", "body", "expected"),
    [
        (0o640, f"RPK_PASS={SENTINEL}\n", "not a private regular file"),
        (0o600, "", "holds no RPK_PASS value"),
        (0o600, "RPK_PASS=\n", "holds no RPK_PASS value"),
        (0o600, f"OTHER={SENTINEL}\n", "holds no RPK_PASS value"),
    ],
)
def test_an_unusable_password_file_is_unreadable_and_never_echoes_the_password(
    tmp_path: Path, mode: int, body: str, expected: str
) -> None:
    env_file = _login_file(tmp_path, mode=mode, body=body)
    calls: list[list[str]] = []

    obs = _observe_with(tmp_path, env_file, calls)

    assert obs.read_ok is False
    assert expected in obs.read_error
    assert SENTINEL not in obs.read_error
    assert calls == []


def test_a_readable_login_reaches_the_lane_and_rpk_authenticates_as_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = _login_file(tmp_path)
    calls: list[list[str]] = []
    seen: dict[str, Any] = {}

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    def fake_observe_lane(lane: ConsumerFlowLane, **_: Any) -> dict[str, Any]:
        seen["lane"] = lane
        lane.rpk("topic", "list")
        raise OSError("stop after the first rpk call")

    monkeypatch.setattr(target_module, "observe_lane", fake_observe_lane)
    handler = HandlerDockerConsumerFlowTarget(runner=runner, repo_root=tmp_path)
    obs = asyncio.run(
        handler.observe(
            _request(
                tmp_path,
                broker_principal=PRINCIPAL,
                broker_password_env_file=env_file,
            )
        )
    )

    assert obs.read_ok is False  # the fake stopped the walk; only the rpk call matters
    assert seen["lane"].broker_principal == PRINCIPAL
    rpk_call = calls[0]
    assert ("--env-file", str(env_file)) in list(pairwise(rpk_call))
    assert f"RPK_USER={PRINCIPAL}" in rpk_call
