# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The C28 probe's broker login, through the handler, the grader and the record (OMN-19812).

The unit tests pin the argv of one rpk call. These run the node's own handler over
the Docker target with a real loopback HTTP projection, so the login's path is the
one the scheduled workflow takes: the request carries the principal and the
password file, the target reads the file before any Docker call, every rpk call
authenticates as that login, and the JSON record the workflow uploads holds the
verdict and never the password.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_consumer_flow import (
    HandlerConsumerFlow,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_consumer_flow_target import (
    HandlerDockerConsumerFlowTarget,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)

pytestmark = pytest.mark.integration

PRINCIPAL = "stability-c28-probe"
SENTINEL = "sentinel-password-never-in-argv-record-or-reasons"


class _RecordingDocker:
    """A fixed Docker view that remembers every argv it was handed."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if argv[1] == "inspect":
            output = json.dumps(
                [
                    {
                        "Id": argv[-1],
                        "State": {"Status": "running", "StartedAt": "stable"},
                    }
                ]
            )
        elif argv[1] == "logs":
            output = ""
        else:
            assert argv[1] == "exec"
            output = "PARTITION HIGH-WATERMARK\n0 10\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    def rpk_calls(self) -> list[list[str]]:
        return [argv for argv in self.calls if argv[1] == "exec"]


def _login_file(tmp_path: Path) -> Path:
    path = tmp_path / "c28-broker-login.env"
    path.write_text(f"RPK_PASS={SENTINEL}\n")
    path.chmod(0o600)
    return path


def test_an_unread_login_is_indeterminate_in_the_record_with_no_docker_call(
    tmp_path: Path,
) -> None:
    """The workflow leaves no file and its cause beside it when the secret read fails."""
    env_file = tmp_path / "c28-broker-login.env"
    (tmp_path / "c28-broker-login.env.reason").write_text(
        "dev secret store identity is not set in the runner environment\n"
    )
    record = tmp_path / "c28-login-absent.json"
    docker = _RecordingDocker()
    request = ModelConsumerFlowRequest(
        subject_lane="c28lane",
        compose_project="omnibase-infra-c28lane",
        settle_seconds=0,
        scratch=tmp_path,
        record=record,
        broker_principal=PRINCIPAL,
        broker_password_env_file=env_file,
    )

    result = asyncio.run(
        HandlerConsumerFlow(
            target=HandlerDockerConsumerFlowTarget(runner=docker)
        ).handle(request)
    )

    receipt = json.loads(record.read_text())
    assert result.outcome == "INDETERMINATE"
    assert not result.as_lab_proof_check_result().passed
    assert any(PRINCIPAL in reason for reason in result.reasons)
    assert any(
        "dev secret store identity is not set in the runner environment" in reason
        for reason in result.reasons
    )
    assert receipt["result"] == result.model_dump(mode="json")
    assert receipt["observation"]["read_ok"] is False
    assert PRINCIPAL in receipt["observation"]["read_error"]
    assert not any(
        call[1] == "exec" for call in docker.calls
    )  # never an unauthenticated rpk


def test_a_readable_login_authenticates_every_rpk_call_and_stays_out_of_the_record(
    tmp_path: Path,
) -> None:
    env_file = _login_file(tmp_path)
    record = tmp_path / "c28-login-present.json"
    docker = _RecordingDocker()

    class RefusingProjection(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format_string: str, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), RefusingProjection) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = ModelConsumerFlowRequest(
                subject_lane="c28lane",
                compose_project="omnibase-infra-c28lane",
                base_url=f"http://127.0.0.1:{server.server_port}",
                settle_seconds=0,
                scratch=tmp_path,
                record=record,
                broker_principal=PRINCIPAL,
                broker_password_env_file=env_file,
            )
            result = asyncio.run(
                HandlerConsumerFlow(
                    target=HandlerDockerConsumerFlowTarget(runner=docker)
                ).handle(request)
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    rpk_calls = docker.rpk_calls()
    assert rpk_calls, "the probe made no rpk call, so the login was never exercised"
    for argv in rpk_calls:
        pairs = list(pairwise(argv))
        assert ("-e", f"RPK_USER={PRINCIPAL}") in pairs
        assert ("--env-file", str(env_file)) in pairs
        assert not any(SENTINEL in part for part in argv)
    assert result.outcome == "INDETERMINATE"  # the refused projection, not the login
    text = record.read_text()
    assert SENTINEL not in text
    assert SENTINEL not in json.dumps(result.model_dump(mode="json"))
