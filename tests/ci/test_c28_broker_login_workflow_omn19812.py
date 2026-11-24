# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The C28 workflow reads the probe's broker login by name and never runs unauthenticated (OMN-19812).

The stability broker requires SASL, so the probe's rpk calls run as its own SCRAM login
``stability-c28-probe``. The workflow step under test reads that login's password from the
dev secret store, by name, with the runner host's machine identity, into a mode-600 file;
the node hands it to ``docker exec --env-file``. The step's own text runs here against a
loopback stub of the store: no repository secret, no password in any output, and every
failed read leaves no file and a reason, which the node grades INDETERMINATE.
"""

from __future__ import annotations

import json
import re
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)

pytestmark = pytest.mark.unit

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/chain-canary-c28-consumer-flow.yml"
)
READ_STEP = "Read the probe's broker login from the dev secret store"
SECRET_NAME = "STABILITY_TEST_KAFKA_SASL_PASSWORD_STABILITY_C28_PROBE"
VALUE = "stub-store-password-value"
CLIENT_SECRET = "stub-client-secret-value"


def _steps() -> list[dict[str, Any]]:
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["c28-consumer-flow"]
    return list(job["steps"])


def _step(name: str) -> dict[str, Any]:
    return next(step for step in _steps() if step["name"] == name)


def _read_step_python() -> str:
    run = _step(READ_STEP)["run"]
    match = re.search(r"<<'PYCODE'\n(.*?)\n\s*PYCODE", run, re.DOTALL)
    assert match is not None, "the read step carries its python in a PYCODE heredoc"
    return match.group(1)


class _Store:
    """A loopback stand-in for the dev secret store's two endpoints."""

    def __init__(
        self,
        *,
        login_status: int = 200,
        login_body: dict[str, Any] | None = None,
        secret: dict[str, Any] | None = None,
    ):
        self.login_status = login_status
        self.login_body = (
            {"accessToken": "stub-token"} if login_body is None else login_body
        )
        self.secret = {"secret": {"secretValue": VALUE}} if secret is None else secret
        self.seen: list[tuple[str, str, dict[str, str]]] = []
        store = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                return

            def _send(self, status: int, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                store.seen.append(
                    ("POST", self.path, {"client": body.get("clientId", "")})
                )
                if body.get("clientSecret") != CLIENT_SECRET:
                    self._send(401, {"error": "bad identity"})
                    return
                if store.login_status != 200:
                    self._send(store.login_status, {"error": "refused"})
                    return
                self._send(200, store.login_body)

            def do_GET(self) -> None:
                store.seen.append(
                    ("GET", self.path, {"auth": self.headers.get("Authorization", "")})
                )
                if self.headers.get("Authorization") != "Bearer stub-token":
                    self._send(401, {"error": "no token"})
                    return
                if not self.path.startswith(f"/api/v3/secrets/raw/{SECRET_NAME}?"):
                    self._send(404, {"error": "no such secret"})
                    return
                self._send(200, store.secret)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> _Store:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def addr(self) -> str:
        host, port = self.server.server_address[:2]
        return f"{host!s}:{port}"


def _run_read_step(
    tmp_path: Path, env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], Path]:
    full_env = {
        "PATH": "/usr/bin:/bin",
        "RUNNER_TEMP": str(tmp_path),
        "C28_BROKER_PRINCIPAL": "stability-c28-probe",
        **env,
    }
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _read_step_python()],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=60,
        check=False,
    )
    return proc, tmp_path / "c28-broker-login.env"


def _identity(store: _Store) -> dict[str, str]:
    return {
        "INFISICAL_ADDR": store.addr,
        "INFISICAL_PROJECT_ID": "proj-1",
        "INFISICAL_CLIENT_ID": "client-1",
        "INFISICAL_CLIENT_SECRET": CLIENT_SECRET,
    }


def test_the_login_is_read_by_name_into_a_private_file_and_never_printed(
    tmp_path: Path,
) -> None:
    with _Store() as store:
        proc, target = _run_read_step(tmp_path, _identity(store))
    assert proc.returncode == 0, proc.stderr
    assert target.read_text() == f"RPK_PASS={VALUE}\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not target.with_name(target.name + ".reason").exists()
    for stream in (proc.stdout, proc.stderr):
        assert VALUE not in stream
        assert CLIENT_SECRET not in stream
        assert "stub-token" not in stream
    get = next(item for item in store.seen if item[0] == "GET")
    assert get[1].startswith(f"/api/v3/secrets/raw/{SECRET_NAME}?")
    assert "environment=dev" in get[1]
    assert "secretPath=%2Fshared%2Fenv%2F" in get[1]
    assert "workspaceId=proj-1" in get[1]


@pytest.mark.live_contact("tests/ci/fixtures/c28_broker_login_store_omn19812.json")
def test_the_step_reads_the_responses_the_live_store_recorded(
    recorded_response: dict[str, object], tmp_path: Path
) -> None:
    """Replay the real login and raw-secret responses (values redacted), body for body."""
    recorded_login = recorded_response["login_response"]
    assert isinstance(recorded_login, dict)
    login = dict(recorded_login)
    secret = json.loads(json.dumps(recorded_response["secret_response"]))
    assert login["accessToken"] == "<redacted>"
    assert secret["secret"]["secretKey"] == SECRET_NAME
    assert secret["secret"]["secretValue"] == "<redacted>"
    login["accessToken"] = "stub-token"
    secret["secret"]["secretValue"] = VALUE
    with _Store(login_body=login, secret=secret) as store:
        proc, target = _run_read_step(tmp_path, _identity(store))
    assert proc.returncode == 0, proc.stderr
    assert target.read_text() == f"RPK_PASS={VALUE}\n"
    assert VALUE not in proc.stdout + proc.stderr


def test_the_secret_name_is_the_principal_upper_cased_with_underscores() -> None:
    step = _step(READ_STEP)
    assert step["env"]["C28_BROKER_PRINCIPAL"] == "stability-c28-probe"
    assert (
        "STABILITY_TEST_KAFKA_SASL_PASSWORD_"
        + step["env"]["C28_BROKER_PRINCIPAL"].upper().replace("-", "_")
        == SECRET_NAME
    )


def test_no_identity_on_the_runner_leaves_no_file_and_names_what_is_missing(
    tmp_path: Path,
) -> None:
    proc, target = _run_read_step(tmp_path, {})
    assert proc.returncode == 0, proc.stderr
    assert not target.exists()
    reason = target.with_name(target.name + ".reason").read_text()
    for name in (
        "INFISICAL_ADDR",
        "INFISICAL_PROJECT_ID",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
    ):
        assert name in reason
    assert SECRET_NAME in reason
    assert "::warning::" in proc.stdout


@pytest.mark.parametrize(
    ("store_kwargs", "identity_secret", "expected"),
    [
        ({}, "wrong-client-secret", "HTTP 401"),
        ({"login_status": 503}, CLIENT_SECRET, "HTTP 503"),
        (
            {"secret": {"secret": {"secretValue": ""}}},
            CLIENT_SECRET,
            "single-line value",
        ),
        (
            {"secret": {"secret": {"secretValue": "two words"}}},
            CLIENT_SECRET,
            "single-line value",
        ),
        ({"secret": {}}, CLIENT_SECRET, "single-line value"),
    ],
)
def test_a_failed_store_read_leaves_no_file_and_a_reason_never_the_value(
    tmp_path: Path,
    store_kwargs: dict[str, Any],
    identity_secret: str,
    expected: str,
) -> None:
    with _Store(**store_kwargs) as store:
        env = {**_identity(store), "INFISICAL_CLIENT_SECRET": identity_secret}
        proc, target = _run_read_step(tmp_path, env)
    assert proc.returncode == 0, proc.stderr
    assert not target.exists()
    reason = target.with_name(target.name + ".reason").read_text()
    assert expected in reason
    for text in (reason, proc.stdout, proc.stderr):
        assert VALUE not in text
        assert identity_secret not in text


def test_an_unanswering_store_is_a_reason_not_a_crash(tmp_path: Path) -> None:
    env = {
        "INFISICAL_ADDR": "127.0.0.1:9",
        "INFISICAL_PROJECT_ID": "proj-1",
        "INFISICAL_CLIENT_ID": "client-1",
        "INFISICAL_CLIENT_SECRET": CLIENT_SECRET,
    }
    proc, target = _run_read_step(tmp_path, env)
    assert proc.returncode == 0, proc.stderr
    assert not target.exists()
    assert "failed:" in target.with_name(target.name + ".reason").read_text()


def test_the_read_precedes_the_node_and_the_cleanup_runs_always_before_the_upload() -> (
    None
):
    names = [step["name"] for step in _steps()]
    read = names.index(READ_STEP)
    prepare = names.index("Prepare the consumer-flow node request")
    node = names.index("Re-measure C28 on the stability lane and grade it")
    cleanup = names.index("Remove the probe's broker login")
    upload = names.index("Upload the C28 consumer-flow record")
    assert read < prepare < node < cleanup < upload
    assert _steps()[cleanup]["if"] == "always()"


def test_the_request_names_the_principal_and_the_file_the_node_validates() -> None:
    prepare = _step("Prepare the consumer-flow node request")
    assert prepare["env"]["C28_BROKER_PRINCIPAL"] == "stability-c28-probe"
    assert "--arg broker_principal" in prepare["run"]
    assert (
        '--arg broker_password_env_file "$RUNNER_TEMP/c28-broker-login.env"'
        in prepare["run"]
    )
    assert "broker_principal: $broker_principal" in prepare["run"]
    assert "broker_password_env_file: $broker_password_env_file" in prepare["run"]
    request = ModelConsumerFlowRequest.model_validate(
        {
            "subject_lane": "c28lane",
            "compose_project": "omnibase-infra-c28lane",
            "broker_principal": "stability-c28-probe",
            "broker_password_env_file": "c28-broker-login.env",
        }
    )
    assert request.broker_principal == "stability-c28-probe"


def test_the_job_uses_no_repository_secret() -> None:
    text = WORKFLOW.read_text()
    assert "secrets." not in re.sub(r"(?m)^\s*#.*$", "", text)
