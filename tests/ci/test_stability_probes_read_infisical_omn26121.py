# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability probes read their credentials from the stability Infisical, not repository secrets (OMN-26121).

C15 (chain-canary.yml), C16 (chain-canary-c16-receipt-identity.yml) and C11
(chain-canary-c11-negative-paths.yml) used to read ``secrets.KAFKA_SASL_*``,
``secrets.CHAIN_CANARY_PROJECTION_DSN`` and ``secrets.CHAIN_CANARY_GATEWAY_API_KEY``.
Operator ruling 2026-11-23T23:06:19Z: stability reads its secrets from Infisical
through a stability machine identity. The workflows now call the composite action
``.github/actions/fetch-stability-secrets``, which logs in with the runner's four
``INFISICAL_*`` names, masks each fetched value, and exports it to the job
environment. The action's own text runs here against a loopback stub of the store:
no repository secret, no value outside the mask command, and a missing value fails
the step naming the variable, the Infisical secret name and the store path.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_ACTION = _ROOT / ".github/actions/fetch-stability-secrets/action.yml"
_WORKFLOWS = _ROOT / ".github/workflows"
_C15 = "chain-canary.yml"
_C16 = "chain-canary-c16-receipt-identity.yml"
_C11 = "chain-canary-c11-negative-paths.yml"

_STORE_NAMES = {
    "KAFKA_SASL_USERNAME": "STABILITY_TEST_KAFKA_SASL_USERNAME_STABILITY_CHAIN_CANARY",
    "KAFKA_SASL_PASSWORD": "STABILITY_TEST_KAFKA_SASL_PASSWORD_STABILITY_CHAIN_CANARY",
    "CHAIN_CANARY_PROJECTION_DSN": "STABILITY_TEST_CHAIN_CANARY_PROJECTION_DSN",
    "CHAIN_CANARY_GATEWAY_API_KEY": "STABILITY_TEST_CHAIN_CANARY_GATEWAY_API_KEY",
}
_NEEDS = {
    # OMN-23566: C15 submits over the bus; it carries no gateway credential.
    _C15: set(_STORE_NAMES) - {"CHAIN_CANARY_GATEWAY_API_KEY"},
    _C16: {
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
        "CHAIN_CANARY_GATEWAY_API_KEY",
    },
    _C11: {"CHAIN_CANARY_GATEWAY_API_KEY"},
}
_PROBE_STEP_PREFIX = {
    _C15: "Fire one live delegation",
    _C16: "Drive the healthy and the dying delegation",
    _C11: "Drive the four negative paths",
}
_CLIENT_SECRET = "stub-client-secret"
_VALUES = {name: f"stub-value-for-{name.lower()}" for name in _STORE_NAMES}


def _job(workflow: str) -> dict[str, Any]:
    jobs = yaml.safe_load((_WORKFLOWS / workflow).read_text())["jobs"]
    assert len(jobs) >= 1
    return next(
        job
        for job in jobs.values()
        if any(
            str(step.get("name", "")).startswith(_PROBE_STEP_PREFIX[workflow])
            for step in job["steps"]
        )
    )


def _fetch_step(workflow: str) -> tuple[int, dict[str, Any]]:
    steps = _job(workflow)["steps"]
    for index, step in enumerate(steps):
        if step.get("uses") == "./.github/actions/fetch-stability-secrets":
            return index, step
    pytest.fail(f"{workflow} never calls ./.github/actions/fetch-stability-secrets")


def _mapping(step: dict[str, Any]) -> dict[str, str]:
    pairs = {}
    for line in str(step["with"]["secrets"]).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        env_name, _, store_name = line.partition("=")
        pairs[env_name.strip()] = store_name.strip()
    return pairs


@pytest.mark.parametrize("workflow", [_C15, _C16, _C11])
def test_no_stability_credential_is_read_from_a_repository_secret(
    workflow: str,
) -> None:
    text = (_WORKFLOWS / workflow).read_text()
    for env_name in _STORE_NAMES:
        assert not re.search(rf"secrets\.{env_name}\b", text), (
            f"{workflow} still reads secrets.{env_name}; stability reads Infisical"
        )


@pytest.mark.parametrize("workflow", [_C15, _C16, _C11])
def test_the_fetch_precedes_the_probe_and_names_exactly_what_the_workflow_needs(
    workflow: str,
) -> None:
    index, step = _fetch_step(workflow)
    steps = _job(workflow)["steps"]
    probe_index = next(
        i
        for i, s in enumerate(steps)
        if str(s.get("name", "")).startswith(_PROBE_STEP_PREFIX[workflow])
    )
    assert index < probe_index, "the credentials are fetched before the probe runs"
    assert _mapping(step) == {name: _STORE_NAMES[name] for name in _NEEDS[workflow]}


@pytest.mark.parametrize("workflow", [_C15, _C16, _C11])
def test_the_probe_steps_do_not_re_declare_the_fetched_names(workflow: str) -> None:
    """A step-level env entry would override the exported, masked value."""
    for step in _job(workflow)["steps"]:
        for env_name in _STORE_NAMES:
            assert env_name not in (step.get("env") or {}), (
                f"{workflow}: step {step.get('name')!r} re-declares {env_name}"
            )


def _action_python() -> str:
    action = yaml.safe_load(_ACTION.read_text())
    run = next(s["run"] for s in action["runs"]["steps"] if "run" in s)
    match = re.search(r"<<'PYCODE'\n(.*?)\n\s*PYCODE", run, re.DOTALL)
    assert match is not None, "the action carries its python in a PYCODE heredoc"
    return match.group(1)


class _Store:
    """A loopback stand-in for the stability store's login and raw-secret routes."""

    def __init__(
        self,
        *,
        present: dict[str, str] | None = None,
        status_for: dict[str, int] | None = None,
        login_status: int = 200,
        login_body: dict[str, Any] | None = None,
        secret_body: dict[str, Any] | None = None,
    ):
        self.login_body = (
            {"accessToken": "stub-token"} if login_body is None else login_body
        )
        self.secret_body = secret_body
        self.present = dict(_VALUES_BY_STORE_NAME) if present is None else present
        self.status_for = status_for or {}
        self.login_status = login_status
        self.seen: list[tuple[str, str]] = []
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
                store.seen.append(("POST", self.path))
                if body.get("clientSecret") != _CLIENT_SECRET:
                    self._send(401, {"error": "bad identity"})
                elif store.login_status != 200:
                    self._send(store.login_status, {"error": "refused"})
                else:
                    self._send(200, store.login_body)

            def do_GET(self) -> None:
                store.seen.append(("GET", self.path))
                if self.headers.get("Authorization") != "Bearer stub-token":
                    self._send(401, {"error": "no token"})
                    return
                name = urlsplit(self.path).path.rsplit("/", 1)[-1]
                if name in store.status_for:
                    self._send(store.status_for[name], {"error": "refused"})
                elif store.secret_body is not None and name in store.present:
                    self._send(200, store.secret_body)
                elif name in store.present:
                    self._send(200, {"secret": {"secretValue": store.present[name]}})
                else:
                    self._send(404, {"error": "no such secret"})

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


_VALUES_BY_STORE_NAME = {_STORE_NAMES[name]: _VALUES[name] for name in _STORE_NAMES}


def _run_action(
    tmp_path: Path,
    store: _Store,
    *,
    names: tuple[str, ...] = tuple(_STORE_NAMES),
    drop_env: tuple[str, ...] = (),
    secret_map: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    github_env = tmp_path / "github_env"
    github_env.write_text("")
    env = {
        "PATH": "/usr/bin:/bin",
        "GITHUB_ENV": str(github_env),
        "STABILITY_SECRET_MAP": secret_map
        or "\n".join(f"{n}={_STORE_NAMES[n]}" for n in names),
        "INFISICAL_ADDR": store.addr,
        "INFISICAL_PROJECT_ID": "proj-1",
        "INFISICAL_CLIENT_ID": "client-1",
        "INFISICAL_CLIENT_SECRET": _CLIENT_SECRET,
        "STABILITY_STORE_ENVIRONMENT": "dev",
        "STABILITY_STORE_PATH": "/shared/env/",
    }
    for name in drop_env:
        env.pop(name)
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _action_python()],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    return proc, github_env


def _unmasked(output: str) -> str:
    return "\n".join(
        line for line in output.splitlines() if not line.startswith("::add-mask::")
    )


def test_every_value_is_masked_then_exported_and_never_otherwise_printed(
    tmp_path: Path,
) -> None:
    with _Store() as store:
        proc, github_env = _run_action(tmp_path, store)
    assert proc.returncode == 0, proc.stderr
    exported = github_env.read_text().splitlines()
    assert sorted(exported) == sorted(f"{n}={v}" for n, v in _VALUES.items())
    masks = [row for row in proc.stdout.splitlines() if row.startswith("::add-mask::")]
    assert sorted(masks) == sorted(f"::add-mask::{v}" for v in _VALUES.values())
    for stream in (_unmasked(proc.stdout), proc.stderr):
        for value in _VALUES.values():
            assert value not in stream
        assert _CLIENT_SECRET not in stream
        assert "stub-token" not in stream
    gets = [path for method, path in store.seen if method == "GET"]
    assert len(gets) == 4
    for path in gets:
        assert "environment=dev" in path
        assert "secretPath=%2Fshared%2Fenv%2F" in path
        assert "workspaceId=proj-1" in path


def test_a_missing_value_fails_naming_the_variable_the_secret_and_the_path(
    tmp_path: Path,
) -> None:
    present = {
        k: v
        for k, v in _VALUES_BY_STORE_NAME.items()
        if k != _STORE_NAMES["CHAIN_CANARY_GATEWAY_API_KEY"]
    }
    with _Store(present=present) as store:
        proc, github_env = _run_action(tmp_path, store)
    assert proc.returncode == 1
    named = proc.stdout + proc.stderr
    assert "CHAIN_CANARY_GATEWAY_API_KEY" in named
    assert _STORE_NAMES["CHAIN_CANARY_GATEWAY_API_KEY"] in named
    assert "dev/shared/env/" in named
    assert "::error::" in named
    assert github_env.read_text() == "", "a partial set is never exported"
    for value in _VALUES.values():
        assert value not in _unmasked(named)


def test_a_refused_read_fails_naming_the_status_not_the_value(tmp_path: Path) -> None:
    status_for = {_STORE_NAMES["KAFKA_SASL_PASSWORD"]: 403}
    with _Store(status_for=status_for) as store:
        proc, github_env = _run_action(tmp_path, store)
    assert proc.returncode == 1
    assert "KAFKA_SASL_PASSWORD" in proc.stdout + proc.stderr
    assert "403" in proc.stdout + proc.stderr
    assert github_env.read_text() == ""


def test_a_runner_without_the_identity_fails_naming_each_missing_name(
    tmp_path: Path,
) -> None:
    with _Store() as store:
        proc, github_env = _run_action(
            tmp_path, store, drop_env=("INFISICAL_CLIENT_ID", "INFISICAL_CLIENT_SECRET")
        )
    assert proc.returncode == 1
    assert "INFISICAL_CLIENT_ID" in proc.stdout + proc.stderr
    assert "INFISICAL_CLIENT_SECRET" in proc.stdout + proc.stderr
    assert github_env.read_text() == ""
    assert store.seen == [], "no call is made without the full identity"


def test_a_refused_login_fails_without_exporting_anything(tmp_path: Path) -> None:
    with _Store(login_status=403) as store:
        proc, github_env = _run_action(tmp_path, store)
    assert proc.returncode == 1
    assert "403" in proc.stdout + proc.stderr
    assert github_env.read_text() == ""


@pytest.mark.live_contact("tests/ci/fixtures/c28_broker_login_store_omn19812.json")
def test_the_action_reads_the_responses_the_live_store_recorded(
    recorded_response: dict[str, object], tmp_path: Path
) -> None:
    """Replay the real login and raw-secret responses (values redacted), body for body.

    The fixture is the live stability store's own answer to the two routes the
    action calls, recorded on 2026-11-23 for the C28 probe's read of one name.
    """
    login = dict(cast("dict[str, Any]", recorded_response["login_response"]))
    secret = json.loads(json.dumps(recorded_response["secret_response"]))
    key = secret["secret"]["secretKey"]
    assert login["accessToken"] == "<redacted>"
    assert secret["secret"]["secretValue"] == "<redacted>"
    login["accessToken"] = "stub-token"
    secret["secret"]["secretValue"] = _VALUES["KAFKA_SASL_PASSWORD"]
    with _Store(
        present={key: _VALUES["KAFKA_SASL_PASSWORD"]},
        login_body=login,
        secret_body=secret,
    ) as store:
        proc, github_env = _run_action(
            tmp_path, store, secret_map=f"KAFKA_SASL_PASSWORD={key}"
        )
    assert proc.returncode == 0, proc.stderr
    assert (
        github_env.read_text()
        == f"KAFKA_SASL_PASSWORD={_VALUES['KAFKA_SASL_PASSWORD']}\n"
    )
    assert _VALUES["KAFKA_SASL_PASSWORD"] not in _unmasked(proc.stdout + proc.stderr)
