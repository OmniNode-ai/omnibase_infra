# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""node_verify_stability_refresh reads the stability broker as the gate's principal (OMN-25927)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_verify_stability_refresh import (
    GATE_PASSWORD_VAR,
    GATE_PRINCIPAL,
    HandlerVerifyStabilityRefresh,
    rpk_exec_argv,
)
from omnibase_infra.models.model_verify_stability_refresh_rpk_request import (
    ModelVerifyStabilityRefreshRpkRequest,
)

pytestmark = pytest.mark.unit

NODE_DIR = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_verify_stability_refresh"
)
# A sentinel that only proves the value never reaches argv; not a credential.
SENTINEL = "omn25927-node-sentinel-not-a-secret"
LOGIN = [
    "-e",
    f"RPK_USER={GATE_PRINCIPAL}",
    "-e",
    "RPK_SASL_MECHANISM=SCRAM-SHA-256",
    "-e",
    "RPK_PASS",
]


def test_contract_binds_the_handler() -> None:
    contract = yaml.safe_load((NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    assert contract["name"] == "node_verify_stability_refresh"
    (route,) = contract["handler_routing"]["handlers"]
    assert route["handler"] == {
        "name": "HandlerVerifyStabilityRefresh",
        "module": "omnibase_infra.handlers.handler_verify_stability_refresh",
    }


def test_argv_carries_the_login_names_and_never_the_password() -> None:
    environ = {GATE_PASSWORD_VAR: SENTINEL}
    argv = rpk_exec_argv("broker", "topic", "list", environ=environ)
    assert argv == ["docker", "exec", *LOGIN, "broker", "rpk", "topic", "list"]
    assert SENTINEL not in " ".join(argv)
    assert environ["RPK_PASS"] == SENTINEL


def test_argv_stays_anonymous_without_a_password() -> None:
    environ: dict[str, str] = {}
    argv = rpk_exec_argv("broker", "cluster", "health", environ=environ)
    assert argv == ["docker", "exec", "broker", "rpk", "cluster", "health"]
    assert "RPK_PASS" not in environ


def test_handle_runs_one_authenticated_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GATE_PASSWORD_VAR, SENTINEL)
    monkeypatch.delenv("RPK_PASS", raising=False)
    seen: list[list[str]] = []

    def runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="NAME\nt\n", stderr="")

    monkeypatch.setattr(subprocess, "run", runner)

    request = ModelVerifyStabilityRefreshRpkRequest(
        broker_container="broker", rpk_args=("topic", "list")
    )
    result = HandlerVerifyStabilityRefresh().handle(request)
    assert (result.exit_code, result.authenticated, result.stdout) == (
        0,
        True,
        "NAME\nt\n",
    )
    assert seen == [["docker", "exec", *LOGIN, "broker", "rpk", "topic", "list"]]


def test_handle_reports_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GATE_PASSWORD_VAR, raising=False)

    def runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(subprocess, "run", runner)

    request = ModelVerifyStabilityRefreshRpkRequest(
        broker_container="broker", rpk_args=("group", "list")
    )
    result = HandlerVerifyStabilityRefresh().handle(request)
    assert (result.exit_code, result.authenticated) == (124, False)
