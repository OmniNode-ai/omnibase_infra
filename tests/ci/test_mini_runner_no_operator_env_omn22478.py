# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The mini host's verify runner is given no operator env file (OMN-22478).

The verify class on the primary host lost the operator env file in omnibase_infra#5709 and
#5720. The mini host's verify runner (`omninode-mini-runner-1`, label `omnibase-verify`) kept
the bind mount, the root-phase copy into its creds volume and the variable that points at the
copy, because it lives in its own compose file. No job on the `omnibase-verify` label reads a
key from that file, so this file gives it none.

An absence passes vacuously when the service is renamed or the parse breaks, so every absence
here is paired with a positive control: the primary host's deploy runner keeps all three, and
the mini runner keeps what it needs to do its job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI_COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners-omninode-mini-runner.yml"
PRIMARY_COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
MINI_SERVICE = "omninode-mini-runner-1"
CONTROL_SERVICE = "omninode-deploy-runner"
OPERATOR_ENV_TARGET = "/run/omnibase-operator.env"
CREDS_COPY = "${RUNNER_USER_HOME:-/home/runner}/.runner-creds/operator.env"


def _service(path: Path, name: str) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    service = loaded["services"][name]
    assert isinstance(service, dict)
    return service


def _entrypoint(service: dict[str, Any]) -> str:
    return " ".join(str(part) for part in service["entrypoint"])


def _code_lines_naming_the_copy(entrypoint: str) -> list[str]:
    return [
        ln.strip()
        for ln in entrypoint.splitlines()
        if "operator.env" in ln and not ln.strip().startswith("#")
    ]


def test_the_mini_runner_has_no_operator_env_mount() -> None:
    volumes = [str(v) for v in _service(MINI_COMPOSE, MINI_SERVICE)["volumes"]]
    assert not [v for v in volumes if OPERATOR_ENV_TARGET in v]
    assert "MINI_RUNNER_OPERATOR_ENV_FILE" not in MINI_COMPOSE.read_text(
        encoding="utf-8"
    )


def test_the_mini_runner_exports_no_operator_env_variable() -> None:
    environment = _service(MINI_COMPOSE, MINI_SERVICE)["environment"]
    assert "OMNIBASE_OPERATOR_ENV_FILE" not in environment


def test_the_mini_root_phase_removes_the_stale_copy_and_installs_none() -> None:
    entrypoint = _entrypoint(_service(MINI_COMPOSE, MINI_SERVICE))
    assert OPERATOR_ENV_TARGET not in entrypoint
    assert _code_lines_naming_the_copy(entrypoint) == [f"rm -f {CREDS_COPY}"]
    code = "\n".join(
        ln for ln in entrypoint.splitlines() if not ln.strip().startswith("#")
    )
    assert "install " not in code
    assert " cp " not in code


def test_the_mini_runner_keeps_what_it_needs() -> None:
    """The parity half: the narrowing must not break the runner's own job."""
    service = _service(MINI_COMPOSE, MINI_SERVICE)
    volumes = [str(v) for v in service["volumes"]]
    assert [v for v in volumes if v.startswith("/var/run/docker.sock:")]
    assert [v for v in volumes if v.endswith("/.lab-credentials:ro")]
    assert [v for v in volumes if v.endswith("/.runner-creds")]
    assert service["environment"]["RUNNER_LABELS"].startswith(
        "self-hosted,omnibase-verify,"
    )
    assert "RUNNER_TOKEN" in service["environment"]


def test_the_deploy_runner_keeps_all_three_as_the_positive_control() -> None:
    service = _service(PRIMARY_COMPOSE, CONTROL_SERVICE)
    volumes = [str(v) for v in service["volumes"]]
    assert len([v for v in volumes if OPERATOR_ENV_TARGET in v]) == 1
    assert "OMNIBASE_OPERATOR_ENV_FILE" in service["environment"]
    assert OPERATOR_ENV_TARGET in _entrypoint(service)
