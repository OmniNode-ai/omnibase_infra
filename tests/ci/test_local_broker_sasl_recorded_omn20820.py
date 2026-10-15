# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay the local broker's real SASL refusals through the laptop checks (OMN-20820).

The catalog-local CI job and ``make check-sasl-local`` decide "the broker
refuses anonymous clients" by reading what ``rpk`` printed. The recording is
that text, captured from a real broker on a lab host with the
``redpanda-local-sasl`` one-shot applied (provenance in the artifact), so the
match string the Makefile target uses is proven against the broker's own words
and not against a sentence assumed to be one.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]

_DOCKER_STUB = """#!/bin/sh
# Prints the recorded broker text for the call the Makefile target makes.
[ "$1" = "info" ] && exit 0
[ "$1" = "exec" ] || exit 2
shift
while [ "$1" = "-e" ]; do
  case "$2" in
    *=*) export "$2" ;;
    *) export "$2=$(printenv "$2")" ;;
  esac
  shift 2
done
if [ -z "${RPK_USER:-}" ]; then
  cat "$RECORDED_ANONYMOUS" >&2; exit 1
fi
if [ "${RPK_PASS:-}" != "$STUB_PASSWORD" ]; then
  cat "$RECORDED_WRONG_PASSWORD" >&2; exit 1
fi
echo "CLUSTER"; exit 0
"""


def _run_target(
    tmp_path: Path, recorded: dict[str, object], *, password: str
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_DOCKER_STUB, encoding="utf-8")
    docker.chmod(0o755)
    files = {}
    for key in ("anonymous_cluster_info", "wrong_password_cluster_info"):
        entry = recorded[key]
        assert isinstance(entry, dict)
        path = tmp_path / f"{key}.txt"
        path.write_text(str(entry["stderr"]) + "\n", encoding="utf-8")
        files[key] = path
    env_file = tmp_path / "local.env"
    env_file.write_text(f"REDPANDA_SUPERUSER_PASSWORD={password}\n", encoding="utf-8")
    return subprocess.run(
        ["make", "check-sasl-local", f"LOCAL_ENV_FILE={env_file}"],
        cwd=_REPO,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "RECORDED_ANONYMOUS": str(files["anonymous_cluster_info"]),
            "RECORDED_WRONG_PASSWORD": str(files["wrong_password_cluster_info"]),
            "STUB_PASSWORD": "recorded-run-secret",
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/local_broker_sasl_omn20820.json"
)
def test_check_sasl_local_accepts_the_brokers_recorded_anonymous_refusal(
    tmp_path: Path, recorded_response
) -> None:
    result = _run_target(
        tmp_path, recorded_response["response"], password="recorded-run-secret"
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/local_broker_sasl_omn20820.json"
)
def test_check_sasl_local_refuses_the_brokers_recorded_bad_credentials_as_a_failure(
    tmp_path: Path, recorded_response
) -> None:
    """A wrong password is also a refusal, but not for want of SASL."""
    result = _run_target(
        tmp_path, recorded_response["response"], password="some-other-value"
    )
    assert result.returncode != 0
    assert "was refused" in result.stderr


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/local_broker_sasl_omn20820.json"
)
def test_recorded_one_shot_runs_follow_the_order_the_script_enforces(
    recorded_response,
) -> None:
    """Both recorded runs, first boot and a later one, end on the positive control."""
    response = recorded_response["response"]
    first, second = response["one_shot_first_run"], response["one_shot_second_run"]
    markers = [
        "superuser grant confirmed",
        "enable_sasl confirmed true",
        "unauthenticated client refused",
        "authenticated client accepted - positive control",
    ]
    for run in (first, second):
        positions = [
            next(i for i, line in enumerate(run) if m in line) for m in markers
        ]
        assert positions == sorted(positions)
    assert any("created" in line for line in first)
    assert any("already present" in line for line in second)
    script = yaml.safe_load(
        (_REPO / "docker/catalog/services/redpanda-local-sasl.yaml").read_text()
    )["command"][0]
    for marker in markers:
        assert re.search(re.escape(marker), script)
