# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: the dev lane expires consumer-group offsets a day after the last member.

The dev-lane broker held 41,398 Empty consumer groups on 2026-10-16, almost all
one-shot per-run groups. The cluster default keeps a group's offsets for seven days,
so a per-run client that opens a fresh group every call piles up ~40k of them. The dev
lane sets ``group_offset_retention_sec`` through the shared broker-profile one-shot;
every other lane (the base infra file here) renders the same one-shot with the value
empty and leaves the cluster default alone.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.unit.infra.test_broker_profile_omn19419 import _RPK_STUB, _render

_ENV = "REDPANDA_GROUP_OFFSET_RETENTION_SEC"
_SERVICE = "redpanda-partition-cap"


def _run_profile(tmp_path: Path, lane: str) -> tuple[str, str, int]:
    service = _render(lane)["services"][_SERVICE]
    script = service["command"][0].replace("$$", "$").replace("/usr/bin/rpk", "rpk")
    env = {
        "PATH": os.environ["PATH"],
        **service["environment"],
        "RPK_USER": "fixture-user",
        "RPK_PASS": "fixture-password",
        "TMPDIR": str(tmp_path),
        "PROFILE_TRACE": str(tmp_path / "trace"),
        "PROFILE_CHANGED": str(tmp_path / "changed"),
        "PROFILE_CREATED": str(tmp_path / "created"),
        "PROFILE_PIN": str(tmp_path / "pin"),
        "PROFILE_FAILURE": "",
        "PROFILE_SASL": "false",
    }
    result = subprocess.run(
        ["/bin/sh", "-c", _RPK_STUB + script],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return (tmp_path / "trace").read_text(), result.stdout, result.returncode


@pytest.mark.unit
def test_dev_lane_declares_a_one_day_group_offset_retention() -> None:
    env = _render("dev")["services"][_SERVICE]["environment"]
    assert env[_ENV] == "86400"


@pytest.mark.unit
def test_dev_lane_profile_sets_the_retention_and_reads_it_back(tmp_path: Path) -> None:
    trace, stdout, returncode = _run_profile(tmp_path, "dev")
    assert returncode == 0, stdout
    set_at = trace.index("cluster config set group_offset_retention_sec 86400")
    assert trace.index("cluster config get group_offset_retention_sec", set_at) > set_at
    assert "broker profile read back" in stdout


@pytest.mark.unit
def test_a_lane_that_declares_nothing_leaves_the_cluster_default(
    tmp_path: Path,
) -> None:
    env = _render("infra")["services"][_SERVICE]["environment"]
    assert env[_ENV] == ""
    trace, stdout, returncode = _run_profile(tmp_path, "infra")
    assert returncode == 0, stdout
    assert "cluster config set group_offset_retention_sec" not in trace
