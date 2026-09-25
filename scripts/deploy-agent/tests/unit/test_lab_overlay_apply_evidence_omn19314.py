# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19314 -- a failed onex-lab apply records why it failed.

omnibase_infra run 35849365129 failed its k3s onex-lab verify job three times
on one record whose ``lab_overlay_applied`` evidence was the first 240
characters of ``apply_lab_lane.sh``'s stdout -- the banner of namespace and
image pins -- under the label "tail". The FATAL line the script printed was on
stderr, which ``stdout or stderr`` never reads, so the reason was lost.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .test_lab_overlay_omn18200 import SHA, STAMP, FakeRunner, _applier

pytestmark = pytest.mark.unit

# Long enough that the head and the tail cannot both fit in the evidence.
BANNER = "== onex-lab apply == lab-tier: shared namespace: onex-dev " + (
    "runtime image: onex-lab/omninode-runtime:pin " * 40
)
FATAL = "FATAL: 1 Deployment(s) did not become ready: omninode-runtime-effects"


@pytest.fixture
def overlay_source(tmp_path: Path) -> Path:
    """A stand-in omninode_infra clone, the same shape the apply tests use."""
    source = tmp_path / "omninode_infra"
    (source / "k8s" / "onex-lab").mkdir(parents=True)
    (source / "k8s" / "onex-lab" / "apply_lab_lane.sh").write_text("#!/bin/sh\n")
    return source


class StreamRunner(FakeRunner):
    """FakeRunner whose apply_lab_lane.sh answer carries both streams."""

    def __init__(self, code: int, stdout: str, stderr: str) -> None:
        super().__init__()
        self.apply_answer = (code, stdout, stderr)

    def __call__(self, argv: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        argv = list(argv)
        if "apply_lab_lane.sh" in " ".join(argv):
            self.calls.append(argv)
            code, out, err = self.apply_answer
            return subprocess.CompletedProcess(argv, code, out, err)
        return super().__call__(argv, **kwargs)


def _applied_evidence(tmp_path: Path, source: Path, runner: FakeRunner) -> str:
    path = _applier(tmp_path, source, runner).apply(
        sha=SHA, stamp=STAMP, correlation_id="cid-19314"
    )
    record = json.loads(path.read_text())
    applied = next(c for c in record["checks"] if c["name"] == "lab_overlay_applied")
    return str(applied["evidence"])


def test_a_failed_apply_records_the_end_of_stderr(
    tmp_path: Path, overlay_source: Path
) -> None:
    runner = StreamRunner(1, BANNER, "NOT READY: omninode-runtime-effects\n" + FATAL)

    evidence = _applied_evidence(tmp_path, overlay_source, runner)

    assert "exited 1" in evidence
    assert "stderr tail:" in evidence
    assert FATAL in evidence
    # The banner is what the record carried before; it must not crowd out the
    # reason.
    assert "lab-tier: shared" not in evidence


def test_a_failed_apply_with_a_long_stderr_keeps_its_last_line(
    tmp_path: Path, overlay_source: Path
) -> None:
    noise = "error: timed out waiting for the condition on deployments " * 30
    runner = StreamRunner(1, BANNER, noise + FATAL)

    evidence = _applied_evidence(tmp_path, overlay_source, runner)

    assert evidence.endswith(FATAL), evidence
    assert "stderr tail: ..." in evidence


def test_without_stderr_the_evidence_is_the_end_of_stdout_not_its_head(
    tmp_path: Path, overlay_source: Path
) -> None:
    runner = StreamRunner(1, BANNER + " last stdout line", "")

    evidence = _applied_evidence(tmp_path, overlay_source, runner)

    assert "stdout tail:" in evidence
    assert evidence.endswith("last stdout line"), evidence
    assert "lab-tier: shared" not in evidence


def test_a_passing_apply_quotes_the_end_of_stdout(
    tmp_path: Path, overlay_source: Path
) -> None:
    runner = StreamRunner(0, BANNER + " ready: omninode-runtime", "warning: noise")

    evidence = _applied_evidence(tmp_path, overlay_source, runner)

    assert "exited 0" in evidence
    assert "stdout tail:" in evidence
    assert evidence.endswith("ready: omninode-runtime"), evidence
