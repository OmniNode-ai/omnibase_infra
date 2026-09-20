# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for OMN-18866 (OMN-15547 R1-R5).

THE INCIDENT. On 2026-09-20, run 35488731787 of ``runtime-rebuild-trigger.yml``
(head ``8324ecd3d3001f8f7e90f3a9d074ee61a98e83c7``), the step that derives this
lane's declared writer consumer groups exited 1 in 0.086 seconds having printed
nothing at all -- before Python started. The compose file it was given was
entirely good: it declares six writer groups and still does.

WHY THAT IS A ``false_red``. The guard's correct verdict on that artifact is
ACCEPT -- six groups, exit 0 -- and what the pipeline produced was a FAIL. The
cause was the invocation shape rather than the parsing logic (the step ran
``cd scripts/runtime_build`` and then invoked the deriver by a relative path,
which is the one step in that job not calling ``uv run python`` with a path
from the repository root), but the classification follows the verdict, not the
cause: real good input, guard said fail.

WHY IT MATTERED. The failing step's stdout was captured by a command
substitution, so its failure was silent AND its step output was never set. The
consuming expression delivered an empty string, and the lag probe read that as
"this lane declares no consumer groups" -- a false statement about a lane that
declares six. Every dev sha then received a non-PASS compose-dev receipt, which
under Operating Rule 24(b) refused delivery to staging for changes that had
nothing wrong with them.

WHAT THE REPLAY PROVES, and what it deliberately does not. It drives the REAL
deriver against the REAL captured bytes and asserts the six groups, which
establishes that the derivation was never the defect and that an empty result
would therefore have been a genuine finding rather than noise. It does not
reproduce the CI shim behaviour, which is environmental; the sibling test
``test_the_deriver_is_invoked_from_the_repository_root`` pins the invocation
shape so the environmental half cannot come back.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = (
    _ROOT / "tests" / "fixtures" / "omn18866" / ("docker-compose.dev-lane.yml.captured")
)
_DERIVER = _ROOT / "scripts" / "runtime_build" / "declared_consumer_groups.py"

#: Recorded so a reformatted or regenerated fixture stops being this fixture.
#: A capture that can be edited is a reconstruction, which is the thing
#: OMN-15547 exists to refuse.
_FIXTURE_SHA256 = "41bd1daf4d719bcfb1fe9f8de8ecda54f9a5cd07d8ac8ba0a67d2b255fc56946"

#: The six groups the lane declared at the incident commit, and still declares.
#: Written out rather than counted, because a count would stay green if the
#: derivation started returning six of the wrong names.
_EXPECTED = (
    "local.omnimarket-projections.delegation-writer.consume.v1",
    "local.omnimarket-projections.live-events-writer.consume.v1",
    "local.omnimarket-projections.registration-writer.consume.v1",
    "local.omnimarket-projections.savings-writer.consume.v1",
    "local.omnimarket-projections.tenant-credentials-writer.consume.v1",
    "local.omnimarket-projections.tenant-registry-writer.consume.v1",
)


def test_the_fixture_is_the_captured_artifact() -> None:
    """R2/R3: the bytes are the ones that were fetched, not a look-alike."""
    digest = hashlib.sha256(_FIXTURE.read_bytes()).hexdigest()
    assert digest == _FIXTURE_SHA256, (
        "the captured compose file has changed; re-fetch it from "
        "git-object:OmniNode-ai/omnibase_infra@"
        "8324ecd3d3001f8f7e90f3a9d074ee61a98e83c7:docker/docker-compose.dev-lane.yml "
        "rather than updating this digest to match a modified file"
    )


def test_the_real_guard_accepts_the_real_artifact() -> None:
    """R4: the guard, not a stand-in, against the incident's own bytes.

    This is the discriminating assertion. The pipeline reported FAIL on this
    artifact; the guard's verdict on it is ACCEPT with six named groups.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(_DERIVER),
            "--compose",
            str(_FIXTURE),
            "--env",
            "local",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
    )
    assert result.returncode == 0, (
        "the deriver failed on the real, good compose file from the incident "
        f"commit; stderr: {result.stderr.strip()}"
    )
    derived = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    assert derived == _EXPECTED


def test_an_empty_derivation_is_rejected_rather_than_reported_as_measured() -> None:
    """The other half: an empty result must REFUSE, not answer 'no groups'.

    This is what makes the accept above meaningful. If the deriver returned
    nothing quietly, the probe downstream would report "this lane declares no
    consumer groups" -- which is the exact false sentence the incident put on
    every dev sha. The guard exits non-zero instead, so an empty derivation
    stops at the step that produced it.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(_DERIVER),
            "--compose",
            str(_FIXTURE),
            # A prefix no group in the file carries. The artifact is unchanged;
            # only the question is different.
            "--env",
            "omn18866-no-such-lane",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
    )
    assert result.returncode == 2
    assert "declares no consumer group" in result.stderr
    assert result.stdout.strip() == ""


def test_the_out_file_carries_the_same_answer_as_stdout(tmp_path: Path) -> None:
    """The file is the channel the workflow reads, so it is the one that counts.

    The incident's proximate cause was the answer travelling as a step output,
    where "unset" and "empty" are the same bytes. It travels as a file now, and
    the file must agree with stdout or the two channels are free to disagree.
    """
    out = tmp_path / "declared-groups.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(_DERIVER),
            "--compose",
            str(_FIXTURE),
            "--env",
            "local",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
    )
    assert result.returncode == 0
    from_file = tuple(
        line.strip()
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    from_stdout = tuple(
        line.strip() for line in result.stdout.splitlines() if line.strip()
    )
    assert from_file == from_stdout == _EXPECTED
