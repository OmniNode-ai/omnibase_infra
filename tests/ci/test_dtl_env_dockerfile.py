# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The delegated test loop's two images (OMN-19358).

The loop runs a model-written test in a throwaway container on a lab Mac. Two
images serve it, and both are pinned here as text so a drift is a red test
rather than a surprise on the lab host:

* ``docker/Dockerfile.gate-runner`` takes its base from a ``BASE_IMAGE`` build
  argument. The default is the tag every existing caller already builds, so the
  .201 gate-runner is untouched; the loop passes a digest-pinned base.
* ``docker/Dockerfile.dtl-env`` layers one repository's locked environment on
  top of that test image: only ``pyproject.toml`` and ``uv.lock`` go in, the
  project itself is never installed, and the image runs as the test image's
  non-root user. The test run then puts the task worktree on ``PYTHONPATH``.

The live half of each property (the image builds, imports pytest, cannot import
the project, and rebuilds to the same image id) is exercised on the lab host and
recorded in the PR body; these tests hold the text that makes it true.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATE_RUNNER = _REPO_ROOT / "docker" / "Dockerfile.gate-runner"
_DTL_ENV = _REPO_ROOT / "docker" / "Dockerfile.dtl-env"

#: The base every existing gate-runner build has used. Changing it changes the
#: .201 gate-runner, which this ticket must not do.
_DEFAULT_BASE = "python:3.13-slim-bookworm"


def _instructions(path: Path) -> list[str]:
    """Dockerfile instructions with comments, blanks and continuations folded."""
    lines: list[str] = []
    pending = ""
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        lines.append((pending + stripped).strip())
        pending = ""
    return lines


def test_gate_runner_base_is_a_build_arg_with_unchanged_default() -> None:
    instructions = _instructions(_GATE_RUNNER)
    froms = [i for i in instructions if i.upper().startswith("FROM ")]
    assert froms == ["FROM ${BASE_IMAGE}"], froms
    first_from = instructions.index("FROM ${BASE_IMAGE}")
    base_args = [i for i in instructions[:first_from] if i.startswith("ARG BASE_IMAGE")]
    assert base_args == [f"ARG BASE_IMAGE={_DEFAULT_BASE}"], base_args


def test_dtl_env_builds_from_the_test_image_by_argument() -> None:
    instructions = _instructions(_DTL_ENV)
    froms = [i for i in instructions if i.upper().startswith("FROM ")]
    assert froms == ["FROM ${TEST_IMAGE}"], froms
    first_from = instructions.index("FROM ${TEST_IMAGE}")
    # No default: an env image built on an unnamed base would not be keyed by
    # the test image it claims to extend.
    assert "ARG TEST_IMAGE" in instructions[:first_from]


def test_dtl_env_copies_only_the_lock_inputs() -> None:
    copies = [
        i for i in _instructions(_DTL_ENV) if i.upper().startswith(("COPY ", "ADD "))
    ]
    assert copies, "the env image must copy the lock inputs"
    for instruction in copies:
        assert instruction.startswith("COPY "), instruction
        sources = instruction.split()[1:-1]
        sources = [s for s in sources if not s.startswith("--")]
        assert set(sources) <= {"pyproject.toml", "uv.lock"}, instruction


def test_dtl_env_syncs_the_locked_environment_without_the_project() -> None:
    runs = " ".join(i for i in _instructions(_DTL_ENV) if i.startswith("RUN "))
    assert re.search(r"\buv sync\b", runs), runs
    for flag in ("--frozen", "--no-install-project", "--all-extras"):
        assert flag in runs, f"uv sync must pass {flag}"
    env = " ".join(i for i in _instructions(_DTL_ENV) if i.startswith("ENV "))
    assert "UV_PROJECT_ENVIRONMENT=/opt/dtl-venv" in env, env


def test_dtl_env_runs_as_the_non_root_test_user() -> None:
    users = [i for i in _instructions(_DTL_ENV) if i.startswith("USER ")]
    assert users, "the env image must end on a USER instruction"
    assert users[-1] == "USER 1000:1000", users


def test_dtl_env_carries_no_entrypoint_or_secret() -> None:
    instructions = _instructions(_DTL_ENV)
    assert not [i for i in instructions if i.startswith("ENTRYPOINT")]
    assert not [i for i in instructions if "--mount=type=secret" in i]
