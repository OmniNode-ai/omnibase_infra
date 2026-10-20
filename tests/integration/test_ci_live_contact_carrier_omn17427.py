# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read back local and CI admission across the bus, Git index and clean-root."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.models.model_ci_live_contact_request import (
    ModelCILiveContactRequest,
)
from omnibase_infra.models.model_ci_live_contact_result import ModelCILiveContactResult

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "src/omnibase_infra/nodes/node_ci_live_contact_effect/contract.yaml"
EXCEPTION = (
    "Live-contact exception: This changes a workflow display label; "
    "there is no API or runner behaviour to replay.\n"
)
LOCAL_CARRIER = ".github/live-contact-exception.md"


@pytest.mark.parametrize(
    ("case", "accepted"),
    [
        ("staged", True),
        ("staged-bytes", True),
        ("working-only", False),
        ("unstaged", False),
        ("missing", False),
        ("root-body", False),
        ("previous-commit", False),
        ("ci-without-body", False),
        ("ci-with-body", True),
    ],
)
def test_carrier_admission_persists_bus_verdict(
    tmp_path: Path, case: str, accepted: bool
) -> None:
    """A real terminal readback must agree with index and committed-head controls."""
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(os.environ),
        )

    def commit() -> None:
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "test repository state",
        )

    git("init", "--quiet")
    commit()
    tooling = repository / "scripts/ci/check.py"
    tooling.parent.mkdir(parents=True)
    tooling.write_text("print('gate')\n")
    git("add", "scripts/ci/check.py")

    carrier = repository / ("PR_BODY.md" if case == "root-body" else LOCAL_CARRIER)
    if case != "missing":
        carrier.parent.mkdir(exist_ok=True)
        carrier.write_text(
            "No exception is declared.\n" if case == "working-only" else EXCEPTION
        )
        if case != "unstaged":
            git("add", str(carrier.relative_to(repository)))
        if case == "staged-bytes":
            carrier.write_text("No exception is declared in the working tree.\n")
        elif case == "working-only":
            carrier.write_text(EXCEPTION)

    if case == "previous-commit" or case.startswith("ci-"):
        commit()
        tooling.write_text("print('next gate')\n")
        git("add", "scripts/ci/check.py")
    if case.startswith("ci-"):
        commit()

    request = ModelCILiveContactRequest(
        repository=str(repository),
        base_ref="HEAD^" if case.startswith("ci-") else None,
        pr_body=EXCEPTION if case == "ci-with-body" else "",
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    state_root = tmp_path / "runtime-state"
    runtime = RuntimeLocal(
        workflow_path=CONTRACT,
        state_root=state_root,
        input_path=request_path,
        backend_overrides={"event_bus": "inmemory"},
        timeout=10,
    )
    runtime.run()

    readback = json.loads((state_root / "workflow_result.json").read_text())
    verdict = ModelCILiveContactResult.model_validate(readback["terminal_payload"])
    assert readback["run_id"] == str(runtime.run_id)
    assert runtime.exit_code == readback["exit_code"] == (0 if accepted else 1)
    assert verdict.success is accepted
    assert verdict.changed_tooling == ("scripts/ci/check.py",)
    if accepted:
        assert verdict.reason.startswith("Declared live-contact exception:")
    else:
        assert "requires a live-contact test" in verdict.reason

    clean_root = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validation/validate_clean_root.py"),
            str(repository),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )
    assert clean_root.returncode == (1 if case == "root-body" else 0), (
        clean_root.stdout + clean_root.stderr
    )
    if case == "root-body":
        assert "PR_BODY.md" in clean_root.stdout
