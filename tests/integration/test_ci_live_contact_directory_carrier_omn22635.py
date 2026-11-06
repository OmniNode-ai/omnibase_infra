# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22635: a staged carrier file under the directory is admitted across the bus.

The node runs through RuntimeLocal on an in-memory bus over a scratch repository
with a staged change to CI tooling, and the verdict is read back from the
terminal result file, as for the shared carrier in
``test_ci_live_contact_carrier_omn17427.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
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
CARRIER_DIR = ".github/live-contact-exception"


@pytest.mark.parametrize(
    ("case", "accepted"),
    [
        ("staged-directory-carrier", True),
        ("unstaged-directory-carrier", False),
        ("short-directory-carrier", False),
        ("file-outside-the-directory", False),
        ("no-carrier", False),
    ],
)
def test_directory_carrier_admission_persists_bus_verdict(
    tmp_path: Path, case: str, accepted: bool
) -> None:
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

    git("init", "--quiet")
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
    tooling = repository / "scripts/ci/check.py"
    tooling.parent.mkdir(parents=True)
    tooling.write_text("print('gate')\n")
    git("add", "scripts/ci/check.py")

    if case != "no-carrier":
        directory = (
            f"{CARRIER_DIR}-extra"
            if case == "file-outside-the-directory"
            else CARRIER_DIR
        )
        carrier = repository / directory / "OMN-22635.md"
        carrier.parent.mkdir(parents=True)
        carrier.write_text(
            "Live-contact exception: impossible\n"
            if case == "short-directory-carrier"
            else EXCEPTION
        )
        if case != "unstaged-directory-carrier":
            git("add", str(carrier.relative_to(repository)))

    request_path = tmp_path / "request.json"
    request_path.write_text(
        ModelCILiveContactRequest(repository=str(repository)).model_dump_json()
    )
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
    assert runtime.exit_code == readback["exit_code"] == (0 if accepted else 1)
    assert verdict.success is accepted
    assert verdict.changed_tooling == ("scripts/ci/check.py",)
    if accepted:
        assert verdict.reason.startswith("Declared live-contact exception:")
    else:
        assert "requires a live-contact test" in verdict.reason
