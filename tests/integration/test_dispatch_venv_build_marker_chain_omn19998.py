# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden and error chains of the dispatch-venv build marker (OMN-19998).

The golden chain promotes the floor of a dispatch venv that has no floor yet:
the pass converges, ``.built-from`` and a copy of the uv.lock land in the venv,
and the core ``ValidatorVenvBuildLock`` judges the venv's installed pins
against that copy without an error — once through the typed
``HandlerVenvAtomicInstall`` and once through the ``promote-floor`` command
line the host reconcile runs. The error chain promotes nothing: a uv.lock that
cannot be read and a venv root that is not a directory both fail the pass and
leave no marker behind, and a moved venv with no site-packages cannot advance
an existing floor, so it fails too and writes no marker.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.validation.validator_venv_build_lock import ValidatorVenvBuildLock
from omnibase_infra.handlers.venv_atomic_install.handler_venv_atomic_install import (
    HandlerVenvAtomicInstall,
)
from omnibase_infra.models.model_venv_atomic_install_request import (
    ModelVenvAtomicInstallRequest,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_venv_atomic_install"
    / "runtime_venv_atomic_install.py"
)

# The reader-side names are literals, as in the seam test: this chain must not
# take them from the engine that writes them.
MARKER_NAME = ".built-from"
BUILD_LOCK_COPY_NAME = ".built-from.uv.lock"
FACTS_NAME = "run.venv-build-lock-facts.json"


def _uv_lock(pins: dict[str, str]) -> str:
    """A minimal uv.lock: one registry-sourced [[package]] block per pin."""
    return "\n".join(
        "[[package]]\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        for name, version in sorted(pins.items())
    )


def read_marker(venv: Path) -> dict[str, str]:
    """Parse the marker's key=value lines the way a reader would."""
    fields: dict[str, str] = {}
    for line in (venv / MARKER_NAME).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key] = value
    return fields


def build_facts(
    tmp_path: Path,
    venv: Path,
    checkout_lock: Path,
    installed: dict[str, str],
) -> Path:
    """Write a facts document whose surface points at the marker's own copy."""
    marker = read_marker(venv)
    facts = tmp_path / FACTS_NAME
    facts.write_text(
        json.dumps(
            {
                "checkout_lock_path": str(checkout_lock),
                "surfaces": [
                    {
                        "label": "dispatch venv",
                        "build_lock_path": str(venv / marker["lock_copy"]),
                        "installed": installed,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return facts


def _error_codes(issues: Any) -> list[str]:
    return [i.code for i in issues if i.severity.value == "error"]


def _make_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    venv.mkdir()
    return venv


def _promote_request(
    venv: Path, lock: Path, workspace_root: Path, floor: Path, *, changed: bool
) -> ModelVenvAtomicInstallRequest:
    return ModelVenvAtomicInstallRequest(
        command="promote_floor",
        venv=venv,
        uv_lock=lock,
        workspace_root=workspace_root,
        floor_path=floor,
        changed=changed,
    )


def test_golden_chain_promote_floor_stamps_a_marker_the_validator_agrees_with(
    tmp_path: Path,
) -> None:
    handler = HandlerVenvAtomicInstall(ModelONEXContainer())
    venv = _make_venv(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text(_uv_lock({"alpha": "1.0.0"}), encoding="utf-8")
    floor = tmp_path / "floor.json"  # deliberately never created

    result = handler.handle(
        _promote_request(venv, lock, tmp_path, floor, changed=False)
    )

    assert result.status == "ok", result
    assert result.exit_code == 0, result
    assert (venv / MARKER_NAME).is_file(), "the marker was not written"
    assert (venv / BUILD_LOCK_COPY_NAME).is_file(), "the lock copy was not written"

    facts = build_facts(tmp_path, venv, lock, {"alpha": "1.0.0"})
    validator = ValidatorVenvBuildLock()
    issues = validator._validate_file(facts, validator._load_contract())

    assert _error_codes(issues) == []


def test_golden_chain_the_host_reconcile_command_line_stamps_the_marker(
    tmp_path: Path,
) -> None:
    venv = _make_venv(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text(_uv_lock({"alpha": "1.0.0"}), encoding="utf-8")
    floor = tmp_path / "floor.json"  # deliberately never created

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNTIME),
            "promote-floor",
            "--venv",
            str(venv),
            "--uv-lock",
            str(lock),
            "--workspace-root",
            str(tmp_path),
            "--floor",
            str(floor),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv / MARKER_NAME).is_file(), "the marker was not written"
    assert (venv / BUILD_LOCK_COPY_NAME).is_file(), "the lock copy was not written"


def test_error_chain_a_uv_lock_that_cannot_be_read_writes_no_marker(
    tmp_path: Path,
) -> None:
    handler = HandlerVenvAtomicInstall(ModelONEXContainer())
    venv = _make_venv(tmp_path)
    lock = tmp_path / "missing.lock"  # deliberately never created
    floor = tmp_path / "floor.json"  # deliberately never created

    result = handler.handle(
        _promote_request(venv, lock, tmp_path, floor, changed=False)
    )

    assert result.status == "failed", result
    assert result.exit_code != 0, result
    assert "FAILED" in result.output, result
    assert not (venv / MARKER_NAME).exists(), "a failed pass wrote the marker"
    assert not (venv / BUILD_LOCK_COPY_NAME).exists(), "a failed pass wrote the copy"


def test_error_chain_a_venv_root_that_is_not_a_directory_creates_nothing(
    tmp_path: Path,
) -> None:
    handler = HandlerVenvAtomicInstall(ModelONEXContainer())
    venv = tmp_path / "never-created"  # deliberately never created
    lock = tmp_path / "uv.lock"
    lock.write_text(_uv_lock({"alpha": "1.0.0"}), encoding="utf-8")
    floor = tmp_path / "floor.json"  # deliberately never created

    result = handler.handle(
        _promote_request(venv, lock, tmp_path, floor, changed=False)
    )

    assert result.status == "failed", result
    assert result.exit_code != 0, result
    assert not venv.exists(), "a failed pass created the venv root"


def test_failing_control_a_moved_venv_without_site_packages_writes_no_marker(
    tmp_path: Path,
) -> None:
    handler = HandlerVenvAtomicInstall(ModelONEXContainer())
    venv = _make_venv(tmp_path)  # no lib/site-packages inside
    lock = tmp_path / "uv.lock"
    lock.write_text(_uv_lock({"alpha": "1.0.0"}), encoding="utf-8")
    floor = tmp_path / "floor.json"
    floor.write_text(json.dumps({"omnimarket_commit": "b" * 40}), encoding="utf-8")

    result = handler.handle(_promote_request(venv, lock, tmp_path, floor, changed=True))

    assert result.status == "failed", result
    assert result.exit_code != 0, result
    assert not (venv / MARKER_NAME).exists(), "marker needs a converged pass"
    assert not (venv / BUILD_LOCK_COPY_NAME).exists(), "copy needs a converged pass"
