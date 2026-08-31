# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reconciler movement-proof gate itself (OMN-17307).

CLAUDE.md rule 5 says a detection tool that is not wired as a pre-merge gate is
advisory and gets ignored. A gate that is wired but does not actually reject
anything is worse — it is advisory with a green checkmark on it. So the tests
below are mostly about what the gate REFUSES, driven against synthetic repo
trees rather than against this repository, so that "the repo happens to be
clean" can never be mistaken for "the gate works".

The last test is the one that runs it against this repo for real.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER = _REPO_ROOT / "scripts" / "check_reconciler_movement_proof.py"
_VERIFIER = _REPO_ROOT / "scripts" / "reconcile_verify_movement.py"
_HOST = _REPO_ROOT / "scripts" / "reconcile-host.sh"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_movement_gate", _CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    return _load()


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A minimal compliant repo: the real orchestrator and the real verifier."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(_HOST, root / "scripts" / "reconcile-host.sh")
    shutil.copy2(_VERIFIER, root / "scripts" / "reconcile_verify_movement.py")
    return root


def test_a_compliant_tree_passes(gate: ModuleType, fake_repo: Path) -> None:
    assert gate.check(fake_repo) == []


# --------------------------------------------------------------------------- #
# Part 1 -- structural, cannot be satisfied by a comment
# --------------------------------------------------------------------------- #
def test_orchestrator_that_stops_calling_the_verifier_fails(
    gate: ModuleType, fake_repo: Path
) -> None:
    host = fake_repo / "scripts" / "reconcile-host.sh"
    host.write_text(
        host.read_text(encoding="utf-8").replace(
            "reconcile_verify_movement.py", "true"
        ),
        encoding="utf-8",
    )
    failures = gate.check(fake_repo)
    assert any("no longer invokes" in f for f in failures)


def test_verdict_signature_growing_an_exit_status_parameter_fails(
    gate: ModuleType, fake_repo: Path
) -> None:
    """The single change that would quietly re-open the whole defect class.

    ``verdict(before, after, target, exit_status)`` would let any caller pass
    "the command succeeded" and get a pass, which is exactly the pre-OMN-17307
    behaviour with a nicer signature.
    """
    verifier = fake_repo / "scripts" / "reconcile_verify_movement.py"
    verifier.write_text(
        verifier.read_text(encoding="utf-8").replace(
            "def verdict(before: str | None, after: str | None, target: str | None) -> Verdict:",
            "def verdict(before: str | None, after: str | None, target: str | None, exit_status: int = 0) -> Verdict:",
        ),
        encoding="utf-8",
    )
    failures = gate.check(fake_repo)
    assert any("must take exactly" in f for f in failures)


def test_missing_orchestrator_fails(gate: ModuleType, fake_repo: Path) -> None:
    (fake_repo / "scripts" / "reconcile-host.sh").unlink()
    assert any("is missing" in f for f in gate.check(fake_repo))


# --------------------------------------------------------------------------- #
# Part 2 -- the declaration ratchet
# --------------------------------------------------------------------------- #
def test_a_new_reconciler_with_no_proof_is_rejected(
    gate: ModuleType, fake_repo: Path
) -> None:
    """Discovery is by glob, so a new reconciler cannot arrive unnoticed."""
    (fake_repo / "scripts" / "reconcile-something-new.sh").write_text(
        "#!/usr/bin/env bash\nuv sync --frozen\nexit 0\n", encoding="utf-8"
    )
    failures = gate.check(fake_repo)
    assert any("reconcile-something-new.sh" in f for f in failures)
    assert any("movement-proof-delegated-to" in f for f in failures)


def test_a_reconciler_nested_under_scripts_is_still_discovered(
    gate: ModuleType, fake_repo: Path
) -> None:
    nested = fake_repo / "scripts" / "runtime_build"
    nested.mkdir(parents=True)
    (nested / "reconcile_deploy_clones.sh").write_text(
        "#!/usr/bin/env bash\ngit pull --ff-only\n", encoding="utf-8"
    )
    assert any("reconcile_deploy_clones.sh" in f for f in gate.check(fake_repo))


def test_a_declared_self_proof_is_accepted(gate: ModuleType, fake_repo: Path) -> None:
    (fake_repo / "scripts" / "reconcile-something-new.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# movement-proof: re-reads HEAD after checkout and asserts it equals the fetched tip\n"
        "git pull --ff-only\n",
        encoding="utf-8",
    )
    assert gate.check(fake_repo) == []


def test_delegation_to_a_real_file_is_accepted(
    gate: ModuleType, fake_repo: Path
) -> None:
    (fake_repo / "scripts" / "reconcile-delegate.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# movement-proof-delegated-to: scripts/reconcile-host.sh\n",
        encoding="utf-8",
    )
    assert gate.check(fake_repo) == []


def test_delegation_to_a_nonexistent_file_is_rejected(
    gate: ModuleType, fake_repo: Path
) -> None:
    """A pointer to nothing is an unproven surface with a comment on it."""
    (fake_repo / "scripts" / "reconcile-delegate.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# movement-proof-delegated-to: scripts/does-not-exist.sh\n",
        encoding="utf-8",
    )
    failures = gate.check(fake_repo)
    assert any("does not exist" in f for f in failures)


# --------------------------------------------------------------------------- #
# Wiring -- the gate must actually be a gate
# --------------------------------------------------------------------------- #
def test_gate_is_wired_into_pre_commit() -> None:
    config = (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "check-reconciler-movement-proof" in config
    assert "scripts/check_reconciler_movement_proof.py" in config


def test_gate_is_wired_into_ci() -> None:
    workflow = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "reconciler-movement-proof:" in workflow
    assert "scripts/check_reconciler_movement_proof.py" in workflow


def test_gate_needs_no_venv_to_run() -> None:
    """It guards the surface that matters when the venv is broken.

    A gate that needs ``uv sync`` to answer cannot answer during the failure it
    exists for, so it is stdlib-only and this asserts it stays that way.
    """
    proc = subprocess.run(
        ["python3", "-I", str(_CHECKER), "--repo-root", str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_this_repository_currently_passes() -> None:
    proc = subprocess.run(
        ["python3", str(_CHECKER), "--repo-root", str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
# Incident replay (OMN-15547) -- the real bytes that shipped
# --------------------------------------------------------------------------- #
_REPLAY_FIXTURE = (
    _REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn17307"
    / "reconcile-workspace-venvs.no-movement-proof.sh.captured"
)


def test_the_shipped_reconciler_that_judged_by_exit_code_is_rejected(
    gate: ModuleType, fake_repo: Path
) -> None:
    """Drive the gate over the verbatim reconciler that shipped on dev.

    Not a synthetic "a reconciler with no marker" string -- the actual blob at
    ``196cdaef``, the wired repair path for every local venv at the time. Its
    repair function ends with ``exit 0`` on the exit status of
    ``install-node-skill-package.sh`` and ``uv sync``, and never re-reads the
    venv. That is the artifact, and ``reject`` is the verdict that did not exist
    when it was reviewed, merged and scheduled.

    Registry: ``omn17307-reconciler-judged-by-exit-code``.
    """
    captured = _REPLAY_FIXTURE.read_bytes()
    target = fake_repo / "scripts" / "reconcile-workspace-venvs.sh"
    target.write_bytes(captured)

    failures = gate.check(fake_repo)

    assert any("reconcile-workspace-venvs.sh" in f for f in failures), (
        "the gate accepted the exact bytes that shipped without any movement "
        "proof; that is the false-green this case exists to make impossible"
    )
    # And the refusal has to say what to do about it, or it is a dead end.
    assert any("movement-proof-delegated-to" in f for f in failures)


def test_the_replay_fixture_is_the_bytes_the_registry_pins(gate: ModuleType) -> None:
    """R1: an edited capture is no longer the artifact that failed."""
    import hashlib

    import yaml

    registry = yaml.safe_load(
        (_REPO_ROOT / "tests" / "incident_replays" / "registry.yaml").read_text(
            encoding="utf-8"
        )
    )
    case = next(
        c
        for c in registry["cases"]
        if c["id"] == "omn17307-reconciler-judged-by-exit-code"
    )
    digest = hashlib.sha256(_REPLAY_FIXTURE.read_bytes()).hexdigest()
    assert digest == case["artifact"]["sha256"]
