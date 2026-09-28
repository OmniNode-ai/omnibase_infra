# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for the lab proof profile registry check (OMN-19565).

THE INCIDENT. Until the registry, the only per-repository declaration of lab
evidence was ``config/release_train_policy.yaml``. It covers seven repositories,
and for five of them it declares ``lab_evidence: none``: omnibase_core,
omnibase_spi, omnimemory, omniintelligence and omniclaude cannot produce a lab
receipt for any sha, and the file says so in its own header. Nothing refused
that. A pull request in any of those repositories merged with no lab proof of
any kind, and every check that read the file was green. The operator ruled on
2026-09-25T13:07:40Z (omni_home rolling ledger, lane merge-drain-83) that every
code PR gets a lab proof before merge; the registry and its check exist so that
a repository with no proof path is a refusal, not a quiet ``none``.

THE ARTIFACT is ``config/release_train_policy.yaml`` exactly as it stood on
omnibase_infra ``dev`` at c7150a15a, read with ``git show`` and committed
unmodified. The replay builds the lab proof registry that those bytes describe
-- a proof row only for the repositories whose captured ``lab_evidence`` is not
``none`` -- and drives the real check over it. It must refuse, naming each
repository the captured file left without lab evidence. The discriminator
drives the same check over the committed registry, which carries a row for
every repository, and requires it to accept, so a check that refused everything
cannot pass.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/omn19565/release_train_policy.yaml.captured"
FIXTURE_SHA256 = "ffa119e3655fb832ecd260043253887ec7f8b924e42cc7e91b2d9b5feaf4280d"
GUARD = REPO_ROOT / "scripts/ci/validate_lab_proof_profiles.py"
REGISTRY = REPO_ROOT / "config/lab_proof_profiles.yaml"

pytestmark = pytest.mark.unit


def _captured_lab_evidence() -> dict[str, str]:
    policy = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    return {
        f"OmniNode-ai/{name}": str(entry["lab_evidence"])
        for name, entry in policy["repos"].items()
    }


def _registry() -> dict[str, Any]:
    loaded = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _run(registry: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GUARD), "--registry", str(registry)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_fixture_is_the_captured_bytes() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_the_captured_file_left_foundation_repos_without_lab_evidence() -> None:
    evidence = _captured_lab_evidence()
    none = sorted(repo for repo, value in evidence.items() if value == "none")
    assert "OmniNode-ai/omnibase_core" in none
    assert "OmniNode-ai/omnibase_spi" in none
    proved = sorted(repo for repo, value in evidence.items() if value != "none")
    assert proved == ["OmniNode-ai/omnibase_infra", "OmniNode-ai/omnimarket"]


def test_the_real_guard_refuses_the_registry_the_captured_file_describes(
    tmp_path: Path,
) -> None:
    evidence = _captured_lab_evidence()
    proved = {repo for repo, value in evidence.items() if value != "none"}
    registry = _registry()
    registry["profiles"] = [
        row for row in registry["profiles"] if row["repo"] in proved
    ]
    path = tmp_path / "lab_proof_profiles.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    result = _run(path)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    for repo, value in evidence.items():
        if value == "none":
            assert repo in output, f"{repo} was left without lab evidence and not named"


def test_the_same_guard_accepts_the_committed_registry() -> None:
    result = _run(REGISTRY)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = _captured_lab_evidence()
    repos = {row["repo"] for row in _registry()["profiles"]}
    assert set(evidence) <= repos
