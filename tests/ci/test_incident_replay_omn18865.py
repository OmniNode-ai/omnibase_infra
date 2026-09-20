# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay (OMN-15547) for the pre-merge wheel content-parity guard.

THE INCIDENT. On 2026-09-19 at 21:19Z the .201 dev lane stopped being able to
rebuild. Every ``BUILD_SOURCE=workspace`` image build died in the OMN-14631
content-parity gate with::

    INSTALLED CONTENT DRIFT for 'omnimarket' (OMN-14631): ...
    differing files (1 total): ['adapters/codex/skills/merge-sweep/SKILL.md']

omnimarket#2670 (squash ``61187c8c``) had adopted the propagated
``public_repo_hygiene`` block, which declared a BARE ``merge-sweep/``. A bare
directory pattern matches at ANY depth, so it also matched this repository's
real, git-tracked package directory
``src/omnimarket/adapters/codex/skills/merge-sweep/``. hatchling applies the
VCS ignore file as a build-time exclude, so the wheel silently lost a tracked
source file while the staged source kept it. omnimarket#2694 (``823bae76``)
anchored the pattern to ``/merge-sweep/``.

THE FALSE GREEN BEING REPLACED is an ABSENCE. omnimarket's own CI was fully
green on #2670 -- nothing in that repository looked at what its wheel
contained, so the defect was undetectable there and was discovered only on
the lab, after merge, as somebody else's broken rebuild. This is the second
occurrence of the class: omnibase_compat's bare ``env/`` did the same thing
to ``src/omnibase_compat/env/`` under OMN-14636.

THE ARTIFACT AND THE DISCRIMINATOR DIFFER BY ONE CHARACTER, which is the
whole point and the reason both are captured rather than one being described.
The artifact is omnimarket's ``.gitignore`` at the #2670 merge commit, line 60
``merge-sweep/``; the discriminator is the same file at the #2694 merge
commit, line 60 ``/merge-sweep/``. A guard that cannot tell those apart is
not enforcing anything.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "ci" / "check_wheel_content_parity.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn18865"

ARTIFACT = FIXTURES / "omnimarket-2670.gitignore.captured"
ARTIFACT_SHA256 = "ea49ceb0b1b707225e6c7ee3c78f3318b74985f30415f31e35fe1b0d1fad9593"
DISCRIMINATOR = FIXTURES / "omnimarket-2694.gitignore.captured"
DISCRIMINATOR_SHA256 = (
    "ca6845a76345ad6cd061f703270fc95cccf5c08de7968d46657704e02b3aaf1b"
)

# The exact path the image gate named in its refusal, reproduced verbatim.
DROPPED_FILE = "src/omnimarket/adapters/codex/skills/merge-sweep/SKILL.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_root(gitignore_bytes: bytes) -> Path:
    """Return a build root this repository's OWN guard accepts for these bytes.

    NOT a convenience. hatchling skips the VCS ignore file entirely when the
    build root itself resolves as ignored, and omnimarket's captured
    ``.gitignore`` carries a bare ``tmp``-shaped pattern -- so building this
    replay under ``/tmp/pytest-...``, which is exactly where a Linux CI
    runner puts ``tmp_path``, DISARMS the pattern under test and the replay
    passes while proving nothing. That is the precise false-green shape
    OMN-15547 exists to refuse, reproduced inside the replay itself.

    So the location is chosen by the guard rather than assumed, and if no
    candidate is acceptable this raises instead of skipping: a replay that
    cannot run has not passed.
    """
    sys.path.insert(0, str(GUARD.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("parity_guard_for_replay", GUARD)
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    stamp = uuid.uuid4().hex[:12]
    candidates = [
        Path.home() / ".cache" / "onex-incident-replay" / f"omn18865-{stamp}",
        REPO_ROOT / f".omn18865-replay-{stamp}",
        Path("/") / "var" / "tmp" / f"omn18865-{stamp}",
    ]
    refusals: list[str] = []
    for candidate in candidates:
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / ".gitignore").write_bytes(gitignore_bytes)
        errors = guard.assert_build_root_is_not_vcs_ignored(candidate)
        if not errors:
            return candidate
        refusals.append(f"{candidate}: {errors[0][:160]}")
        shutil.rmtree(candidate, ignore_errors=True)

    raise AssertionError(
        "No candidate build root survives the captured .gitignore, so this "
        "replay cannot be run honestly. Every location tried would have "
        "disarmed the very pattern under test and produced a green that "
        "proves nothing:\n  " + "\n  ".join(refusals)
    )


def _lay_down_project(root: Path, gitignore: Path) -> None:
    """Reproduce omnimarket's packaging shape around the captured ignore file."""
    (root / ".gitignore").write_bytes(gitignore.read_bytes())
    (root / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n'
        "\n"
        "[project]\n"
        'name = "omnimarket"\n'
        'version = "0.4.134"\n'
        "\n"
        "[tool.hatch.build.targets.wheel]\n"
        'packages = ["src/omnimarket"]\n'
        "\n"
        "[tool.hatch.build]\n"
        'artifacts = ["src/omnimarket/**/*.yaml", "src/omnimarket/**/*.sql"]\n'
    )
    pkg = root / "src" / "omnimarket"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    dropped = root / DROPPED_FILE
    dropped.parent.mkdir(parents=True, exist_ok=True)
    dropped.write_text("# merge sweep skill\n")


def _run_guard(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--repo-root",
            str(root),
            "--package",
            "omnimarket",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_captured_artifacts_are_the_bytes_they_claim_to_be() -> None:
    """A fixture whose digest has moved is no longer the artifact that failed."""
    assert _sha256(ARTIFACT) == ARTIFACT_SHA256
    assert _sha256(DISCRIMINATOR) == DISCRIMINATOR_SHA256
    # The one-character difference the whole incident turns on.
    assert "\nmerge-sweep/\n" in ARTIFACT.read_text()
    assert "\n/merge-sweep/\n" in DISCRIMINATOR.read_text()


@pytest.mark.slow
def test_the_real_guard_rejects_the_real_incident_state() -> None:
    """The guard must refuse the tree omnimarket#2670 actually merged."""
    root = _build_root(ARTIFACT.read_bytes())
    try:
        _lay_down_project(root, ARTIFACT)
        result = _run_guard(root)
        assert result.returncode == 1, (
            "the guard did not refuse the tree that took the dev lane down:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        # The same single path the image gate named at 21:19Z.
        assert "adapters/codex/skills/merge-sweep/SKILL.md" in result.stderr
        assert "DROPPED" in result.stderr
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.slow
def test_the_repaired_gitignore_is_accepted() -> None:
    """The discriminator: anchoring the pattern must clear the refusal.

    Mandatory, not a formality. A guard that refused BOTH trees would look
    identical to a working one on the artifact alone, and would red every
    omnimarket pull request forever. The fix is one leading slash; the guard
    has to see it.
    """
    root = _build_root(DISCRIMINATOR.read_bytes())
    try:
        _lay_down_project(root, DISCRIMINATOR)
        result = _run_guard(root)
        assert result.returncode == 0, (
            "the guard refused the REPAIRED tree, which would wedge every "
            f"pull request in that repository:\n{result.stdout}\n{result.stderr}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
