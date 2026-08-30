# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the OMN-13412 fresh-deploy fitness gate scanners.

Covers the three NEW validators wired in this PR:
  * scripts/check_terminal_cost_completeness.py  (item 5)
  * scripts/check_context_field_presence.py      (item 6)
  * scripts/check_release_identity.py            (item 7)

Each test proves a deliberately-broken input fails (non-zero exit) and a
correct input passes (exit 0) — the DoD requirement for an enforcement gate.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.version import InvalidVersion, Version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


# --------------------------------------------------------------------------- #
# Item 5: terminal cost completeness                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_terminal_cost_detects_hardcoded_zero(tmp_path: Path) -> None:
    bad = tmp_path / "bad_terminal.py"
    bad.write_text(
        "def emit():\n"
        "    return ModelLlmCallCompleted(\n"
        "        tokens_used=1234,\n"
        "        cost_usd=0.0,\n"
        "    )\n",
        encoding="utf-8",
    )
    result = _run("check_terminal_cost_completeness.py", str(bad))
    assert result.returncode == 1, result.stderr
    assert "cost_usd=0.0" in result.stderr


@pytest.mark.unit
def test_terminal_cost_allows_annotated_zero(tmp_path: Path) -> None:
    ok = tmp_path / "ok_terminal.py"
    ok.write_text(
        "def emit():\n"
        "    return ModelLlmCallCompleted(\n"
        "        tokens_used=0,\n"
        "        cost_usd=0.0,  # cost-zero-ok: error path, no tokens consumed\n"
        "    )\n",
        encoding="utf-8",
    )
    result = _run("check_terminal_cost_completeness.py", str(ok))
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_terminal_cost_ignores_real_value_and_substring(tmp_path: Path) -> None:
    ok = tmp_path / "real_cost.py"
    ok.write_text(
        "def emit():\n"
        "    return Model(\n"
        "        cost_usd=self._estimate_cost(tokens),\n"
        "        estimated_cost_usd=0.0,\n"  # different field — must not match
        "    )\n",
        encoding="utf-8",
    )
    result = _run("check_terminal_cost_completeness.py", str(ok))
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# Item 6: context field presence                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_context_field_detects_claim_without_hash(tmp_path: Path) -> None:
    bad = tmp_path / "contract.yaml"
    bad.write_text(
        "name: node_demo\n"
        "node_type: COMPUTE_GENERIC\n"
        "metadata:\n"
        "  context_roi:\n"
        "    tokens_saved: 4096\n",
        encoding="utf-8",
    )
    result = _run("check_context_field_presence.py", str(bad))
    assert result.returncode == 1, result.stderr
    assert "context_pack_hash" in result.stderr


@pytest.mark.unit
def test_context_field_passes_claim_with_hash(tmp_path: Path) -> None:
    ok = tmp_path / "contract.yaml"
    ok.write_text(
        "name: node_demo\n"
        "node_type: COMPUTE_GENERIC\n"
        "metadata:\n"
        "  context_roi:\n"
        "    tokens_saved: 4096\n"
        "    context_pack_hash: sha256:abc123\n",
        encoding="utf-8",
    )
    result = _run("check_context_field_presence.py", str(ok))
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_context_field_passes_no_claim(tmp_path: Path) -> None:
    ok = tmp_path / "contract.yaml"
    ok.write_text(
        "name: node_demo\nnode_type: COMPUTE_GENERIC\n",
        encoding="utf-8",
    )
    result = _run("check_context_field_presence.py", str(ok))
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# Item 7: release identity                                                    #
# --------------------------------------------------------------------------- #


def _repo_is_version_ahead() -> tuple[bool, str]:
    """Is this checkout's ``project.version`` strictly ahead of its newest tag?

    Both facts are AMBIENT: the version is whatever the branch carries, and the
    tag set is whatever the local clone has fetched. The test below cannot create
    either, so it reports the precondition instead of failing on it.
    """
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as fh:
        version = tomllib.load(fh)["project"]["version"]
    tags = subprocess.run(
        ["git", "tag", "--list"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    parsed = []
    for tag in tags:
        try:
            parsed.append(Version(tag.lstrip("v")))
        except InvalidVersion:
            continue
    if not parsed:
        return True, "no published tags in this clone -- the gate is exempt"
    newest = max(parsed)
    if Version(version) > newest:
        return True, f"{version} > {newest}"
    return False, f"pyproject {version} is not ahead of newest tag {newest}"


@pytest.mark.unit
def test_release_identity_passes_when_version_ahead() -> None:
    """The strict (no ``--base``) run must pass when the version IS ahead.

    OMN-16989: this used to assert the precondition rather than state it, so
    between a release tag landing and the post-release bump landing (OMN-13912)
    it went red on EVERY developer clone -- and only on a developer clone.
    ``actions/checkout`` does not fetch tags by default, so in CI
    ``git tag --list`` is empty, the gate is exempt, and the test passes; locally
    the tags exist and it fails. A red visible only on the machine running the
    governed pre-push hook, saying nothing about the diff, blocked every local
    push in the repo. Measured 2026-08-30: dev at 0.38.14 with `v0.38.14` cut.

    The gate itself is NOT relaxed -- `check-release-identity` (pre-commit) and
    the CI gate both still run the real script against the real diff. What
    changes is that this test now names the ambient precondition it needs
    instead of reporting its absence as a failure of the script."""
    ahead, why = _repo_is_version_ahead()
    if not ahead:
        pytest.skip(
            f"repo-state precondition absent: {why}. This asserts the "
            "version-ahead branch of the gate and cannot create that state; "
            "the post-release bump (OMN-13912) restores it."
        )
    result = _run("check_release_identity.py")
    assert result.returncode == 0, result.stderr
    assert "ahead of latest published" in result.stdout


@pytest.mark.unit
def test_release_identity_exempts_non_src_diff() -> None:
    # A changed-file list with no src/** entry is exempt regardless of version.
    result = _run(
        "check_release_identity.py",
        "--changed-file",
        "docs/readme.md",
        "--changed-file",
        ".github/workflows/ci.yml",
    )
    assert result.returncode == 0, result.stderr
    assert "version bump not required" in result.stdout
