# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit proof for the post-release dev version bump (OMN-13912).

The defect being pinned is not "a helper miscomputes a patch number". It is a
*sequencing* defect in the release train: publishing X.Y.Z from dev HEAD arms
the OMN-13412 release-identity gate against dev itself, and nothing in the train
disarms it. Two measured windows in the current series:

* v0.38.10 tagged 2026-08-26T01:38:23Z at ``5d3f77792``; dev sat at 0.38.10
  until ``a07fefde4`` (OMN-16536, unrelated) bumped it 2026-08-26T03:44:31Z.
* v0.38.11 tagged 2026-08-28T00:49:31Z at ``4529c3486``; dev sat at 0.38.11
  until ``93c42ada4`` (OMN-16769, unrelated) bumped it 2026-08-28T02:27:16Z.

So the properties under test are the ones that make the train disarm its own
gate, and each maps to a leg of that failure:

  * **armed case**   -- dev == published is a BUMP, not a shrug (the incident)
  * **behind case**  -- dev < published is also a BUMP to published+1, never a
                        downgrade to dev+1 (a release cut off-dev must still
                        leave dev ahead)
  * **idempotent**   -- dev already ahead is a NOOP, so a re-dispatched tag or
                        a human who bumped first does not open a second PR
  * **final-only**   -- an rc/pre-release tag is refused, never patch-bumped
  * **narrow write** -- only ``[project].version`` moves; a ``version`` key in
                        any other table is byte-identical afterwards
  * **disarming**    -- the post-bump version actually satisfies the gate's
                        own invariant (strictly greater than published)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.post_release_dev_bump import (
    ACTION_BUMP,
    ACTION_NOOP,
    BumpConfigError,
    apply_decision,
    decide,
    next_patch,
    parse_final_version,
    read_project_version,
    rewrite_project_version,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "post_release_dev_bump.py"

# A pyproject shaped like this repo's: [project].version is the one that moves,
# and there is a decoy `version` under a later table that must not.
PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["hatchling"]

[project]
name = "omnibase-infra"
version = "{version}"
requires-python = ">=3.12"

[tool.some-vendor]
version = "9.9.9"
"""


def _write_pyproject(tmp_path: Path, version: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(PYPROJECT_TEMPLATE.format(version=version), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.38.11", (0, 38, 11)), ("v0.38.11", (0, 38, 11)), (" v1.0.0 ", (1, 0, 0))],
)
def test_parse_accepts_final_versions_with_or_without_v(
    raw: str, expected: tuple[int, int, int]
) -> None:
    assert parse_final_version(raw, label="version") == expected


@pytest.mark.parametrize(
    "raw", ["", "0.38", "0.38.11rc1", "v0.38.11-rc.1", "0.38.11.post1", "latest"]
)
def test_parse_refuses_non_final_versions(raw: str) -> None:
    # An rc must never drive a dev bump: it would consume a real patch number
    # and compare against a version the release-identity gate does not publish.
    with pytest.raises(BumpConfigError):
        parse_final_version(raw, label="released version")


def test_next_patch_increments_only_the_patch_component() -> None:
    assert next_patch("v0.38.11") == "0.38.12"
    assert next_patch("1.2.9") == "1.2.10"


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def test_armed_case_dev_equals_published_is_a_bump() -> None:
    # The exact v0.38.11 incident: tag cut at dev HEAD, dev still says 0.38.11.
    decision = decide(dev_version="0.38.11", released_version="v0.38.11")
    assert decision.action == ACTION_BUMP
    assert decision.target_version == "0.38.12"
    assert "ARMED" in decision.reason


def test_behind_case_bumps_past_published_not_past_dev() -> None:
    # A release cut from somewhere other than dev HEAD leaves dev BEHIND the
    # published version. Bumping dev+1 (0.38.10) would still be <= published and
    # would leave the gate armed; the target must be published+1.
    decision = decide(dev_version="0.38.9", released_version="v0.38.11")
    assert decision.action == ACTION_BUMP
    assert decision.target_version == "0.38.12"


def test_already_ahead_is_a_noop_so_the_step_is_idempotent() -> None:
    # Re-dispatching the same tag, or a human who already bumped, must not open
    # a second bump PR.
    decision = decide(dev_version="0.38.12", released_version="v0.38.11")
    assert decision.action == ACTION_NOOP
    assert decision.target_version == "0.38.12"


def test_decision_target_satisfies_the_release_identity_invariant() -> None:
    # The gate's own rule: dev must be STRICTLY greater than the highest
    # published version. Prove the chosen target actually clears it.
    released = "v0.38.11"
    for dev in ("0.38.9", "0.38.11"):
        decision = decide(dev_version=dev, released_version=released)
        assert parse_final_version(
            decision.target_version, label="target"
        ) > parse_final_version(released, label="released")


# ---------------------------------------------------------------------------
# The write
# ---------------------------------------------------------------------------


def test_rewrite_touches_only_the_project_table(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, "0.38.11")
    updated = rewrite_project_version(path.read_text(encoding="utf-8"), "0.38.12")
    assert 'version = "0.38.12"' in updated
    # The decoy under [tool.some-vendor] is untouched.
    assert 'version = "9.9.9"' in updated
    assert updated.count('version = "0.38.12"') == 1


def test_apply_bumps_the_file_and_noop_leaves_it_byte_identical(
    tmp_path: Path,
) -> None:
    path = _write_pyproject(tmp_path, "0.38.11")
    before = path.read_text(encoding="utf-8")

    bump = decide(read_project_version(path), "v0.38.11")
    assert apply_decision(path, bump) is True
    assert read_project_version(path) == "0.38.12"

    # Second pass over the already-bumped file: noop, and nothing is rewritten.
    after_first = path.read_text(encoding="utf-8")
    noop = decide(read_project_version(path), "v0.38.11")
    assert noop.action == ACTION_NOOP
    assert apply_decision(path, noop) is False
    assert path.read_text(encoding="utf-8") == after_first
    assert after_first != before


def test_rewrite_refuses_a_pyproject_with_no_project_version() -> None:
    with pytest.raises(BumpConfigError):
        rewrite_project_version('[project]\nname = "x"\n', "0.38.12")


def test_read_refuses_a_pyproject_with_no_project_version(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(BumpConfigError):
        read_project_version(path)


# ---------------------------------------------------------------------------
# The CLI the release job actually invokes
# ---------------------------------------------------------------------------


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_emits_a_json_decision_and_applies_the_bump(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, "0.38.11")
    result = _run(["--released", "v0.38.11", "--pyproject", str(path), "--apply"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["action"] == ACTION_BUMP
    assert payload["target_version"] == "0.38.12"
    assert payload["applied"] is True
    assert read_project_version(path) == "0.38.12"


def test_cli_without_apply_decides_but_does_not_write(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, "0.38.11")
    result = _run(["--released", "v0.38.11", "--pyproject", str(path)])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["action"] == ACTION_BUMP
    assert payload["applied"] is False
    assert read_project_version(path) == "0.38.11"


def test_cli_exits_2_on_a_prerelease_tag(tmp_path: Path) -> None:
    path = _write_pyproject(tmp_path, "0.38.11")
    result = _run(["--released", "v0.39.0rc1", "--pyproject", str(path), "--apply"])
    assert result.returncode == 2
    assert "final X.Y.Z" in result.stderr
    assert read_project_version(path) == "0.38.11"
