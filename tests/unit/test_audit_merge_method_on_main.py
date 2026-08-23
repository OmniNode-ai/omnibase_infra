# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for scripts/audit-merge-method-on-main.sh.

OMN-8843 — verifies the post-merge bypass detector exits non-zero when
2-parent merge commits appear on main and zero when only single-parent
(squash) commits are present. Mocks `gh` via a tempdir-shimmed binary
so no real GitHub API calls are made.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit-merge-method-on-main.sh"


def _write_gh_shim(tmpdir: Path, response_json: str) -> Path:
    """Create a fake `gh` executable that prints *response_json* and exits 0."""
    shim = tmpdir / "gh"
    # Use a heredoc so the JSON is emitted verbatim regardless of arguments.
    # Bash escaping: replace single quotes inside the JSON payload.
    safe_payload = response_json.replace("'", "'\\''")
    shim.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            cat <<'GH_RESPONSE_EOF'
            {safe_payload}
            GH_RESPONSE_EOF
            """
        )
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return shim


def _build_history_response(parent_counts: list[int]) -> str:
    """Build a fake GraphQL response with one node per entry in parent_counts."""
    nodes = [
        {
            "oid": f"deadbeefdeadbeef{i:08x}",
            "messageHeadline": f"chore: commit {i}",
            "committedDate": "2026-04-15T12:00:00Z",
            "parents": {"totalCount": parents},
            "author": {"user": {"login": "test-user"}},
        }
        for i, parents in enumerate(parent_counts)
    ]
    return json.dumps(
        {
            "data": {
                "repository": {
                    "defaultBranchRef": {"target": {"history": {"nodes": nodes}}}
                }
            }
        }
    )


def _run_script(
    gh_shim_dir: Path, *extra_args: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{gh_shim_dir}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), "--repo", "omniclaude", *extra_args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.unit
def test_passes_when_only_single_parent_commits(tmp_path: Path) -> None:
    """All squash commits → exit 0, OK line emitted."""
    response = _build_history_response([1, 1, 1, 1, 1])
    _write_gh_shim(tmp_path, response)
    result = _run_script(tmp_path)
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK    omniclaude" in result.stdout
    assert "no merge-queue bypass detected" in result.stdout


@pytest.mark.unit
def test_fails_when_two_parent_commit_present(tmp_path: Path) -> None:
    """A 2-parent commit (admin-bypass merge) → exit 1, FAIL line emitted."""
    response = _build_history_response([1, 1, 2, 1, 1])
    _write_gh_shim(tmp_path, response)
    result = _run_script(tmp_path)
    assert result.returncode == 1, (
        f"expected exit 1, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "FAIL  omniclaude" in result.stdout
    assert "1 2-parent commit(s)" in result.stdout
    assert "admin-bypass suspected" in result.stdout
    assert "OMN-8843" in result.stdout


@pytest.mark.unit
def test_fails_with_multiple_two_parent_commits(tmp_path: Path) -> None:
    """Multiple 2-parent commits are all reported and counted."""
    response = _build_history_response([2, 1, 2, 1, 2])
    _write_gh_shim(tmp_path, response)
    result = _run_script(tmp_path)
    assert result.returncode == 1
    assert "3 2-parent commit(s)" in result.stdout
    assert "3 2-parent merge commit(s) on main bypassed" in result.stdout


@pytest.mark.unit
def test_skips_repo_when_gh_query_fails(tmp_path: Path) -> None:
    """If `gh api` returns non-zero, the repo is skipped; if all repos are
    skipped (AUDITED == 0), exit 2 signals an inconclusive audit.

    This prevents a false-green when token scope blocks all API calls.
    """
    failing_shim = tmp_path / "gh"
    failing_shim.write_text("#!/usr/bin/env bash\nexit 1\n")
    failing_shim.chmod(
        failing_shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    result = _run_script(tmp_path)
    assert result.returncode == 2
    assert "SKIP  omniclaude" in result.stderr
    assert "no repos were auditable" in result.stdout


@pytest.mark.unit
def test_skip_message_surfaces_real_gh_stderr(tmp_path: Path) -> None:
    """OMN-16107: a failed `gh api` call must surface its actual stderr in the
    SKIP line, not the previous blanket "token scope or repo missing" guess
    that swallowed the real diagnostic (`2>/dev/null`). This is the audit
    blind spot found live in the omnibase_compat GraphQL SKIP on the
    Merge Queue Bypass Audit workflow, where the true cause was
    undiagnosable from the log."""
    failing_shim = tmp_path / "gh"
    failing_shim.write_text(
        "#!/usr/bin/env bash\n"
        'echo "Get \\"https://api.github.com/graphql\\": unexpected EOF" >&2\n'
        "exit 1\n"
    )
    failing_shim.chmod(
        failing_shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    result = _run_script(tmp_path)
    assert result.returncode == 2
    assert "unexpected EOF" in result.stderr, (
        f"real gh stderr was not surfaced in SKIP line\nstderr: {result.stderr}"
    )


@pytest.mark.unit
def test_allowlisted_two_parent_commit_does_not_fail(tmp_path: Path) -> None:
    """OMN-16107: a 2-parent commit whose short SHA is in
    ALLOWLISTED_TWO_PARENT_SHAS is reported (ALLOW line) but must not count
    toward FAILED/TOTAL_BYPASS — this is the fix for the OMN-15028
    ancestry-bridge exception re-triggering the audit on every scheduled run
    for 10+ days."""
    # First node's oid[0:8] == "11a040da" — matches the allowlisted SHA the
    # script ships with for OMN-15028.
    response = json.dumps(
        {
            "data": {
                "repository": {
                    "defaultBranchRef": {
                        "target": {
                            "history": {
                                "nodes": [
                                    {
                                        "oid": "11a040da" + "0" * 32,
                                        "messageHeadline": (
                                            "chore(OMN-15028): ancestry bridge"
                                        ),
                                        "committedDate": "2026-07-26T07:19:08Z",
                                        "parents": {"totalCount": 2},
                                        "author": {"user": {"login": "jonahgabriel"}},
                                    },
                                    {
                                        "oid": "deadbeef" + "0" * 32,
                                        "messageHeadline": "chore: unrelated squash",
                                        "committedDate": "2026-04-15T12:00:00Z",
                                        "parents": {"totalCount": 1},
                                        "author": {"user": {"login": "test-user"}},
                                    },
                                ]
                            }
                        }
                    }
                }
            }
        }
    )
    _write_gh_shim(tmp_path, response)
    result = _run_script(tmp_path)
    assert result.returncode == 0, (
        f"expected exit 0 (allowlisted commit must not fail the audit), "
        f"got {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "ALLOW omniclaude" in result.stdout
    assert "OMN-15028" in result.stdout
    assert "FAIL" not in result.stdout


@pytest.mark.unit
def test_allowlisted_and_unallowed_two_parent_commits_mixed(tmp_path: Path) -> None:
    """An allowlisted 2-parent commit alongside a genuine unexplained one
    still fails the audit, counting only the unallowed commit."""
    response = json.dumps(
        {
            "data": {
                "repository": {
                    "defaultBranchRef": {
                        "target": {
                            "history": {
                                "nodes": [
                                    {
                                        "oid": "11a040da" + "0" * 32,
                                        "messageHeadline": (
                                            "chore(OMN-15028): ancestry bridge"
                                        ),
                                        "committedDate": "2026-07-26T07:19:08Z",
                                        "parents": {"totalCount": 2},
                                        "author": {"user": {"login": "jonahgabriel"}},
                                    },
                                    {
                                        "oid": "deadbeef" + "1" * 32,
                                        "messageHeadline": "chore: unexplained merge",
                                        "committedDate": "2026-08-01T00:00:00Z",
                                        "parents": {"totalCount": 2},
                                        "author": {"user": {"login": "someone"}},
                                    },
                                ]
                            }
                        }
                    }
                }
            }
        }
    )
    _write_gh_shim(tmp_path, response)
    result = _run_script(tmp_path)
    assert result.returncode == 1
    assert "1 2-parent commit(s)" in result.stdout
    assert "OMN-15028" in result.stdout
