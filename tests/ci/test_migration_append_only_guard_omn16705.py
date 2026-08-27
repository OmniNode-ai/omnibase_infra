# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RED/GREEN proof for the append-only migration guard (OMN-16705).

``scripts/validation/check_migration_append_only.py`` exists so the defect class
behind OMN-16705 becomes impossible rather than merely fixed once: an in-place
edit to a migration a database has already applied permanently bricks
``forward-migration`` on that database, and nothing detected it.

Each case builds a throwaway git repository that mirrors the real layout
(``docker/migrations/forward/`` plus its ``_ledger`` TSVs), commits a base
revision, then applies the change under test on a branch -- so the guard is
exercised through the same ``git diff`` path CI uses, not through a mocked one.

The RED case reproduces the shape of ``7de798a4a`` exactly: edit an
already-declared migration and update its manifest checksum to match, which is
precisely the combination every pre-existing gate accepts.

Ticket: OMN-16705
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validation.check_migration_append_only import (
    AppendOnlyViolationError,
    check,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "validation" / "check_migration_append_only.py"
FORWARD = "docker/migrations/forward"

_APPLIED = "nodes/node_example/0001_create_example.sql"
_APPLIED_BODY = "CREATE TABLE IF NOT EXISTS example (id BIGINT);\n"
_SUCCESSOR = "nodes/node_example/0002_example_not_null.sql"
_SUCCESSOR_BODY = "ALTER TABLE example ALTER COLUMN id SET NOT NULL;\n"


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest_row(artifact_path: str, body: str) -> str:
    node = artifact_path.split("/")[1]
    filename = artifact_path.split("/")[2]
    stream = f"node:{node}"
    return "\t".join(
        [artifact_path, stream, stream, "tenant", f"{stream}:{filename}", _sha256(body)]
    )


def _write(repo: Path, relative: str, text: str) -> None:
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A base revision with one declared, already-applied migration."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q", "-b", "dev")
    _run(repo, "config", "user.email", "guard@test.invalid")
    _run(repo, "config", "user.name", "guard test")
    _write(repo, f"{FORWARD}/{_APPLIED}", _APPLIED_BODY)
    _write(
        repo,
        f"{FORWARD}/_ledger/application-migrations.tsv",
        _manifest_row(_APPLIED, _APPLIED_BODY) + "\n",
    )
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "base")
    _run(repo, "branch", "base-marker")
    return repo


def _edit_applied_migration(repo: Path, new_body: str) -> None:
    """The exact shape of 7de798a4a: rewrite the file AND restamp the manifest."""
    _write(repo, f"{FORWARD}/{_APPLIED}", new_body)
    _write(
        repo,
        f"{FORWARD}/_ledger/application-migrations.tsv",
        _manifest_row(_APPLIED, new_body) + "\n",
    )


def test_rewriting_an_applied_migration_is_rejected(repo: Path) -> None:
    """RED: the defect that produced OMN-16705 is caught."""
    _edit_applied_migration(
        repo, _APPLIED_BODY + "ALTER TABLE example ADD COLUMN x INT;\n"
    )
    _run(repo, "commit", "-qam", "rewrite in place")

    violations = check(repo, base="base-marker", staged=False)

    assert len(violations) == 1, violations
    assert _APPLIED in violations[0]
    assert "supersession" in violations[0]


def test_deleting_an_applied_migration_is_rejected(repo: Path) -> None:
    """Applied history cannot be removed either -- bootstrap.sql still reads it."""
    (repo / FORWARD / _APPLIED).unlink()
    _write(repo, f"{FORWARD}/_ledger/application-migrations.tsv", "")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "delete applied migration")

    violations = check(repo, base="base-marker", staged=False)

    assert len(violations) == 1, violations
    assert _APPLIED in violations[0]


def test_adding_a_new_ordinal_is_allowed(repo: Path) -> None:
    """GREEN: the sanctioned forward-fix shape passes untouched."""
    _write(repo, f"{FORWARD}/{_SUCCESSOR}", _SUCCESSOR_BODY)
    _write(
        repo,
        f"{FORWARD}/_ledger/application-migrations.tsv",
        _manifest_row(_APPLIED, _APPLIED_BODY)
        + "\n"
        + _manifest_row(_SUCCESSOR, _SUCCESSOR_BODY)
        + "\n",
    )
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "add successor")

    assert check(repo, base="base-marker", staged=False) == []


def _land_supersession(repo: Path, *, with_successor: bool) -> None:
    _edit_applied_migration(repo, "-- restored to the applied bytes\n" + _APPLIED_BODY)
    rows = [
        _manifest_row(_APPLIED, "-- restored to the applied bytes\n" + _APPLIED_BODY)
    ]
    if with_successor:
        _write(repo, f"{FORWARD}/{_SUCCESSOR}", _SUCCESSOR_BODY)
        rows.append(_manifest_row(_SUCCESSOR, _SUCCESSOR_BODY))
    _write(
        repo, f"{FORWARD}/_ledger/application-migrations.tsv", "\n".join(rows) + "\n"
    )
    _write(
        repo,
        f"{FORWARD}/_ledger/migration-supersessions.tsv",
        "\t".join([_APPLIED, _SUCCESSOR, "OMN-16705", "restore applied bytes"]) + "\n",
    )
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "supersede")


def test_supersession_with_its_successor_is_allowed(repo: Path) -> None:
    """GREEN: this repair's own shape -- restore the bytes, land the successor."""
    _land_supersession(repo, with_successor=True)

    assert check(repo, base="base-marker", staged=False) == []


def test_supersession_without_its_successor_is_rejected(repo: Path) -> None:
    """A supersession row is not a standing waiver."""
    _land_supersession(repo, with_successor=False)

    violations = check(repo, base="base-marker", staged=False)

    assert len(violations) == 1, violations
    assert "not ADDED by this change" in violations[0]


def test_a_stale_supersession_cannot_authorise_a_second_edit(repo: Path) -> None:
    """The escape hatch is one-shot: it does not survive into the next change."""
    _land_supersession(repo, with_successor=True)
    _run(repo, "branch", "landed")
    _edit_applied_migration(repo, "-- second, unauthorised rewrite\n")
    _run(repo, "commit", "-qam", "second rewrite")

    violations = check(repo, base="landed", staged=False)

    assert len(violations) == 1, violations
    assert "not ADDED by this change" in violations[0]


def test_a_migration_added_in_this_change_can_still_be_amended(repo: Path) -> None:
    """Only history that already reached the base ref is frozen."""
    _write(repo, f"{FORWARD}/{_SUCCESSOR}", _SUCCESSOR_BODY)
    _write(
        repo,
        f"{FORWARD}/_ledger/application-migrations.tsv",
        _manifest_row(_APPLIED, _APPLIED_BODY)
        + "\n"
        + _manifest_row(_SUCCESSOR, _SUCCESSOR_BODY)
        + "\n",
    )
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "add successor")
    amended = _SUCCESSOR_BODY + "-- amended within the same change\n"
    _write(repo, f"{FORWARD}/{_SUCCESSOR}", amended)
    _write(
        repo,
        f"{FORWARD}/_ledger/application-migrations.tsv",
        _manifest_row(_APPLIED, _APPLIED_BODY)
        + "\n"
        + _manifest_row(_SUCCESSOR, amended)
        + "\n",
    )
    _run(repo, "commit", "-qam", "amend the new migration")

    assert check(repo, base="base-marker", staged=False) == []


def test_staged_mode_sees_the_change_before_it_is_committed(repo: Path) -> None:
    """Pre-commit mode is the same predicate against the index."""
    _edit_applied_migration(repo, "-- staged rewrite\n")
    _run(repo, "add", "-A")

    violations = check(repo, base=None, staged=True)

    assert len(violations) == 1, violations
    assert _APPLIED in violations[0]


def test_an_empty_base_manifest_fails_closed(tmp_path: Path) -> None:
    """Anti-vacuity: no declarations at the base must not mean 'everything passes'."""
    repo = tmp_path / "empty"
    repo.mkdir()
    _run(repo, "init", "-q", "-b", "dev")
    _run(repo, "config", "user.email", "guard@test.invalid")
    _run(repo, "config", "user.name", "guard test")
    _write(repo, "README.md", "no migrations here\n")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "base")

    with pytest.raises(AppendOnlyViolationError, match="anti-vacuity"):
        check(repo, base="HEAD", staged=False)


def test_the_guard_runs_as_a_script_and_reports_a_failing_exit_code(
    repo: Path,
) -> None:
    """The CI step and the pre-commit hook both invoke it as a process."""
    _edit_applied_migration(repo, "-- rewrite\n")
    _run(repo, "commit", "-qam", "rewrite")

    result = subprocess.run(
        [sys.executable, str(GUARD), "--repo-root", str(repo), "--base", "base-marker"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "OMN-16705" in result.stderr


_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn16705"
_CREDENTIALS_ARTIFACT = "nodes/node_projection_tenant_credentials/0000_create_tenant_inference_credentials.sql"
# Verbatim blobs, captured with `git show` and never edited since. The applied
# bytes are what the .201 dev lane's platform_catalog.schema_migrations records;
# the rewritten bytes are what OMN-16450 (#2866) merged over them.
_APPLIED_CAPTURE = (
    _FIXTURES / "0000_create_tenant_inference_credentials.559ee461a.sql.captured",
    "c1691130f33e3e7ca1cbf64572d58324d1a3c5d1e4156f60cb4f1b7a612ea68c",
)
_REWRITTEN_CAPTURE = (
    _FIXTURES / "0000_create_tenant_inference_credentials.7de798a4a.sql.captured",
    "d113ac80e173e79ad90563ddc1b09be85280c751191aa34eb0bc6be7a6f82ec5",
)


def _captured(entry: tuple[Path, str]) -> str:
    path, expected = entry
    body = path.read_text(encoding="utf-8")
    actual = _sha256(body)
    assert actual == expected, (
        f"{path.name} is no longer the bytes that were captured "
        f"({actual} != {expected}); a reformatted artifact is not the artifact"
    )
    return body


def test_the_omn16450_rewrite_is_rejected_by_the_real_guard(tmp_path: Path) -> None:
    """Incident replay: the merge that bricked the .201 dev lane (OMN-16705).

    Reconstructs #2866 exactly -- a base revision declaring the bytes the lane
    had applied, and a head revision carrying the rewritten bytes with the
    manifest checksum restamped to match, which is the combination every
    pre-existing gate accepted. Drives the real ``check()`` and requires reject.
    """
    applied = _captured(_APPLIED_CAPTURE)
    rewritten = _captured(_REWRITTEN_CAPTURE)
    assert applied != rewritten, "the replay would be vacuous"

    repo = tmp_path / "omn16450-replay"
    repo.mkdir()
    _run(repo, "init", "-q", "-b", "dev")
    _run(repo, "config", "user.email", "guard@test.invalid")
    _run(repo, "config", "user.name", "guard test")
    _write(repo, f"{FORWARD}/{_CREDENTIALS_ARTIFACT}", applied)
    _write(
        repo,
        f"{FORWARD}/_ledger/application-migrations.tsv",
        _manifest_row(_CREDENTIALS_ARTIFACT, applied) + "\n",
    )
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "dev before #2866")
    _run(repo, "branch", "dev-before-2866")

    _write(repo, f"{FORWARD}/{_CREDENTIALS_ARTIFACT}", rewritten)
    _write(
        repo,
        f"{FORWARD}/_ledger/application-migrations.tsv",
        _manifest_row(_CREDENTIALS_ARTIFACT, rewritten) + "\n",
    )
    _run(repo, "commit", "-qam", "fix(OMN-16450): restore node migration shape proof")

    violations = check(repo, base="dev-before-2866", staged=False)

    assert len(violations) == 1, violations
    assert _CREDENTIALS_ARTIFACT in violations[0]
    assert "supersession" in violations[0]


def test_the_guard_is_wired_as_a_precommit_hook() -> None:
    """Rule 5: detection that is not enforcement gets ignored."""
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "check_migration_append_only.py" in config


def test_the_guard_is_wired_as_a_ci_gate() -> None:
    """It must run inside a job that already blocks merge."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "check_migration_append_only.py" in workflow


def test_this_repository_satisfies_its_own_guard() -> None:
    """The repair itself must pass the rule it introduces."""
    try:
        violations = check(REPO_ROOT, base="origin/dev", staged=False)
    except AppendOnlyViolationError as exc:
        pytest.skip(f"base ref unavailable in this checkout: {exc}")
    assert violations == [], violations
