# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The release-notes node runs through the real ``onex node`` CLI (OMN-17427).

The release train's cut job runs ``onex node node_release_notes_apply_effect`` over
the checkout it is cutting. These tests drive the same CLI entry point
(``run_node_by_name``) against throwaway clones, so the contract's single-shot
handler block, the request model and the exit code a refusal must produce are
exercised through the runtime and not only by calling the handler.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_node import run_node_by_name

_REPO_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.integration

_NODE_NAME = "node_release_notes_apply_effect"
_CONTRACT = (
    _REPO_ROOT / "src" / "omnibase_infra" / "nodes" / _NODE_NAME / "contract.yaml"
)

_ENTRY = """## v0.47.66 (2026-11-04)

### Release
- Cut omnibase-core from dev at 0.47.66 by the scheduled release train.

### Included Since v0.47.62
- fix: a ticket-cited advisory annotation is not suppression growth (#2257)
- feat: theme contract gains the board tokens (#2231)
"""

_README = """# omnibase_core

## v0.47.62 (2026-11-02)

### Release
- older notes
"""

_CONTRACT_YAML = """ticket: OMN-18595
dod_evidence:
  - id: dod-omn18595-ac1-release-0-47-62-notes
    description: older.
    source: manual
    binds_ac: [AC1]
    checks:
      - check_type: test_passes
        check_value: uv run pytest tests/ci/test_release_cut_identity_omn18595.py -q -k changelog
        cwd: ${OMNI_HOME}/omnibase_core
"""

_IDENTITY_TEST = """REPO_ROOT = None


def test_release_changelog_retains_version_0_47_62_in_root_readme() -> None:
    assert True
"""


def _readme_only_clone(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / ".readme-only-docs.yaml").write_text("exceptions: []\n", encoding="utf-8")
    (root / "README.md").write_text(_README, encoding="utf-8")
    (root / "contracts").mkdir()
    (root / "contracts" / "OMN-18595.yaml").write_text(_CONTRACT_YAML, encoding="utf-8")
    (root / "tests" / "ci").mkdir(parents=True)
    (root / "tests" / "ci" / "test_release_cut_identity_omn18595.py").write_text(
        _IDENTITY_TEST, encoding="utf-8"
    )
    return root


def _run_node(
    tmp_path: Path,
    clone: Path,
    entry: Path,
    *,
    repo_name: str,
    visibility: Literal["public", "private", "internal"] = "private",
    vocabulary: Path | None = None,
) -> tuple[int, str]:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "clone_path": str(clone),
                "entry_file": str(entry),
                "repo_name": repo_name,
                "repository_visibility": visibility,
                "hygiene_vocabulary_file": str(vocabulary) if vocabulary else None,
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        run_node_by_name,
        [
            _NODE_NAME,
            "--contract",
            str(_CONTRACT),
            "--input",
            str(request),
            "--state-root",
            str(tmp_path / "state"),
            "--timeout",
            "60",
            "--output",
            "receipt",
        ],
    )
    return result.exit_code, result.stdout


@pytest.fixture
def repository_name_release(tmp_path: Path) -> tuple[Path, Path, str]:
    """Keep the recorded release subject, substituting a synthetic private slug."""
    subject = (
        "fix: the writer-app workflow pin classifier admits Example_Secret's "
        "OMNIBASE_CORE_COMMIT env pin bump (#4610)"
    )
    entry = tmp_path / "entry.md"
    entry.write_text(
        f"## v0.4.321 (2026-11-18)\n\n### Included Since v0.4.320\n- {subject}\n"
        "- fix: retry example_secret-tool without changing its name (#4611)\n"
        "- fix: align Example-Other, example_secret and example.other (#4612)\n",
        encoding="utf-8",
    )
    vocabulary = tmp_path / "vocabulary.yaml"
    vocabulary.write_text(
        "private_repo_names: [example_secret, example-other, example.other]\n",
        encoding="utf-8",
    )
    return entry, vocabulary, subject


def test_public_repository_changelog_filters_private_subject_names(
    tmp_path: Path, repository_name_release: tuple[Path, Path, str]
) -> None:
    entry, vocabulary, original_subject = repository_name_release
    (tmp_path / "CHANGELOG.md").write_text("", encoding="utf-8")
    code, output = _run_node(
        tmp_path,
        tmp_path,
        entry,
        repo_name="public-example",
        visibility="public",
        vocabulary=vocabulary,
    )
    assert code == 0, output
    changelog = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert original_subject not in changelog
    assert "Example_Secret" not in changelog
    assert "OMNIBASE_CORE_COMMIT env pin bump (#4610)" in changelog
    assert "retry example_secret-tool without changing its name (#4611)" in changelog
    assert "Example-Other" not in changelog
    assert "example_secret and" not in changelog
    assert "example.other" not in changelog
    assert "(#4612)" in changelog


def test_private_repository_changelog_preserves_the_release_block(
    tmp_path: Path, repository_name_release: tuple[Path, Path, str]
) -> None:
    entry, vocabulary, original_subject = repository_name_release
    (tmp_path / "CHANGELOG.md").write_text("", encoding="utf-8")
    code, output = _run_node(
        tmp_path,
        tmp_path,
        entry,
        repo_name="private-example",
        vocabulary=vocabulary,
    )
    assert code == 0, output
    changelog = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert original_subject in changelog
    assert changelog == entry.read_text(encoding="utf-8").rstrip("\n") + "\n\n"


@pytest.mark.parametrize("invalid_vocabulary", [None, "private_repo_names: []", "["])
def test_public_repository_refuses_to_write_without_a_valid_vocabulary(
    tmp_path: Path,
    repository_name_release: tuple[Path, Path, str],
    invalid_vocabulary: str | None,
) -> None:
    entry, vocabulary, _ = repository_name_release
    if invalid_vocabulary is not None:
        vocabulary.write_text(invalid_vocabulary, encoding="utf-8")
    code, output = _run_node(
        tmp_path,
        tmp_path,
        entry,
        repo_name="public-example",
        visibility="public",
        vocabulary=vocabulary if invalid_vocabulary is not None else None,
    )
    assert code == 1, output
    assert (
        "release hygiene vocabulary" in output
        or "require a hygiene vocabulary" in output
    )
    assert not (tmp_path / "CHANGELOG.md").exists()


def _read_staged_paths(
    tmp_path: Path, receipt: str
) -> subprocess.CompletedProcess[bytes]:
    (tmp_path / "release-notes-receipt.json").write_text(receipt, encoding="utf-8")
    workflow = (
        _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"
    ).read_text(encoding="utf-8")
    start = workflow.index("          if ! staged_lines=")
    end = workflow.index('          if [ "${NEEDS_BUMP}"', start)
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\nRUNNER_TEMP="$1"\n'
            + workflow[start:end]
            + '\nprintf "%s\\0" "${staged[@]}"',
            "release-notes-readback",
            str(tmp_path),
        ],
        capture_output=True,
        check=False,
    )


class TestRunThroughTheNodeCli:
    """The cut job runs ``onex node node_release_notes_apply_effect``; so do these."""

    def test_the_cut_job_reads_every_path_from_the_runtime_receipt(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path / "clone with spaces")
        (clone / "CHANGELOG.md").write_text("stale notes\n", encoding="utf-8")
        entry = tmp_path / "entry with spaces.md"
        entry.write_text(_ENTRY, encoding="utf-8")
        code, output = _run_node(tmp_path, clone, entry, repo_name="omnibase_core")
        assert code == 0, output
        json.loads(output)
        result = _read_staged_paths(tmp_path, output)
        assert result.returncode == 0, result.stderr
        assert result.stdout.split(b"\0") == [
            b"README.md",
            b"CHANGELOG.md",
            b"contracts/OMN-18595.yaml",
            b"tests/ci/test_release_cut_identity_omn18595.py",
            b"",
        ]
        assert not (clone / "CHANGELOG.md").exists()

    @pytest.mark.parametrize(
        "receipt",
        [
            "{",
            '{"result": {"success": false, "staged": ["README.md"]}}',
            '{"result": {"success": true, "staged": []}}',
        ],
        ids=["malformed", "refused", "empty"],
    )
    def test_the_cut_job_refuses_an_unusable_receipt(
        self, tmp_path: Path, receipt: str
    ) -> None:
        result = _read_staged_paths(tmp_path, receipt)
        assert result.returncode == 1
        assert b"does not report success with staged paths" in result.stdout

    def test_the_cut_job_accepts_a_release_pr_body_receipt_with_no_staged_path(
        self, tmp_path: Path
    ) -> None:
        receipt = (
            '{"result": {"success": true, "target": "release-pr-body", '
            '"staged": [], "notes": "## v0.4.324 (2026-11-21)"}}'
        )
        result = _read_staged_paths(tmp_path, receipt)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip(b"\0") == b""

    def test_the_node_writes_a_readme_only_clone_and_exits_zero(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path / "clone")
        entry = tmp_path / "entry.md"
        entry.write_text(_ENTRY, encoding="utf-8")
        code, output = _run_node(tmp_path, clone, entry, repo_name="omnibase_core")
        assert code == 0, output
        readme = (clone / "README.md").read_text(encoding="utf-8")
        assert readme.index("## v0.47.66 (") < readme.index("## v0.47.62 (")
        assert "def test_release_changelog_retains_version_0_47_66" in (
            clone / "tests" / "ci" / "test_release_cut_identity_omn18595.py"
        ).read_text(encoding="utf-8")

    def test_the_node_exits_nonzero_on_a_refusal(self, tmp_path: Path) -> None:
        clone = tmp_path / "clone"
        clone.mkdir()
        (clone / ".readme-only-docs.yaml").write_text("exceptions: []\n", "utf-8")
        entry = tmp_path / "entry.md"
        entry.write_text(_ENTRY, encoding="utf-8")
        code, output = _run_node(tmp_path, clone, entry, repo_name="x")
        assert code == 1, output
        assert not (clone / "README.md").exists()

    def test_the_node_ignores_fenced_examples_and_refuses_a_duplicate_cut(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path / "clone")
        example = "```markdown\n## v0.47.66 (2026-11-04)\n```\n\n"
        (clone / "README.md").write_text(example + _README, encoding="utf-8")
        entry = tmp_path / "entry.md"
        entry.write_text(_ENTRY, encoding="utf-8")
        code, output = _run_node(tmp_path, clone, entry, repo_name="omnibase_core")
        assert code == 0, output
        readme = (clone / "README.md").read_text(encoding="utf-8")
        assert readme.startswith(example + "# omnibase_core\n\n## v0.47.66 (")
        before = {p: p.read_bytes() for p in clone.rglob("*") if p.is_file()}
        code, output = _run_node(tmp_path, clone, entry, repo_name="omnibase_core")
        assert code == 1, output
        assert "already has" in output
        assert {p: p.read_bytes() for p in before} == before

    def test_the_node_refuses_a_trailing_contract_key_without_writes(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path / "clone")
        contract = clone / "contracts/OMN-18595.yaml"
        contract.write_text(_CONTRACT_YAML + '"x-foo": true\n', encoding="utf-8")
        entry = tmp_path / "entry.md"
        entry.write_text(_ENTRY, encoding="utf-8")
        before = {p: p.read_bytes() for p in clone.rglob("*") if p.is_file()}
        code, output = _run_node(tmp_path, clone, entry, repo_name="omnibase_core")
        assert code == 1, output
        assert "dod_evidence" in output
        assert {p: p.read_bytes() for p in before} == before

    def test_a_repo_without_a_docs_policy_keeps_the_changelog_prepend(
        self, tmp_path: Path
    ) -> None:
        clone = tmp_path / "clone"
        clone.mkdir()
        (clone / "CHANGELOG.md").write_text("## v0.1.0 (2026-01-01)\n", "utf-8")
        entry = tmp_path / "entry.md"
        entry.write_text(_ENTRY, encoding="utf-8")
        code, output = _run_node(tmp_path, clone, entry, repo_name="omnibase_infra")
        assert code == 0, output
        changelog = (clone / "CHANGELOG.md").read_text(encoding="utf-8")
        assert changelog.startswith("## v0.47.66 (")
        assert changelog.endswith("\n\n## v0.1.0 (2026-01-01)\n")
        assert not (clone / "README.md").exists()
