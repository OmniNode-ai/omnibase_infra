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
    tmp_path: Path, clone: Path, entry: Path, *, repo_name: str
) -> tuple[int, str]:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "clone_path": str(clone),
                "entry_file": str(entry),
                "repo_name": repo_name,
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
