# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A README-only repo's release block goes in README.md, not CHANGELOG.md (OMN-17427).

omnibase_core deleted its root ``CHANGELOG.md`` under the README-only documentation
policy (OMN-17427, omnibase_core#2045). The train kept writing that file, so every
core release PR was born red on ``kb-doc-gate`` and the Pre-commit Suite
(``readme-only-docs`` rule ``root-readme-only``). The accepted shape, hand-made for
0.47.59 to 0.47.62 and 0.47.66, is the release block in root ``README.md`` above the
newest release section, plus a pinned identity test and a ``dod_evidence`` item.

The owner of "does this repo take a changelog" is the repo's own
``.readme-only-docs.yaml``, so the writer reads that file and no train policy key
carries a second copy.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "ci" / "release_train.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"


def _load() -> Any:
    name = "release_train_readme_only_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = f"cannot load {_MODULE}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load()

_SUBJECTS = (
    "fix: a ticket-cited advisory annotation is not suppression growth (#2257)",
    "feat: middle change (#2250)",
    "feat: theme contract gains the board tokens (#2231)",
)

_README = """# omnibase_core

Intro line.

## v0.47.62 (2026-11-02)

### Release
- older notes
"""

_CONTRACT = """ticket: OMN-18595
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

_TEST = """REPO_ROOT = None


def test_release_changelog_retains_version_0_47_62_in_root_readme() -> None:
    assert True
"""


def _entry() -> str:
    return rt.render_changelog_entry(
        package="omnibase-core",
        version="0.47.66",
        previous_tag="v0.47.62",
        subjects=_SUBJECTS,
    )


def _readme_only_clone(tmp_path: Path, *, exceptions: str = "[]") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".readme-only-docs.yaml").write_text(
        f"---\nexceptions: {exceptions}\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(_README, encoding="utf-8")
    (tmp_path / "contracts").mkdir()
    (tmp_path / "contracts" / "OMN-18595.yaml").write_text(_CONTRACT, encoding="utf-8")
    (tmp_path / "tests" / "ci").mkdir(parents=True)
    (tmp_path / "tests" / "ci" / "test_release_cut_identity_omn18595.py").write_text(
        _TEST, encoding="utf-8"
    )
    return tmp_path


@pytest.mark.unit
class TestReleaseNotesTarget:
    def test_a_repo_without_the_policy_file_keeps_the_changelog(
        self, tmp_path: Path
    ) -> None:
        assert rt.release_notes_target(tmp_path) == "CHANGELOG.md"

    def test_a_repo_carrying_the_policy_file_takes_its_notes_in_the_readme(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        assert rt.release_notes_target(clone) == "README.md"

    def test_a_policy_file_that_excepts_the_changelog_keeps_the_changelog(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(
            tmp_path, exceptions="[{path: CHANGELOG.md, reason: published}]"
        )
        assert rt.release_notes_target(clone) == "CHANGELOG.md"

    def test_an_unparseable_policy_file_refuses_rather_than_guessing(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".readme-only-docs.yaml").write_text("exceptions: [", "utf-8")
        with pytest.raises(rt.ReleaseTrainConfigError):
            rt.release_notes_target(tmp_path)


@pytest.mark.unit
class TestApplyReleaseNotes:
    def test_a_changelog_repo_gets_the_historical_prepend(self, tmp_path: Path) -> None:
        (tmp_path / "CHANGELOG.md").write_text("## v0.1.0 (2026-01-01)\n", "utf-8")
        staged = rt.apply_release_notes(tmp_path, _entry())
        assert staged == ["CHANGELOG.md"]
        text = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
        assert text.startswith("## v0.47.66 (")
        assert text.endswith("\n\n## v0.1.0 (2026-01-01)\n")

    def test_a_readme_only_repo_gets_the_block_above_the_newest_section(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        staged = rt.apply_release_notes(clone, _entry(), repo_name="omnibase_core")
        assert staged == [
            "README.md",
            "contracts/OMN-18595.yaml",
            "tests/ci/test_release_cut_identity_omn18595.py",
        ]
        readme = (clone / "README.md").read_text(encoding="utf-8")
        assert readme.index("## v0.47.66 (") < readme.index("## v0.47.62 (2026-11-02)")
        assert readme.startswith("# omnibase_core\n\nIntro line.\n\n## v0.47.66 (")
        assert "\n\n## v0.47.62 (2026-11-02)\n" in readme
        assert not (clone / "CHANGELOG.md").exists()

    def test_a_refused_root_changelog_is_deleted_and_staged(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        (clone / "CHANGELOG.md").write_text("stale\n", encoding="utf-8")
        staged = rt.apply_release_notes(clone, _entry(), repo_name="omnibase_core")
        assert "CHANGELOG.md" in staged
        assert not (clone / "CHANGELOG.md").exists()

    def test_the_contract_item_and_identity_test_are_well_formed_and_pin_the_block(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        rt.apply_release_notes(clone, _entry(), repo_name="omnibase_core")
        contract = yaml.safe_load(
            (clone / "contracts" / "OMN-18595.yaml").read_text(encoding="utf-8")
        )
        item = contract["dod_evidence"][-1]
        assert item["id"] == "dod-omn18595-ac1-release-0-47-66-notes"
        assert len(item["id"]) <= 50
        assert item["binds_ac"] == ["AC1"]
        assert item["checks"][1]["cwd"] == "${OMNI_HOME}/omnibase_core"
        command = item["checks"][1]["check_value"]
        assert "## v0.47.66 (" in command
        assert "### Included Since v0.47.62" in command
        assert '"(#2257)" in notes and "(#2231)" in notes' in command
        source = (
            clone / "tests" / "ci" / "test_release_cut_identity_omn18595.py"
        ).read_text(encoding="utf-8")
        ast.parse(source)
        assert "def test_release_changelog_retains_version_0_47_66_in_root_readme" in (
            source
        )
        assert "### Included Since v0.47.62" in source
        assert _SUBJECTS[0] in source

    def test_a_contract_that_does_not_end_in_dod_evidence_refuses(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        path = clone / "contracts" / "OMN-18595.yaml"
        path.write_text(path.read_text("utf-8") + "notes: later\n", encoding="utf-8")
        with pytest.raises(rt.ReleaseTrainConfigError):
            rt.apply_release_notes(clone, _entry(), repo_name="omnibase_core")

    def test_a_readme_only_repo_without_a_readme_refuses(self, tmp_path: Path) -> None:
        (tmp_path / ".readme-only-docs.yaml").write_text("exceptions: []\n", "utf-8")
        with pytest.raises(rt.ReleaseTrainConfigError):
            rt.apply_release_notes(tmp_path, _entry())

    def test_an_entry_in_another_shape_refuses(self) -> None:
        with pytest.raises(rt.ReleaseTrainConfigError):
            rt.parse_release_entry("# not a release block\n")

    def test_the_subcommand_prints_one_staged_path_per_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        clone = _readme_only_clone(tmp_path / "clone")
        entry = tmp_path / "entry.md"
        entry.write_text(_entry(), encoding="utf-8")
        code = rt.main(
            [
                "apply-release-notes",
                "--clone",
                str(clone),
                "--entry-file",
                str(entry),
                "--repo",
                "omnibase_core",
            ]
        )
        assert code == 0
        assert capsys.readouterr().out.split() == [
            "README.md",
            "contracts/OMN-18595.yaml",
            "tests/ci/test_release_cut_identity_omn18595.py",
        ]


@pytest.mark.unit
class TestWorkflowWritesThroughTheSubcommand:
    def test_the_cut_job_no_longer_writes_the_changelog_unconditionally(self) -> None:
        body = _WORKFLOW.read_text(encoding="utf-8")
        assert 'release_train.py" apply-release-notes' in body
        assert 'pathlib.Path("CHANGELOG.md")' not in body, (
            "an inline CHANGELOG.md write is the second writer that ignores the "
            "repo's .readme-only-docs.yaml"
        )
        assert "staged=(CHANGELOG.md)" not in body
