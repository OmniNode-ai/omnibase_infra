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
``.readme-only-docs.yaml``, so the node reads that file and no train policy key
carries a second copy. ``scripts/ci/release_train.py`` is baselined shrink-only
(OMN-20304), so this behaviour is a node + handler the cut job runs, and the
cut job's wiring is asserted at the end.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_release_notes_apply import (
    HandlerReleaseNotesApply,
    apply_release_notes,
    parse_release_entry,
    release_notes_target,
)
from omnibase_infra.models.model_release_notes_apply_request import (
    ModelReleaseNotesApplyRequest,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NODE_NAME = "node_release_notes_apply_effect"
_CONTRACT = (
    _REPO_ROOT / "src" / "omnibase_infra" / "nodes" / _NODE_NAME / "contract.yaml"
)
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"
_TRAIN = _REPO_ROOT / "scripts" / "ci" / "release_train.py"

_SUBJECTS = (
    "fix: a ticket-cited advisory annotation is not suppression growth (#2257)",
    "feat: middle change (#2250)",
    "feat: theme contract gains the board tokens (#2231)",
)

_ENTRY = """## v0.47.66 (2026-11-04)

### Release
- Cut omnibase-core from dev at 0.47.66 by the scheduled release train.

### Included Since v0.47.62
- fix: a ticket-cited advisory annotation is not suppression growth (#2257)
- feat: middle change (#2250)
- feat: theme contract gains the board tokens (#2231)
"""

_README = """# omnibase_core

Intro line.

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

_STAGED_FOR_README_ONLY = [
    "README.md",
    "contracts/OMN-18595.yaml",
    "tests/ci/test_release_cut_identity_omn18595.py",
]


def _readme_only_clone(tmp_path: Path, *, exceptions: str = "[]") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".readme-only-docs.yaml").write_text(
        f"---\nexceptions: {exceptions}\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(_README, encoding="utf-8")
    (tmp_path / "contracts").mkdir()
    (tmp_path / "contracts" / "OMN-18595.yaml").write_text(
        _CONTRACT_YAML, encoding="utf-8"
    )
    (tmp_path / "tests" / "ci").mkdir(parents=True)
    (tmp_path / "tests" / "ci" / "test_release_cut_identity_omn18595.py").write_text(
        _IDENTITY_TEST, encoding="utf-8"
    )
    return tmp_path


@pytest.mark.unit
class TestReleaseNotesTarget:
    def test_a_repo_without_the_policy_file_keeps_the_changelog(
        self, tmp_path: Path
    ) -> None:
        assert release_notes_target(tmp_path) == "CHANGELOG.md"

    def test_a_repo_carrying_the_policy_file_takes_its_notes_in_the_readme(
        self, tmp_path: Path
    ) -> None:
        assert release_notes_target(_readme_only_clone(tmp_path)) == "README.md"

    def test_a_policy_file_that_excepts_the_changelog_keeps_the_changelog(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(
            tmp_path, exceptions="[{path: CHANGELOG.md, reason: published}]"
        )
        assert release_notes_target(clone) == "CHANGELOG.md"

    def test_an_unparseable_policy_file_refuses_rather_than_guessing(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".readme-only-docs.yaml").write_text("exceptions: [", "utf-8")
        with pytest.raises(ValueError, match="does not parse"):
            release_notes_target(tmp_path)


@pytest.mark.unit
class TestApplyReleaseNotes:
    def test_a_changelog_repo_gets_the_historical_prepend(self, tmp_path: Path) -> None:
        (tmp_path / "CHANGELOG.md").write_text("## v0.1.0 (2026-01-01)\n", "utf-8")
        target, staged = apply_release_notes(tmp_path, _ENTRY)
        assert (target, staged) == ("CHANGELOG.md", ["CHANGELOG.md"])
        text = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
        assert text.startswith("## v0.47.66 (")
        assert text.endswith("\n\n## v0.1.0 (2026-01-01)\n")

    def test_a_readme_only_repo_gets_the_block_above_the_newest_section(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        target, staged = apply_release_notes(clone, _ENTRY, repo_name="omnibase_core")
        assert target == "README.md"
        assert staged == _STAGED_FOR_README_ONLY
        readme = (clone / "README.md").read_text(encoding="utf-8")
        assert readme.index("## v0.47.66 (") < readme.index("## v0.47.62 (2026-11-02)")
        assert readme.startswith("# omnibase_core\n\nIntro line.\n\n## v0.47.66 (")
        assert "\n\n## v0.47.62 (2026-11-02)\n" in readme
        assert not (clone / "CHANGELOG.md").exists()

    def test_a_readme_without_a_release_section_takes_the_block_at_the_end(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        (clone / "README.md").write_text("# omnibase_core\n\nIntro.\n", "utf-8")
        apply_release_notes(clone, _ENTRY, repo_name="omnibase_core")
        readme = (clone / "README.md").read_text(encoding="utf-8")
        assert readme.startswith("# omnibase_core\n\nIntro.\n\n## v0.47.66 (")

    def test_a_readme_only_repo_without_the_companions_stages_only_the_readme(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".readme-only-docs.yaml").write_text("exceptions: []\n", "utf-8")
        (tmp_path / "README.md").write_text(_README, encoding="utf-8")
        _, staged = apply_release_notes(tmp_path, _ENTRY)
        assert staged == ["README.md"]

    def test_a_refused_root_changelog_is_deleted_and_staged(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        (clone / "CHANGELOG.md").write_text("stale\n", encoding="utf-8")
        _, staged = apply_release_notes(clone, _ENTRY, repo_name="omnibase_core")
        assert staged == [
            "README.md",
            "CHANGELOG.md",
            "contracts/OMN-18595.yaml",
            "tests/ci/test_release_cut_identity_omn18595.py",
        ]
        assert not (clone / "CHANGELOG.md").exists()

    def test_the_contract_item_and_identity_test_are_well_formed_and_pin_the_block(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        apply_release_notes(clone, _ENTRY, repo_name="omnibase_core")
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

    def test_a_contract_that_does_not_end_in_dod_evidence_refuses_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path)
        path = clone / "contracts" / "OMN-18595.yaml"
        path.write_text(path.read_text("utf-8") + "notes: later\n", encoding="utf-8")
        with pytest.raises(ValueError, match="dod_evidence"):
            apply_release_notes(clone, _ENTRY, repo_name="omnibase_core")
        assert (clone / "README.md").read_text(encoding="utf-8") == _README

    def test_a_readme_only_repo_without_a_readme_refuses(self, tmp_path: Path) -> None:
        (tmp_path / ".readme-only-docs.yaml").write_text("exceptions: []\n", "utf-8")
        with pytest.raises(ValueError, match=r"README\.md, which is absent"):
            apply_release_notes(tmp_path, _ENTRY)

    def test_an_entry_in_another_shape_refuses(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            parse_release_entry("# not a release block\n")

    def test_an_entry_without_an_included_since_section_refuses(self) -> None:
        with pytest.raises(ValueError, match="Included Since"):
            parse_release_entry("## v1.2.3 (2026-01-01)\n\n### Release\n- x\n")

    def test_the_trains_own_rendered_block_parses(self) -> None:
        name = "release_train_rendering_under_test"
        spec = importlib.util.spec_from_file_location(name, _TRAIN)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        rendered = module.render_changelog_entry(
            package="omnibase-core",
            version="0.47.66",
            previous_tag="v0.47.62",
            subjects=_SUBJECTS,
        )
        parsed = parse_release_entry(rendered)
        assert (parsed.version, parsed.since) == ("0.47.66", "v0.47.62")
        assert parsed.subjects == _SUBJECTS


@pytest.mark.unit
class TestHandler:
    async def test_handle_returns_the_target_and_the_paths_to_stage(
        self, tmp_path: Path
    ) -> None:
        clone = _readme_only_clone(tmp_path / "clone")
        entry = tmp_path / "entry.md"
        entry.write_text(_ENTRY, encoding="utf-8")
        result = await HandlerReleaseNotesApply().handle(
            ModelReleaseNotesApplyRequest(
                clone_path=str(clone), entry_file=str(entry), repo_name="omnibase_core"
            )
        )
        assert result.success is True
        assert result.target == "README.md"
        assert list(result.staged) == _STAGED_FOR_README_ONLY
        assert result.error_message is None

    async def test_handle_reports_a_refusal_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".readme-only-docs.yaml").write_text("exceptions: []\n", "utf-8")
        entry = tmp_path / "entry.md"
        entry.write_text(_ENTRY, encoding="utf-8")
        result = await HandlerReleaseNotesApply().handle(
            ModelReleaseNotesApplyRequest(
                clone_path=str(tmp_path), entry_file=str(entry)
            )
        )
        assert result.success is False
        assert result.staged == ()
        assert result.error_message is not None
        assert "README.md, which is absent" in result.error_message

    async def test_handle_reports_a_missing_entry_file(self, tmp_path: Path) -> None:
        result = await HandlerReleaseNotesApply().handle(
            ModelReleaseNotesApplyRequest(
                clone_path=str(tmp_path), entry_file=str(tmp_path / "absent.md")
            )
        )
        assert result.success is False
        assert result.error_message is not None


@pytest.mark.unit
class TestCutJobWiring:
    def test_the_node_is_a_packaged_entry_point(self) -> None:
        manifest = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text("utf-8"))
        nodes = manifest["project"]["entry-points"]["onex.nodes"]
        assert nodes[_NODE_NAME] == f"omnibase_infra.nodes.{_NODE_NAME}"

    def test_the_contract_names_the_handler_the_runtime_will_run(self) -> None:
        contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
        assert contract["handler"]["class"] == "HandlerReleaseNotesApply"
        assert "terminal_event" not in contract

    def test_the_cut_job_applies_notes_through_the_node(self) -> None:
        body = _WORKFLOW.read_text(encoding="utf-8")
        assert f"onex node {_NODE_NAME}" in body
        assert 'pathlib.Path("CHANGELOG.md")' not in body, (
            "an inline CHANGELOG.md write is the second writer that ignores the "
            "repo's .readme-only-docs.yaml"
        )
        assert "staged=(CHANGELOG.md)" not in body

    def test_the_cut_job_reads_the_staged_paths_without_masking_a_bad_receipt(
        self,
    ) -> None:
        body = _WORKFLOW.read_text(encoding="utf-8")
        assert "mapfile -t staged" in body
        assert "jq -er '(.result.handler_result // .result)" in body
        assert "select(.success == true)" in body
        assert "read -r -d '' -a staged < <(jq" not in body
        assert "release-notes-stderr.log" in body, (
            "a node that never reaches a receipt must still surface its stderr"
        )

    def test_the_node_runs_from_the_train_checkout_without_the_app_token(
        self,
    ) -> None:
        body = _WORKFLOW.read_text(encoding="utf-8")
        start = body.index(f"onex node {_NODE_NAME}")
        invocation = body[body.rindex("if ! ", 0, start) : start]
        assert "env -u GH_TOKEN -u GITHUB_TOKEN" in invocation
        assert '--project "${GITHUB_WORKSPACE}/_train"' in invocation
        assert "--frozen" in invocation

    def test_the_train_script_holds_no_second_writer(self) -> None:
        source = _TRAIN.read_text(encoding="utf-8")
        assert "apply_release_notes" not in source
        assert ".readme-only-docs.yaml" not in source
