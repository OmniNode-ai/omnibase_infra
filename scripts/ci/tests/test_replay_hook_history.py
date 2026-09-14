# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Focused tests for the OMN-17474 replay parser and summariser.

Scope is deliberately the two pure functions the measurement's honesty rests
on. A misparse that drops a `Failed` row turns a firing hook into a deletion
candidate, and a summariser that counts a skip as a passing run manufactures
the false zero the harness exists to avoid. Neither is caught by running the
harness, because both fail silently in the direction of a clean result.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "replay_hook_history.py"
_spec = importlib.util.spec_from_file_location("replay_hook_history", _MODULE_PATH)
assert _spec and _spec.loader
replay_hook_history = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations through sys.modules.
sys.modules["replay_hook_history"] = replay_hook_history
_spec.loader.exec_module(replay_hook_history)

CommitRecord = replay_hook_history.CommitRecord
HookRun = replay_hook_history.HookRun
load_hook_stages = replay_hook_history.load_hook_stages
parse_precommit_output = replay_hook_history.parse_precommit_output
summarise = replay_hook_history.summarise

# Captured verbatim from `pre-commit run --verbose --color never` 4.6.1.
TRANSCRIPT = """Always passes............................................................Passed
- hook id: always-pass
- duration: 0.01s
Always fails.............................................................Failed
- hook id: always-fail
- duration: 1.25s
- exit code: 1

BOOM
/Users/someone/thing.py:3: hardcoded path

Only python files....................................(no files to check)Skipped
- hook id: only-py
Trailing whitespace......................................................Failed
- hook id: trailing-whitespace
- duration: 0.08s
- exit code: 1

Fixing a.txt
"""


@pytest.mark.unit
def test_parse_reads_id_status_and_duration() -> None:
    runs = parse_precommit_output(TRANSCRIPT)
    by_id = {run.hook_id: run for run in runs}
    assert set(by_id) == {
        "always-pass",
        "always-fail",
        "only-py",
        "trailing-whitespace",
    }
    assert by_id["always-pass"].status == "passed"
    assert by_id["always-pass"].duration_s == 0.01
    assert by_id["always-fail"].status == "failed"
    assert by_id["always-fail"].exit_code == 1
    assert by_id["always-fail"].duration_s == 1.25


@pytest.mark.unit
def test_skipped_hook_is_neither_a_run_nor_a_fire() -> None:
    """A hook whose file filter matched nothing was never asked a question."""
    runs = parse_precommit_output(TRANSCRIPT)
    skipped = next(run for run in runs if run.hook_id == "only-py")
    assert skipped.status == "skipped"
    assert skipped.ran is False
    assert skipped.fired is False
    assert skipped.duration_s is None


@pytest.mark.unit
def test_a_fixer_that_rewrote_a_file_counts_as_a_fire() -> None:
    """pre-commit reports a fixer as Failed; it would have blocked the commit."""
    runs = parse_precommit_output(TRANSCRIPT)
    fixer = next(run for run in runs if run.hook_id == "trailing-whitespace")
    assert fixer.fired is True


@pytest.mark.unit
def test_hook_output_that_looks_like_a_status_line_invents_no_row() -> None:
    """Rows are keyed off `- hook id:`, so stray output cannot create one."""
    noisy = (
        "Real hook...............................................................Failed\n"
        "- hook id: real-hook\n"
        "- duration: 0.4s\n"
        "- exit code: 2\n"
        "\n"
        "Imitation line..........................................................Passed\n"
        "still just output, no hook id follows\n"
    )
    runs = parse_precommit_output(noisy)
    assert [run.hook_id for run in runs] == ["real-hook"]
    assert runs[0].exit_code == 2


@pytest.mark.unit
def test_empty_transcript_yields_no_rows() -> None:
    assert parse_precommit_output("") == []


@pytest.mark.unit
def test_summarise_counts_runs_fires_and_skips_separately() -> None:
    records = [
        CommitRecord(
            commit="a" * 40,
            committed_at="2026-09-01T00:00:00Z",
            stage="pre-commit",
            changed_files=2,
            wall_seconds=1.0,
            runs=[
                HookRun("quiet", "passed", 0.1, None),
                HookRun("noisy", "failed", 0.5, 1),
                HookRun("filtered", "skipped", None, None),
            ],
        ),
        CommitRecord(
            commit="b" * 40,
            committed_at="2026-09-02T00:00:00Z",
            stage="pre-commit",
            changed_files=1,
            wall_seconds=1.0,
            runs=[
                HookRun("quiet", "passed", 0.3, None),
                HookRun("noisy", "failed", 0.7, 1),
                HookRun("filtered", "skipped", None, None),
            ],
        ),
    ]
    stages = {"quiet": "pre-commit", "noisy": "pre-commit", "filtered": "pre-push"}
    by_id = {s.hook_id: s for s in summarise(records, stages)}

    assert by_id["quiet"].runs == 2
    assert by_id["quiet"].fires == 0
    assert by_id["quiet"].median_duration_s == 0.2

    assert by_id["noisy"].fires == 2
    assert by_id["noisy"].first_fire_commit == "a" * 40
    assert by_id["noisy"].last_fire_commit == "b" * 40

    # The load-bearing one: a hook that only ever skipped has a zero DENOMINATOR,
    # not a zero fire rate, and must never read as "ran twice, never fired".
    assert by_id["filtered"].runs == 0
    assert by_id["filtered"].skips == 2
    assert by_id["filtered"].median_duration_s is None
    assert by_id["filtered"].stage == "pre-push"


@pytest.mark.unit
def test_summarise_orders_by_fires_then_runs() -> None:
    records = [
        CommitRecord(
            commit="c" * 40,
            committed_at="2026-09-03T00:00:00Z",
            stage="pre-commit",
            changed_files=1,
            wall_seconds=1.0,
            runs=[
                HookRun("low", "passed", 0.1, None),
                HookRun("high", "failed", 0.1, 1),
            ],
        )
    ]
    ordered = [s.hook_id for s in summarise(records, {})]
    assert ordered == ["high", "low"]


@pytest.mark.unit
def test_load_hook_stages_applies_default_and_hook_level_stages(tmp_path: Path) -> None:
    """A hook-level `stages` beats `default_stages` -- the OMN-17468 asymmetry."""
    config = tmp_path / ".pre-commit-config.yaml"
    config.write_text(
        """
repos:
  - repo: local
    hooks:
      - id: inherits-default
        name: d
        entry: 'true'
        language: system
      - id: declares-push
        name: p
        entry: 'true'
        language: system
        stages: [pre-push]
      - id: commit-msg-only
        name: m
        entry: 'true'
        language: system
        stages: [commit-msg]
default_stages: [pre-commit]
""",
        encoding="utf-8",
    )
    stages = load_hook_stages(config)
    assert stages["inherits-default"] == "pre-commit"
    assert stages["declares-push"] == "pre-push"
    # Out of the measured tiers entirely; attributing it to one would be a lie.
    assert "commit-msg-only" not in stages


@pytest.mark.unit
def test_load_hook_stages_attributes_a_dual_stage_hook_to_the_commit_tier(
    tmp_path: Path,
) -> None:
    config = tmp_path / ".pre-commit-config.yaml"
    config.write_text(
        """
repos:
  - repo: local
    hooks:
      - id: both
        name: b
        entry: 'true'
        language: system
        stages: [pre-commit, pre-push]
""",
        encoding="utf-8",
    )
    assert load_hook_stages(config)["both"] == "pre-commit"


@pytest.mark.unit
def test_classify_fire_names_a_broken_machine_as_environment() -> None:
    """The web repo's `type-check` fired 83/83 on a missing toolchain.

    The second assertion changed with OMN-17474: an import error in a hook's
    output is no longer environmental on the strength of the words alone. A
    hook that CATCHES a missing import in the code under test prints the same
    sentence, so the text cannot separate the two. It is environmental only
    when the interpreter the hook names does not resolve, which the tool-
    presence rule decides and the assertion below exercises.
    """
    classify_fire = replay_hook_history.classify_fire
    assert classify_fire("Executable `pnpm` not found") == "environment"
    assert classify_fire("ModuleNotFoundError: No module named 'omnibase_core'") == (
        "finding"
    )
    assert (
        classify_fire(
            "ModuleNotFoundError: No module named 'omnibase_core'",
            exit_code=1,
            entry="definitely-not-a-real-interpreter-omn17474 -m check",
            language="system",
        )
        == "environment"
    )


@pytest.mark.unit
def test_classify_fire_defaults_to_finding() -> None:
    """Unrecognised and empty output stay findings -- the safe direction."""
    classify_fire = replay_hook_history.classify_fire
    assert classify_fire("src/x.py:3: hardcoded absolute path") == "finding"
    assert classify_fire("") == "finding"


# Captured verbatim from the OMN-17474 replay artifact for `omniclaude`,
# hook `onex-validate-links`, commit 0dacfbd9. This is a REAL CATCH: the hook
# found five broken relative documentation links. The substring classifier read
# `not found` inside the hook's own finding text and labelled it environmental,
# which excluded a live gate from both populations silently.
REAL_CATCH_BROKEN_LINKS = """Validating markdown links in: /clones/omniclaude
Check external: False
  BROKEN: [Skill Lifecycle](docs/architecture/skill-lifecycle.md) - Target file not found: docs/architecture/skill-lifecycle.md
  BROKEN: [QUICKSTART.md](QUICKSTART.md) - Target file not found: QUICKSTART.md
  OK: [CLAUDE.md](CLAUDE.md)
"""

# Captured verbatim from the same run, `omniweb` hook `type-check`. This is a
# MISSING TOOLCHAIN: pre-commit itself could not execute the hook's entry, and
# said so in its own words rather than the hook's.
REAL_MISSING_TOOLCHAIN = "Executable `pnpm` not found"


@pytest.mark.unit
def test_classify_fire_keeps_a_real_catch_that_says_not_found() -> None:
    """A finding whose own text contains `not found` is still a finding.

    This is the OMN-17474 classifier defect in one assertion: broken-link and
    missing-file findings describe themselves with the same words a broken
    machine does, so message text cannot separate them.
    """
    classify_fire = replay_hook_history.classify_fire
    assert (
        classify_fire(
            REAL_CATCH_BROKEN_LINKS,
            exit_code=1,
            entry="bash scripts/validate_links.sh",
            language="system",
            repo=Path(__file__).resolve().parents[3],
        )
        == "finding"
    )


@pytest.mark.unit
def test_classify_fire_keeps_a_missing_toolchain_environmental() -> None:
    """pre-commit's own non-execution report still classifies as environment."""
    classify_fire = replay_hook_history.classify_fire
    assert classify_fire(REAL_MISSING_TOOLCHAIN, exit_code=1) == "environment"


@pytest.mark.unit
def test_classify_fire_reads_exit_shape_not_text() -> None:
    """126/127 are the shell's reserved `could not execute` codes."""
    classify_fire = replay_hook_history.classify_fire
    assert classify_fire("anything at all", exit_code=127) == "environment"
    assert classify_fire("anything at all", exit_code=126) == "environment"
    assert classify_fire("anything at all", exit_code=1) == "finding"


@pytest.mark.unit
def test_classify_fire_reads_entry_tool_presence(tmp_path: Path) -> None:
    """An entry whose binary does not resolve means the hook never ran."""
    classify_fire = replay_hook_history.classify_fire
    assert (
        classify_fire(
            "src/x.py:3: hardcoded absolute path",
            exit_code=1,
            entry="definitely-not-a-real-binary-omn17474 --check",
            language="system",
            repo=tmp_path,
        )
        == "environment"
    )
    # A repo-relative script that is present resolves, so the fire is the code.
    (tmp_path / "check.sh").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    assert (
        classify_fire(
            "src/x.py:3: hardcoded absolute path",
            exit_code=1,
            entry="./check.sh",
            language="script",
            repo=tmp_path,
        )
        == "finding"
    )


@pytest.mark.unit
def test_classify_fire_ignores_entry_for_managed_languages(tmp_path: Path) -> None:
    """pre-commit installs `language: python` entries into its own env.

    `ruff` not being on the ambient PATH proves nothing about whether the hook
    ran, so the entry rule must not apply -- classifying it environmental would
    delete a live gate on no evidence.
    """
    classify_fire = replay_hook_history.classify_fire
    assert (
        classify_fire(
            "src/x.py:1:1: F401 unused import",
            exit_code=1,
            entry="ruff check --force-exclude",
            language="python",
            repo=tmp_path,
        )
        == "finding"
    )


@pytest.mark.unit
def test_load_hook_specs_carries_entry_and_language(tmp_path: Path) -> None:
    """Classification needs the hook's entry, which `stages` alone never had."""
    config = tmp_path / ".pre-commit-config.yaml"
    config.write_text(
        """
repos:
  - repo: local
    hooks:
      - id: local-check
        entry: bash scripts/check.sh
        language: system
        stages: [pre-commit]
  - repo: https://example.invalid/remote
    rev: v1
    hooks:
      - id: remote-check
        stages: [pre-commit]
""",
        encoding="utf-8",
    )
    specs = replay_hook_history.load_hook_specs(config)
    assert specs["local-check"].entry == "bash scripts/check.sh"
    assert specs["local-check"].language == "system"
    assert specs["local-check"].stage == "pre-commit"
    # A remote hook declares its entry in the remote repo, not here.
    assert specs["remote-check"].entry is None


@pytest.mark.unit
def test_reclassify_artifact_reports_class_changes(tmp_path: Path) -> None:
    """The classification step re-runs over an artifact without replaying."""
    artifact = tmp_path / "omniclaude.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "omn17474.hook_replay.v1",
                "repo": "omniclaude",
                "environment_suspect_fire_hooks": ["onex-validate-links"],
                "fire_samples": {
                    "onex-validate-links": [
                        {
                            "commit": "0dacfbd9",
                            "exit_code": 1,
                            "output": REAL_CATCH_BROKEN_LINKS,
                        }
                    ],
                    "type-check": [
                        {
                            "commit": "a9b5153f",
                            "exit_code": 1,
                            "output": REAL_MISSING_TOOLCHAIN,
                        }
                    ],
                },
                "summaries": [
                    {"hook_id": "onex-validate-links", "runs": 111, "fires": 1},
                    {"hook_id": "type-check", "runs": 83, "fires": 83},
                    {"hook_id": "trailing-whitespace", "runs": 111, "fires": 0},
                ],
                "zero_fire_hooks": ["trailing-whitespace"],
            }
        ),
        encoding="utf-8",
    )
    result = replay_hook_history.reclassify_artifact(artifact)
    assert result["environment_suspect_before"] == ["onex-validate-links"]
    assert result["environment_suspect_after"] == ["type-check"]
    assert result["became_finding"] == ["onex-validate-links"]
    assert result["became_environment"] == ["type-check"]
    # A hook that never fired has nothing to classify, so no zero-fire
    # candidate can change class. This is what makes the delete list stable.
    assert result["zero_fire_hooks_changed"] == []
