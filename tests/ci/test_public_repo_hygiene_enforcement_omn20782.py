# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20782: omnibase_infra enforces the five hygiene classes on added lines.

The public-repo hygiene gate blocks a class only when the repository's own
``.public-repo-hygiene.yaml`` lists it in ``enforce_classes`` (and, for the
added-lines scope, declares ``enforce_scope: added-lines``). Until this change
the file declared neither, so the gate could not fail a pull request. The
caller workflow and the pre-commit hook must sit on the same omniclaude
revision, which must carry the added-lines scope (omniclaude 9a58ef705) and the
env-var vocabulary path (omniclaude 6e47aa311).

Falsifier: restore the previous config, pin, or drop the pre-commit hook and the
named test fails.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXTERNAL_SWEEP_EXCLUSIONS,
    evaluate_external_sweep,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / ".public-repo-hygiene.yaml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "public-repo-hygiene.yml"
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"

# omniclaude dev 6e47aa311, a descendant of 9a58ef705 (added-lines scope).
GATE_REV = "6e47aa311ac5d177d116d9597b07966a233d2463"
# The pin before this change: it carries the added-lines scope but not the
# env-var vocabulary path the exported pre-commit hook reads.
PREVIOUS_REV = "3b8addff365b1a53e04106284b9275b6d5fad3c6"
FIVE_CLASSES = {
    "private-repo-name",
    "internal-kb-prose",
    "lab-config",
    "person-name",
    "private-network",
}
REUSABLE = "OmniNode-ai/omniclaude/.github/workflows/public-repo-hygiene-reusable.yml"
HYGIENE_CONTEXT = "public-repo-hygiene / public-repo-hygiene"
NOW = datetime(2026, 10, 9, 18, 0, 0, tzinfo=UTC)


def _config() -> dict[str, object]:
    loaded = yaml.safe_load(CONFIG.read_text())
    assert isinstance(loaded, dict)
    return loaded


def test_config_enforces_the_five_classes_on_added_lines() -> None:
    config = _config()
    assert config.get("enforce_scope") == "added-lines"
    classes = config.get("enforce_classes")
    assert isinstance(classes, list)
    assert set(classes) == FIVE_CLASSES
    assert len(classes) == len(FIVE_CLASSES)


def test_caller_workflow_is_pinned_to_the_added_lines_revision() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    uses = workflow["jobs"]["public-repo-hygiene"]["uses"]
    assert uses == f"{REUSABLE}@{GATE_REV}"
    assert PREVIOUS_REV not in WORKFLOW.read_text()
    assert workflow["jobs"]["public-repo-hygiene"]["secrets"] == "inherit"


def _hygiene_hook_blocks() -> list[dict[str, object]]:
    precommit = yaml.safe_load(PRECOMMIT.read_text())
    blocks = []
    for repo in precommit["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "public-repo-hygiene":
                blocks.append({"repo": repo, "hook": hook})
    return blocks


def test_precommit_runs_the_gate_hook_at_the_same_revision() -> None:
    blocks = _hygiene_hook_blocks()
    assert len(blocks) == 1, "exactly one public-repo-hygiene hook is declared"
    repo = blocks[0]["repo"]
    hook = blocks[0]["hook"]
    assert isinstance(repo, dict) and isinstance(hook, dict)
    assert repo["repo"] == "https://github.com/OmniNode-ai/omniclaude"
    assert repo["rev"] == GATE_REV
    # The exported hook already runs over the staged added lines (--diff-staged),
    # always_run, in the pre-commit stage. An override here would narrow it.
    narrowing = {"entry", "args", "files", "exclude", "stages", "always_run"}
    assert not narrowing & set(hook), sorted(narrowing & set(hook))


def test_hygiene_context_is_not_excluded_from_the_ci_summary_sweep() -> None:
    """A red gate must fail the CI Summary umbrella: no exclusion admits it."""
    assert not [k for k in EXTERNAL_SWEEP_EXCLUSIONS if "hygiene" in k.lower()]


def _row(conclusion: str) -> dict[str, object]:
    return {
        "id": 20782,
        "name": HYGIENE_CONTEXT,
        "status": "completed",
        "conclusion": conclusion,
        "started_at": "2026-10-09T17:00:00Z",
        "completed_at": "2026-10-09T17:05:00Z",
        "head_sha": "a" * 40,
    }


def _sweep(conclusion: str) -> list[str]:
    failures, _in_flight, swept, _excluded, _provisional = evaluate_external_sweep(
        [_row(conclusion)],
        expected=(),
        in_run_names=frozenset(),
        self_name="CI Summary",
        exclusions=EXTERNAL_SWEEP_EXCLUSIONS,
        events={},
        now=NOW,
    )
    assert swept == [HYGIENE_CONTEXT], "the sweep must have judged the row"
    return failures


def test_a_red_hygiene_gate_fails_the_ci_summary_sweep_and_green_passes() -> None:
    assert _sweep("failure") == [f"{HYGIENE_CONTEXT} (failure)"]
    assert _sweep("success") == []


@pytest.mark.live_contact(
    "tests/ci/fixtures/public_repo_hygiene_added_lines_omn20782.json"
)
def test_gate_at_the_pin_fails_a_planted_lab_config_literal(
    recorded_response: dict[str, object],
) -> None:
    """The recorded run of the real gate, at the revision both callers use."""
    recording = cast("dict[str, object]", recorded_response["response"])
    assert recording["gate_rev"] == GATE_REV
    assert recording["exit_code"] == 1
    assert "1 enforced finding(s) on lines this change adds" in cast(
        "str", recording["enforce_scope_line"]
    )
    assert [
        f.rsplit(": ", 1)[-1] for f in cast("list[str]", recording["added_findings"])
    ] == ["lab-config"]
    assert recording["gate_rev"] in WORKFLOW.read_text()
    assert recording["gate_rev"] in PRECOMMIT.read_text()
