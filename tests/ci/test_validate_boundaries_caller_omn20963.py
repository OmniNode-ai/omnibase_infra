# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20963: migration-conflict-check calls omnibase_core's validate-boundaries action.

The job keeps its id, name and check-run context and changes only its ``uses:``
line. The recorded artifact holds what the two implementations did on a lab
host over the same scratch trees, so the claim that the repoint changes no
verdict rests on bytes captured from the real handlers, not on a stub.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
JOB_ID = "migration-conflict-check"
CONTEXT = "Cross-Repo Migration Conflicts"
ACTION = "OmniNode-ai/omnibase_core/.github/actions/validate-boundaries"
# First eight hex digits of the omnibase_core#1942 merge; the rest must still be a full sha.
ACTION_SHA_PREFIX = "de3398d1"
_PINNED_USES = re.compile(rf"^{re.escape(ACTION)}@{ACTION_SHA_PREFIX}[0-9a-f]{{32}}$")
_RETIRED_ACTION = re.compile(
    r"onex_change_control/\.github/actions/validate-boundaries"
)


def _job() -> dict[str, Any]:
    workflow = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    return workflow["jobs"][JOB_ID]


def _retired_references(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if _RETIRED_ACTION.search(line)]


def test_job_calls_the_core_action_at_the_merge_sha() -> None:
    uses = [s["uses"].split(" ")[0] for s in _job()["steps"] if "uses" in s]

    assert [u for u in uses if _PINNED_USES.fullmatch(u)] != []


def test_no_workflow_calls_the_retired_change_control_action() -> None:
    offenders = {
        path.name: _retired_references(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOWS.glob("*.yml"))
    }

    assert {name: refs for name, refs in offenders.items() if refs} == {}


def test_retired_action_reference_is_detected() -> None:
    old = (
        "      - uses: OmniNode-ai/onex_change_control/.github/actions/"
        "validate-boundaries@main"
    )

    assert _retired_references(old) == [old.strip()]


def test_inputs_and_job_name_are_unchanged() -> None:
    job = _job()
    step = next(s for s in job["steps"] if ACTION in s.get("uses", ""))

    assert job["name"] == CONTEXT
    assert step["with"] == {
        "checks": "migration-conflicts",
        "repos": "omniclaude,omnidash,omniintelligence,omnibase_core,omnimemory",
        "warn-only": "false",
    }


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn20963_migration_conflicts_core_vs_occ.json"
)
def test_recorded_core_handler_gives_the_same_verdict_as_the_retired_script(
    recorded_response: dict[str, Any],
) -> None:
    runs = recorded_response["response"]
    clean = runs["clean_tree"]
    control = runs["control_without_warn_only"]

    assert clean["omnibase_core"] == clean["onex_change_control"]
    assert (
        control["conflicting_migration_omnibase_core"]
        == (control["conflicting_migration_onex_change_control"])
    )
    # Positive control: the injected migration changes the output and the count,
    # so identical verdicts above cannot come from a handler that ignores its input.
    unedited = control["unedited_tree_omnibase_core"]
    edited = control["conflicting_migration_omnibase_core"]
    assert edited["conflicts_reported"] == unedited["conflicts_reported"] + 1
    assert edited["normalized_sha256"] != unedited["normalized_sha256"]
    assert unedited["normalized_sha256"] == clean["omnibase_core"]["normalized_sha256"]


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn20963_migration_conflicts_core_vs_occ.json"
)
def test_recorded_core_handlers_pass_a_clean_tree_and_fail_a_drifted_one(
    recorded_response: dict[str, Any],
) -> None:
    runs = recorded_response["response"]
    parity = runs["boundary_parity"]
    migrations = runs["control_without_warn_only"]
    called_pins = [
        s["uses"].split(" ")[0].partition("@")[2]
        for s in _job()["steps"]
        if s.get("uses", "").startswith(f"{ACTION}@")
    ]

    # The recorded handlers are the ones the job calls: the pin ci.yml names is the
    # pin they were recorded against, so a job still on the retired action fails here.
    assert called_pins == [runs["handler_source_pin"]]
    assert parity["clean_tree"]["exit_code"] == 0
    assert parity["clean_tree"]["mismatches"] == 0
    assert parity["drifted_topic_tree"]["exit_code"] != 0
    assert parity["drifted_topic_tree"]["mismatches"] == 1
    assert parity["drifted_topic_tree"]["error"].startswith(
        "topic not found in producer"
    )
    assert runs["clean_tree"]["omnibase_core"]["exit_code"] == 0
    assert migrations["conflicting_migration_omnibase_core"]["exit_code"] != 0
