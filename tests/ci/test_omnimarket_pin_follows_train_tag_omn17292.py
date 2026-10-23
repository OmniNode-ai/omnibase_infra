# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: the omnimarket contract pin follows the release train's tag.

The refresh used to run on a 6-hourly cron, resolve omnimarket ``dev`` to a
commit sha and open a new timestamped-branch PR each time it moved, so the
open sha-bump PRs stacked and each cost CI on every head. The pin now names an
omnimarket ``dev-<UTC>-<sha12>`` tag, a scheduled nightly release-train run
is the only automatic start of the refresh, the refresh picks the newest ``dev-*`` tag and
never a branch tip's sha, and it keeps one PR on one fixed branch.

Falsifier: restore the cron trigger, the ``commits/dev`` sha resolution, the
timestamped branch or a sha in the pin file, and the matching test names it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
REFRESH = WORKFLOWS / "omnimarket-contract-pin-refresh.yml"
TRAIN = WORKFLOWS / "release-train-nightly.yml"
DEV_TAG = re.compile(r"^dev-[0-9]{14}-[0-9a-f]{12}$")


def _load(path: Path) -> dict[object, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _triggers(document: dict[object, object]) -> dict[str, object]:
    # `on:` parses as the boolean True under YAML 1.1.
    triggers = document.get("on", document.get(True))
    assert isinstance(triggers, dict)
    return triggers


def _refresh_run_text() -> str:
    job = _load(REFRESH)["jobs"]["refresh"]  # type: ignore[index]
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


def test_pin_and_sibling_pin_name_a_release_train_tag() -> None:
    pin = _load(REPO_ROOT / ".github" / "omnimarket-contract-pin.yaml")
    sibling = _load(REPO_ROOT / ".github" / "sibling-pins.yaml")
    ref = pin["omnimarket_contract_ref"]
    assert isinstance(ref, str) and DEV_TAG.fullmatch(ref), ref
    assert sibling["pins"]["omnimarket"] == ref  # type: ignore[index]


def test_refresh_has_no_schedule_of_its_own() -> None:
    triggers = _triggers(_load(REFRESH))
    assert set(triggers) == {"workflow_run", "workflow_dispatch"}, (
        "only the nightly release train (workflow_run) or an operator "
        f"(workflow_dispatch) may move the pin, got {sorted(triggers)}"
    )


def test_only_a_scheduled_nightly_train_run_starts_the_refresh() -> None:
    train = _load(TRAIN)
    assert "schedule" in _triggers(train)
    workflow_run = _triggers(_load(REFRESH))["workflow_run"]
    assert workflow_run == {
        "workflows": [train["name"]],
        "types": ["completed"],
        "branches": ["dev"],
    }
    job = _load(REFRESH)["jobs"]["refresh"]  # type: ignore[index]
    # The dependency-advance dispatch of the train is not an overnight run.
    assert " ".join(job["if"].split()) == (
        "github.event_name == 'workflow_dispatch' || "
        "github.event.workflow_run.event == 'schedule'"
    )


def test_refresh_resolves_the_newest_dev_tag_never_a_branch_sha() -> None:
    run = _refresh_run_text()
    assert "commits/dev" not in run
    assert "commits/${target_ref}" not in run
    assert "git ls-remote --tags" in run
    assert "refs/tags/dev-*" in run
    assert "^dev-[0-9]{14}-[0-9a-f]{12}$" in run


def test_refresh_keeps_one_pr_on_a_fixed_branch() -> None:
    run = _refresh_run_text()
    assert 'BRANCH="bot/omnimarket-contract-tag"' in run
    assert "date -u" not in run, "a timestamped branch stacks one PR per run"
    assert 'gh pr list --base dev --head "$BRANCH" --state open' in run
    assert "gh pr edit" in run
    assert "gh pr create" in run
