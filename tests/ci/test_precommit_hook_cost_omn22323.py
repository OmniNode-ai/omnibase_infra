# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No commit hook is removed, delayed or loosened by the commit-cost change (OMN-22323).

The fixture lists every hook of ``origin/dev`` at the commit named in it, with the
stage it runs in and a digest of its ``exclude`` pattern. It lives in this repository so
the test runs on every runner and is never skipped. A hook retired on purpose is removed
from the fixture in the change that retires it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/precommit_hook_cost_omn22323/hooks_at_origin_dev.json"
#: Run in the commit stage; a hook that moves to any of these runs later than a commit.
_LATER_STAGES = frozenset({"pre-push", "manual", "post-commit", "post-checkout"})


def _digest(value: object) -> str | None:
    return (
        None if value is None else hashlib.sha256(str(value).encode()).hexdigest()[:12]
    )


def _hooks(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    default = config.get("default_stages")
    found: dict[str, list[dict[str, Any]]] = {}
    for repo in config["repos"]:
        for hook in repo.get("hooks", []):
            found.setdefault(hook["id"], []).append(
                {
                    "stages": hook.get("stages", default),
                    "exclude": _digest(hook.get("exclude")),
                }
            )
    return found


def problems(
    before: dict[str, list[dict[str, Any]]], after: dict[str, list[dict[str, Any]]]
) -> list[str]:
    """What the ``after`` config removed, delayed or loosened relative to ``before``."""
    found: list[str] = []
    for hook_id, old_rows in sorted(before.items()):
        new_rows = after.get(hook_id)
        if new_rows is None or len(new_rows) < len(old_rows):
            found.append(f"{hook_id}: hook removed")
            continue
        for old, new in zip(old_rows, new_rows, strict=False):
            if set(new["stages"] or []) & _LATER_STAGES - set(old["stages"] or []):
                found.append(
                    f"{hook_id}: stage moved later {old['stages']} -> {new['stages']}"
                )
            elif old["stages"] != new["stages"]:
                found.append(
                    f"{hook_id}: stages changed {old['stages']} -> {new['stages']}"
                )
            if old["exclude"] != new["exclude"]:
                found.append(f"{hook_id}: exclude pattern changed")
    return found


def _current() -> dict[str, list[dict[str, Any]]]:
    return _hooks(yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text()))


@pytest.mark.unit
def test_every_hook_of_origin_dev_still_runs_where_it_ran() -> None:
    before = json.loads(FIXTURE.read_text())["hooks"]
    assert len(before) > 100
    assert problems(before, _current()) == []


@pytest.mark.unit
def test_the_comparison_catches_a_removed_delayed_or_loosened_hook() -> None:
    before = json.loads(FIXTURE.read_text())["hooks"]

    def mutated(hook_id: str, **change: Any) -> dict[str, list[dict[str, Any]]]:
        after = json.loads(json.dumps(before))
        after[hook_id][0].update(change)
        return after

    removed = {k: v for k, v in before.items() if k != "topic-naming-lint"}
    assert problems(before, removed) == ["topic-naming-lint: hook removed"]
    later = mutated("topic-naming-lint", stages=["pre-push"])
    assert (
        "topic-naming-lint: stage moved later ['pre-commit'] -> ['pre-push']"
        in problems(before, later)
    )
    assert problems(before, mutated("topic-naming-lint", exclude="0" * 12)) == [
        "topic-naming-lint: exclude pattern changed"
    ]


@pytest.mark.unit
def test_topic_naming_lint_hands_the_staged_files_to_the_staged_entry() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook["id"] == "topic-naming-lint"
    )
    assert hook["entry"] == "bash scripts/validation/run_topic_lint.sh"
    script = (ROOT / "scripts/validation/run_topic_lint.sh").read_text()
    assert "staged_commit_hook_scans topic-naming-lint" in script
    assert hook["pass_filenames"] is True
    assert hook.get("stages") in (None, ["pre-commit"])
