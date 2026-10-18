# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17427 - the hook push rehearsal caller matches the omnibase_core reusable it calls.

A controller worker's push runs this repository's pre-push hook set, and a hook change that makes
that push fail loses the worker's fix. The caller workflow runs the omnibase_core reusable
rehearsal on a pull request that changes a hook path. These tests hold the caller to the
reusable it pins: the pin is a full commit sha on omnibase_core dev, every input the caller
passes is declared by the reusable at that sha, and every required input is passed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLER = REPO_ROOT / ".github" / "workflows" / "hook-push-rehearsal.yml"
JOB_ID = "hook-push-rehearsal"
USES = re.compile(
    r"^OmniNode-ai/omnibase_core/\.github/workflows/hook-push-rehearsal\.yml@(?P<sha>[0-9a-f]{40})$"
)


def _caller_job() -> dict[str, Any]:
    document = yaml.safe_load(CALLER.read_text(encoding="utf-8"))
    return dict(document["jobs"][JOB_ID])


def _triggers() -> dict[str, Any]:
    document = yaml.safe_load(CALLER.read_text(encoding="utf-8"))
    # PyYAML reads the bare key `on` as the boolean True.
    return dict(document.get("on") or document[True])


def test_the_caller_runs_on_pull_requests_and_merge_queue_entries() -> None:
    assert set(_triggers()) == {"pull_request", "merge_group"}


def test_the_caller_job_reads_contents_and_has_no_runner_of_its_own() -> None:
    job = _caller_job()
    assert job["permissions"] == {"contents": "read"}
    assert "runs-on" not in job


def test_the_caller_pins_the_reusable_by_full_commit_sha() -> None:
    assert USES.match(_caller_job()["uses"]), _caller_job()["uses"]


def test_the_path_regex_matches_hook_paths_and_not_the_caller_itself() -> None:
    pattern = re.compile(_caller_job()["with"]["path-regex"])
    for changed in (
        ".pre-commit-config.yaml",
        ".pre-commit-hooks.yaml",
        ".pre-commit-hooks/reject-deploy-gate-skip-token.sh",
        "scripts/hooks/example.sh",
    ):
        assert pattern.search(changed), changed
    # The caller is not a hook: a hosted runner holds no sibling clones, so the pre-push hooks that
    # read them (architecture layers, deploy-scope DoD) refuse any push there, and a caller change
    # that matched its own regex would fail its own rehearsal.
    for unrelated in (
        "README.md",
        "src/omnibase_infra/utils/correlation.py",
        ".github/workflows/hook-push-rehearsal.yml",
    ):
        assert not pattern.search(unrelated), unrelated


def test_the_sample_file_is_a_tracked_python_source() -> None:
    sample = _caller_job()["with"]["sample-file"]
    assert sample.endswith(".py")
    assert (REPO_ROOT / sample).is_file()


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/hook_push_rehearsal_core_reusable_omn17427.json"
)
def test_the_caller_passes_what_the_recorded_reusable_declares(
    recorded_response: dict[str, Any],
) -> None:
    """Replay the reusable as omnibase_core holds it at the squash commit of omnibase_core#2025."""
    commit = recorded_response["commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert "origin/dev" in recorded_response["remote_branches_containing_commit"]
    assert USES.match(_caller_job()["uses"]).group("sha") == commit

    reusable = yaml.safe_load(recorded_response["workflow"])
    declared = reusable[True]["workflow_call"]["inputs"]
    passed = _caller_job()["with"]
    assert set(passed) <= set(declared), sorted(set(passed) - set(declared))
    required = {name for name, spec in declared.items() if spec.get("required")}
    assert required <= set(passed), sorted(required - set(passed))
    for name in passed:
        assert declared[name]["type"] == "string"
