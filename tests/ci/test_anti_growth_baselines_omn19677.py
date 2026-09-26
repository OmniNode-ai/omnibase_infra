# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wiring tests for OMN-19677's shrink-only debt baselines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import ci_summary_gate

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
PIN = "3faefdcd5750b05f69d2efcb8cf200bc359483c3"
REUSABLE = (
    f"OmniNode-ai/omniclaude/.github/workflows/anti-growth-baseline-reusable.yml@{PIN}"
)
OMNICLAUDE_REPO = "https://github.com/OmniNode-ai/omniclaude"

BASELINES = (
    (
        "noncanonical-class-allowlist-oneway",
        "Noncanonical Class Allowlist One-way (OMN-19677)",
        ".onex_ratchets/noncanonical_class_allowlist.yaml",
        "yaml-list:allowlist",
        "anti-growth-noncanonical-class-allowlist",
    ),
    (
        "topic-naming-baseline-oneway",
        "Topic Naming Baseline One-way (OMN-19677)",
        "scripts/validation/topic_naming_baseline.txt",
        "line-set",
        "anti-growth-topic-naming-baseline",
    ),
    (
        "validator-requirements-baseline-oneway",
        "Validator Requirements Baseline One-way (OMN-19677)",
        "architecture-handshakes/validator-requirements-baseline.yaml",
        "yaml-list:gaps",
        "anti-growth-validator-requirements-baseline",
    ),
    (
        "runtime-profiles-allowlist-oneway",
        "Runtime Profiles Allowlist One-way (OMN-19677)",
        "config/validation/runtime_profiles_allowlist.yaml",
        "yaml-list:allowlist",
        "anti-growth-runtime-profiles-allowlist",
    ),
    (
        "skip-count-baseline-oneway",
        "Skip Count Baseline One-way (OMN-19677)",
        "config/skip_count_baseline.yaml",
        "yaml-counted-list-map:suites",
        "anti-growth-skip-count-baseline",
    ),
)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _omniclaude_repo() -> dict[str, Any]:
    repos = _load_yaml(PRECOMMIT_CONFIG)["repos"]
    matches = [repo for repo in repos if repo.get("repo") == OMNICLAUDE_REPO]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    ("job_id", "name", "baseline_path", "parser", "alias"),
    BASELINES,
    ids=[row[0] for row in BASELINES],
)
def test_shrink_only_baseline_is_wired_locally_and_in_ci(
    job_id: str,
    name: str,
    baseline_path: str,
    parser: str,
    alias: str,
) -> None:
    job = _load_yaml(CI_WORKFLOW)["jobs"][job_id]
    assert job["name"] == name
    assert job["uses"] == REUSABLE
    assert job["with"] == {"baseline-path": baseline_path, "parser": parser}
    assert not ({"if", "needs", "secrets"} & job.keys())
    assert job["permissions"] == {"contents": "read"}

    assert f"{name} / anti-growth-baseline" in ci_summary_gate.STRICT_GATE_JOBS

    repo = _omniclaude_repo()
    ci_pin = str(job["uses"]).rsplit("@", 1)[1]
    assert repo["rev"] == ci_pin == PIN
    hooks = [hook for hook in repo["hooks"] if hook.get("alias") == alias]
    assert len(hooks) == 1
    hook = hooks[0]
    assert hook["id"] == "anti-growth-baseline"
    assert hook["args"] == [
        "--baseline",
        baseline_path,
        "--parser",
        parser,
        "--base-ref",
        "origin/dev",
    ]
    assert re.fullmatch(str(hook["files"]), baseline_path)
    assert (REPO_ROOT / baseline_path).is_file()


def test_ci_and_precommit_share_one_immutable_omniclaude_pin() -> None:
    workflow_text = CI_WORKFLOW.read_text(encoding="utf-8")
    ci_pins = re.findall(
        r"anti-growth-baseline-reusable\.yml@([0-9a-f]{40})", workflow_text
    )
    assert len(ci_pins) == len(BASELINES)

    precommit_pin = str(_omniclaude_repo()["rev"])
    assert re.fullmatch(r"[0-9a-f]{40}", precommit_pin)
    assert set(ci_pins) == {precommit_pin}
