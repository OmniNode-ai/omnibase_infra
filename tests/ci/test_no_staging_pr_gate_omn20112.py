# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No omnibase_infra pull-request gate may require the staging plane (OMN-20112).

Operator ruling 2026-09-29: staging is being turned off, and nothing outside
omninode_infra may require it. The env-parity workflow checked out omninode_infra
and failed a pull request unless the staging ConfigMap agreed with docker-compose,
so a contributor without access to omninode_infra could never pass it. A
compose-vs-manifest parity check belongs in omninode_infra, where the manifests
live.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
_PR_TRIGGERS = frozenset({"pull_request", "pull_request_target", "merge_group"})
_STAGING_MANIFEST_ROOT = "k8s/onex-dev"


def _triggers(workflow: dict[object, object]) -> frozenset[str]:
    # PyYAML reads the bare key `on` as the boolean True.
    raw = workflow.get("on", workflow.get(True))
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, list):
        return frozenset(str(item) for item in raw)
    if isinstance(raw, dict):
        return frozenset(str(key) for key in raw)
    return frozenset()


def test_env_parity_workflow_is_removed() -> None:
    assert not (WORKFLOWS / "env-parity.yml").exists()


def test_no_pull_request_workflow_reads_the_staging_manifests() -> None:
    offenders = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict):
            continue
        if _triggers(workflow) & _PR_TRIGGERS and _STAGING_MANIFEST_ROOT in (
            path.read_text(encoding="utf-8")
        ):
            offenders.append(path.name)
    assert offenders == []
