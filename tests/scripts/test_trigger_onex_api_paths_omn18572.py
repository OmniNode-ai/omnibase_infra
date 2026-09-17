# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``docker/onex-api`` is lane STATE for the dev lane (OMN-18572).

MEASURED 2026-09-17. ``omninode_infra#1523`` fixed
``ModelKafkaClientConfig``'s missing SASL fields -- the defect that made tenant
bootstrap on the ``.201`` dev lane impossible -- and merged at 09:00:53Z. At
11:15Z the lane was still running ``onex-lab/omnicloud-core:f37261c2-...``, the
PARENT of that squash: ``git merge-base --is-ancestor 4528c42c f37261c2`` exits
1 and the reverse exits 0. Tenant creation on the lab was impossible for the
whole window and two lanes lost a morning to it.

WHY NOTHING FIRED, TWICE OVER. ``omninode_infra`` has no rebuild trigger of any
kind, which OMN-18572's caller workflow closes. And even once it calls this
publisher, the canonical deploy-gate classifier does not match these paths: its
``RUNTIME_PATH_PATTERNS`` carry ``docker/Dockerfile*`` and
``docker/**/*.Dockerfile``, neither of which matches ``docker/onex-api/Dockerfile``
(three segments against a two-segment pattern, and no ``.Dockerfile`` suffix),
and nothing at all matches ``docker/onex-api/main.py``. That classifier is right
to miss them -- its question is "does this PR need deploy EVIDENCE" for the
CLOUD plane, and the cloud onex-api image is built and pinned by an entirely
separate workflow.

This trigger asks the wider question OMN-18072 already separated out: does this
merge change what the LANE RUNS. The dev lane runs ``onex-api`` from
``${ONEX_API_IMAGE}``, built by the lab-overlay applier from exactly this
directory in the omninode_infra overlay tree. It answers yes.

NO FALSE POSITIVE IN THIS REPOSITORY. ``docker/onex-api`` does not exist in
``omnibase_infra``, so adding the pattern cannot make one of this repo's own
merges rebuild for a path it does not have. The negative controls below assert
that rather than assume it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"

#: The changed-file list of the omninode_infra merge that produced no delivery.
OMN_18561_FILES = [
    "docker/onex-api/kafka_client_config.py",
    "docker/onex-api/tests/test_kafka_client_config.py",
]


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("_trigger_rebuild_omn18572", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trigger_rebuild_omn18572"] = module
    spec.loader.exec_module(module)
    return module


def test_the_merge_that_produced_no_delivery_now_matches() -> None:
    module = _load_module()
    assert module.find_lane_state_paths(OMN_18561_FILES) == OMN_18561_FILES


def test_the_dockerfile_and_the_app_source_both_match() -> None:
    """Both halves, because the image changes when either does."""
    module = _load_module()
    files = ["docker/onex-api/Dockerfile", "docker/onex-api/main.py"]
    assert module.find_lane_state_paths(files) == files


def test_the_workflow_contracts_file_matches() -> None:
    """Deny-by-default workflow types decide what the lane's API will accept."""
    module = _load_module()
    files = ["docker/onex-api/workflow-contracts.yaml"]
    assert module.find_lane_state_paths(files) == files


def test_a_neighbouring_docker_directory_does_not_match() -> None:
    """Negative control: the pattern is this directory, not all of docker/.

    Without this, a pattern typo widening to ``docker/**`` would pass every
    assertion above while making every documentation change in that tree cost a
    full dev-lane rebuild.
    """
    module = _load_module()
    assert module.find_lane_state_paths(["docker/onex-api-notes/README.md"]) == []
    assert module.find_lane_state_paths(["docker/omniweb/Dockerfile"]) == []


def test_the_pattern_names_no_path_this_repository_has() -> None:
    """Negative control: this addition cannot re-trigger omnibase_infra itself."""
    assert not (_REPO_ROOT / "docker" / "onex-api").exists(), (
        "docker/onex-api now exists in omnibase_infra, so this pattern would "
        "start matching this repository's own merges. That is a decision to "
        "make deliberately, not a test to update."
    )


def test_the_pattern_is_declared_rather_than_derived() -> None:
    """The list is the readable artifact; a match found by accident is not policy."""
    module = _load_module()
    assert "docker/onex-api/**" in module.LANE_STATE_PATH_PATTERNS
