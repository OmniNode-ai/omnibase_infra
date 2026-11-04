# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The agent_status_events drift proof must execute, not be collected and skipped.

The sharded test splits deselect ``ephemeral_pg`` and the Migration Integration
Test job selects it by path. A proof that sat in the skipping OMN-15376 module
instead would add two skipped ids per run, which the shrink-only skip baseline
(OMN-18776, OMN-19677) refuses. These checks hold the placement: out of the
skipping module and the baseline, deselected from the splits by marker, and in a
module the job that runs it names.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PROOF = (
    "tests/integration/migrations/test_102_tenant_projection_writer_role_omn15425.py"
)
PROOF_TESTS = (
    "test_unfixed_agent_actions_create_is_red_on_a_drifted_agent_status_events",
    "test_agent_actions_create_converges_a_drifted_agent_status_events",
)
SKIPPING_MODULE = (
    REPO_ROOT
    / "tests/integration/migrations/test_node_migration_shape_drift_omn15376.py"
)
CI_WORKFLOW = REPO_ROOT / ".github/workflows/ci.yml"
BASELINE = REPO_ROOT / "config/skip_count_baseline.yaml"


def test_each_proof_test_is_deselected_from_the_splits_by_marker() -> None:
    tree = ast.parse((REPO_ROOT / PROOF).read_text(encoding="utf-8"))
    marked = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(d) == "pytest.mark.ephemeral_pg" for d in node.decorator_list
        )
    }
    assert set(PROOF_TESTS) <= marked
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert workflow.count("and not ephemeral_pg") >= 2


def test_the_migration_integration_job_executes_the_proof_module_by_path() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    step = re.search(
        r"EphemeralPostgres live-apply proofs \(OMN-19550\).*?-m ephemeral_pg",
        workflow,
        re.S,
    )
    assert step is not None
    assert PROOF in step.group(0)
    assert 'ONEX_MIGRATION_PROOF_REQUIRE_PG: "1"' in step.group(0)


def test_the_skipping_module_carries_no_omn21920_case() -> None:
    assert "OMN-21920" not in SKIPPING_MODULE.read_text(encoding="utf-8")


def test_the_skip_baseline_names_no_omn21920_id() -> None:
    assert "OMN-21920" not in BASELINE.read_text(encoding="utf-8")
