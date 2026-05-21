# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest

from scripts.ci.detect_test_paths import compute_selection, resolve_test_paths
from scripts.ci.test_selection_models import EnumFullSuiteReason

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
ADJ = REPO_ROOT / "scripts/ci/test_selection_adjacency.yaml"


# ---------------------------------------------------------------------------
# resolve_test_paths — direct path resolution
# ---------------------------------------------------------------------------


def test_single_module_change_resolves_to_one_test_dir() -> None:
    changed_files = ["src/omnibase_infra/cli/foo.py"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert paths == ["tests/unit/cli/"]


def test_test_only_change_runs_only_changed_test_dir() -> None:
    changed_files = ["tests/unit/nodes/test_foo.py"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert paths == ["tests/unit/nodes/"]


def test_integration_test_only_change_does_not_select_unit_tests() -> None:
    # Integration test changes do not contribute to unit-job selection;
    # the integration job runs all integration tests on every PR anyway.
    changed_files = ["tests/integration/nodes/test_foo.py"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert paths == []


def test_unknown_source_path_produces_no_selection() -> None:
    # Files outside src/ and tests/unit/ — no unit-test mapping.
    changed_files = ["docs/README.md", ".github/workflows/foo.yml"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert paths == []


def test_leaf_module_change_expands_to_its_reverse_deps() -> None:
    # `diagnostics` has no reverse deps — only its own unit tests run.
    changed_files = ["src/omnibase_infra/diagnostics/foo.py"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert paths == ["tests/unit/diagnostics/"]


def test_services_module_expands_to_reverse_deps() -> None:
    # services is imported by adapters, dlq, handlers, runtime
    # (protocols removed: tests/unit/protocols/ does not exist)
    changed_files = ["src/omnibase_infra/services/foo.py"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    expected = sorted(
        f"tests/unit/{m}/"
        for m in ("services", "adapters", "dlq", "handlers", "runtime")
    )
    assert paths == expected


# ---------------------------------------------------------------------------
# compute_selection — escalation logic
# ---------------------------------------------------------------------------


def test_shared_module_change_escalates_to_full_suite() -> None:
    selection = compute_selection(
        changed_files=["src/omnibase_infra/models/foo.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.SHARED_MODULE
    assert selection.split_count == 15
    assert selection.matrix == list(range(1, 16))


def test_test_infrastructure_change_escalates_to_full_suite() -> None:
    selection = compute_selection(
        changed_files=["tests/conftest.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.TEST_INFRASTRUCTURE


def test_pyproject_toml_escalates_to_full_suite() -> None:
    selection = compute_selection(
        changed_files=["pyproject.toml"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.TEST_INFRASTRUCTURE


def test_threshold_module_count_escalates() -> None:
    # 6 distinct non-shared modules changed → THRESHOLD_MODULES.
    changed_files = [
        f"src/omnibase_infra/{m}/x.py"
        for m in ["cli", "clients", "configs", "decorators", "docker", "gateways"]
    ]
    selection = compute_selection(
        changed_files=changed_files,
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.THRESHOLD_MODULES


def test_main_branch_always_full_suite() -> None:
    selection = compute_selection(
        changed_files=["src/omnibase_infra/cli/x.py"],
        adjacency_path=ADJ,
        ref_name="main",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.MAIN_BRANCH


def test_small_change_returns_smart_selection_no_reason() -> None:
    selection = compute_selection(
        changed_files=["src/omnibase_infra/cli/foo.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.full_suite_reason is None
    assert "tests/unit/cli/" in selection.selected_paths
    assert 1 <= selection.split_count <= 5
    assert selection.matrix == list(range(1, selection.split_count + 1))


def test_no_matching_files_falls_back_to_unit_root() -> None:
    # Doc-only change has no unit-test mapping → conservative fallback.
    selection = compute_selection(
        changed_files=["docs/something.md"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.selected_paths == ["tests/unit/"]
    assert selection.split_count == 1


def test_feature_flag_off_returns_full_suite() -> None:
    selection = compute_selection(
        changed_files=["src/omnibase_infra/cli/foo.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
        feature_flag_enabled=False,
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.FEATURE_FLAG_OFF
    assert selection.split_count == 15
    assert selection.matrix == list(range(1, 16))


def test_schedule_event_escalates_to_full_suite() -> None:
    selection = compute_selection(
        changed_files=["src/omnibase_infra/cli/foo.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
        event_name="schedule",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.SCHEDULED
    assert selection.split_count == 15
    assert selection.matrix == list(range(1, 16))


def test_merge_group_event_escalates_to_full_suite() -> None:
    selection = compute_selection(
        changed_files=["src/omnibase_infra/cli/foo.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
        event_name="merge_group",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.MERGE_GROUP
    assert selection.split_count == 15
    assert selection.matrix == list(range(1, 16))


def test_full_suite_split_count_is_15() -> None:
    """Infra uses 15 splits (not 40 like core)."""
    selection = compute_selection(
        changed_files=["src/omnibase_infra/models/foo.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.split_count == 15
    assert len(selection.matrix) == 15
