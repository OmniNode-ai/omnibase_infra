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


def test_ci_process_change_selects_ci_tests() -> None:
    changed_files = [
        ".github/workflows/ci.yml",
        "scripts/ci/ci_summary_gate.py",
        "config/runner_routing_policy.yaml",
    ]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert paths == ["tests/ci/"]


def test_ci_test_change_selects_ci_tests() -> None:
    paths = resolve_test_paths(
        ["tests/ci/test_ci_summary_gate.py"],
        adjacency_path=ADJ,
    )
    assert paths == ["tests/ci/"]


def test_unknown_source_path_produces_no_selection() -> None:
    # Files outside src/ and tests/unit/ — no unit-test mapping.
    changed_files = ["docs/README.md"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert paths == []


def test_leaf_module_change_expands_to_its_reverse_deps() -> None:
    # `diagnostics` has no reverse deps — only its own unit tests run.
    changed_files = ["src/omnibase_infra/diagnostics/foo.py"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert paths == ["tests/unit/diagnostics/"]


def test_services_module_expands_to_reverse_deps() -> None:
    # services is imported by adapters, dlq, handlers, runtime.
    # All four reverse-dep test dirs exist on disk (tests/unit/dlq/ was added by
    # the DLQ overlay work, OMN-12634), so every reverse dep is selected. Only
    # existing test directories are emitted — a missing path would abort pytest
    # collection with exit code 5.
    changed_files = ["src/omnibase_infra/services/foo.py"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    expected = sorted(
        f"tests/unit/{m}/"
        for m in ("services", "adapters", "dlq", "handlers", "runtime")
    )
    assert paths == expected
    assert "tests/unit/dlq/" in paths


def test_missing_test_directories_are_filtered_out() -> None:
    # Regression: a module present in the adjacency map (e.g. `dlq`) may have
    # source under src/ but no tests/unit/<module>/ directory. Such paths must
    # never be emitted, or pytest exits 5 ("no tests ran") and blocks the gate.
    changed_files = ["src/omnibase_infra/services/foo.py"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert all((REPO_ROOT / p).is_dir() for p in paths), paths


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


def test_no_matching_non_doc_files_falls_back_to_unit_root() -> None:
    # An unclassified, non-doc change (no src/, tests/unit/, CI-process, or
    # docs mapping) has no unit-test mapping → conservative fallback.
    selection = compute_selection(
        changed_files=["scripts/some_new_uncategorized_tool.sh"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.selected_paths == ["tests/unit/"]
    assert selection.split_count == 1


# ---------------------------------------------------------------------------
# Docs-only exemption (OMN-14753 regression coverage)
# ---------------------------------------------------------------------------


def test_docs_only_markdown_change_selects_nothing() -> None:
    # Reproduces the reported bug: a single new .md file under docs/runbooks/
    # must NOT map to selected_paths=['tests/unit/'] (the full unit tree).
    selection = compute_selection(
        changed_files=["docs/runbooks/some-new-runbook.md"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.full_suite_reason is None
    assert selection.selected_paths == []
    assert selection.split_count == 1
    assert selection.matrix == [1]


def test_docs_only_top_level_markdown_selects_nothing() -> None:
    # A top-level markdown file (e.g. CLAUDE.md) not under docs/ is still
    # provably documentation by its .md suffix.
    selection = compute_selection(
        changed_files=["CLAUDE.md"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.selected_paths == []


def test_multiple_docs_only_files_select_nothing() -> None:
    selection = compute_selection(
        changed_files=[
            "docs/runbooks/foo.md",
            "docs/architecture/bar.md",
            "README.md",
        ],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.selected_paths == []


def test_docs_plus_shared_module_change_still_escalates() -> None:
    # A mixed diff (docs + a shared-module source file) must NOT take the
    # docs-only exemption -- shared-module escalation still applies.
    selection = compute_selection(
        changed_files=["docs/runbooks/foo.md", "src/omnibase_infra/models/x.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.SHARED_MODULE


def test_docs_plus_unclassified_code_change_falls_back_not_exempt() -> None:
    # Mixed diff: docs + an unrelated, unclassified non-doc path. Not ALL
    # files are docs, so the exemption must not fire; the conservative
    # tests/unit/ fallback still applies (ambiguous changes still escalate).
    selection = compute_selection(
        changed_files=[
            "docs/runbooks/foo.md",
            "scripts/some_new_uncategorized_tool.sh",
        ],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.selected_paths == ["tests/unit/"]


def test_docs_under_test_infrastructure_path_still_escalates() -> None:
    # A markdown file under a test-infrastructure directory (tests/fixtures/)
    # is ambiguous/shared by path, not provably inert -- test-infrastructure
    # escalation (checked before the docs-only exemption) still wins.
    selection = compute_selection(
        changed_files=["tests/fixtures/README.md"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.TEST_INFRASTRUCTURE


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
