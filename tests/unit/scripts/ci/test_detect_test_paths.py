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


def test_integration_test_change_selects_its_own_directory() -> None:
    # OMN-15245. This test previously asserted `paths == []` on the premise
    # that "the integration job runs all integration tests on every PR anyway".
    # That premise is false: no CI job runs all of tests/integration/ on a PR.
    # `test-parallel` is the only job that runs it, and in smart-selection mode
    # it runs ONLY selected_paths -- so dropping integration paths here meant a
    # changed integration module was never collected on its own PR (recorded
    # live on OMN-15263 / omnibase_infra#2504).
    changed_files = ["tests/integration/nodes/test_foo.py"]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    assert paths == ["tests/integration/nodes/"]


def test_ci_process_change_selects_ci_tests() -> None:
    changed_files = [
        ".github/workflows/ci.yml",
        "scripts/ci/ci_summary_gate.py",
        "config/runner_routing_policy.yaml",
    ]
    paths = resolve_test_paths(changed_files, adjacency_path=ADJ)
    # tests/ci/ for the CI-process mapping, plus (OMN-15245) the two families
    # that exercise scripts/ — scripts/ci/ci_summary_gate.py is a scripts/ file.
    assert paths == ["tests/ci/", "tests/scripts/", "tests/unit/scripts/"]


def test_workflow_only_change_selects_ci_tests_alone() -> None:
    # No scripts/ file in this diff → no scripts test families.
    paths = resolve_test_paths(
        [".github/workflows/ci.yml", "config/runner_routing_policy.yaml"],
        adjacency_path=ADJ,
    )
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
    # An unclassified, non-doc change (no src/, tests/, scripts/, CI-process,
    # or docs mapping) has no test mapping → conservative fallback.
    # NOTE (OMN-15245): this used to be probed with a scripts/*.sh path, which
    # is no longer unclassified — scripts/ now maps to the tests that actually
    # exercise it. Probed here with a docker catalog manifest instead.
    selection = compute_selection(
        changed_files=["docker/catalog/services/some_new_service.yaml"],
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
            "docker/catalog/services/some_new_service.yaml",
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


# ---------------------------------------------------------------------------
# OMN-15245: changed-test coverage invariant (fail-closed)
#
# Invariant: any CHANGED path under tests/ is covered by the emitted selection.
# Narrowing may add tests, never drop a test file the diff itself touched.
#
# Two recorded live instances are replayed verbatim below. Both produced a green
# CI run that never collected the changed test modules:
#   * OMN-15218 / omnibase_infra#2493 -- scripts/ + tests/scripts/ diff selected
#     ["tests/unit/"] (22053 tests, none of them the 47 new ones).
#   * OMN-15263 / omnibase_infra#2504 -- six-test-file diff selected
#     ["tests/ci/"] (run 30296123866, Detect Changes job 90082641865); the five
#     changed tests/integration/** modules were never collected on the very PR
#     that existed to repair them.
# ---------------------------------------------------------------------------

# Recorded diff, OMN-15218 / omnibase_infra#2493.
OMN_15218_DIFF = [
    "scripts/preflight_lane_deploy_attribution.py",
    "scripts/deploy-runtime.sh",
    "scripts/runtime_build/refresh_stability_lane.sh",
    "tests/scripts/test_preflight_lane_deploy_attribution.py",
    "tests/scripts/test_deploy_runtime_lane_attribution.py",
]

# Recorded diff, OMN-15263 / omnibase_infra#2504.
OMN_15263_DIFF = [
    "tests/ci/test_compose_required_env_coverage.py",
    "tests/integration/docker/test_docker_integration.py",
    "tests/integration/infra/test_dev_runtime_compose_render.py",
    "tests/integration/infra/test_judge_compose_render.py",
    "tests/integration/infra/test_prod_runtime_compose_render.py",
    "tests/integration/infra/test_stability_test_runtime_compose_render.py",
]


def _is_collected_by(selected_paths: list[str], changed_path: str) -> bool:
    """True when pytest, given `selected_paths`, would collect `changed_path`."""
    return any(changed_path.startswith(sel) for sel in selected_paths)


def test_omn15218_scripts_diff_selects_the_changed_test_modules() -> None:
    selection = compute_selection(
        changed_files=OMN_15218_DIFF,
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    uncollected = [
        p
        for p in OMN_15218_DIFF
        if p.startswith("tests/") and not _is_collected_by(selection.selected_paths, p)
    ]
    assert not uncollected, (
        f"changed test files dropped by narrowing: {uncollected} "
        f"(selection={selection.selected_paths})"
    )
    # The recorded fail-open output, asserted as a non-result.
    assert selection.selected_paths != ["tests/unit/"]


def test_omn15263_test_file_diff_selects_the_changed_integration_modules() -> None:
    selection = compute_selection(
        changed_files=OMN_15263_DIFF,
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    # tests/integration/docker/ is excluded by design (see the docker carve-out
    # test below); every other changed test path must be collected.
    expected_collected = [
        p for p in OMN_15263_DIFF if not p.startswith("tests/integration/docker/")
    ]
    uncollected = [
        p
        for p in expected_collected
        if not _is_collected_by(selection.selected_paths, p)
    ]
    assert not uncollected, (
        f"changed test files dropped by narrowing: {uncollected} "
        f"(selection={selection.selected_paths})"
    )
    assert "tests/ci/" in selection.selected_paths
    assert "tests/integration/infra/" in selection.selected_paths
    # The recorded fail-open output, asserted as a non-result.
    assert selection.selected_paths != ["tests/ci/"]


@pytest.mark.parametrize(
    "changed_test_path",
    [
        "tests/scripts/test_some_script.py",
        "tests/integration/runtime/test_some_seam.py",
        "tests/nodes/test_some_node.py",
        "tests/services/test_some_service.py",
        "tests/replay/test_some_replay.py",
        "tests/audit/test_some_audit.py",
        "tests/redeploy/test_some_redeploy.py",
        "tests/ci/test_some_ci_gate.py",
        "tests/unit/cli/test_some_cli.py",
    ],
)
def test_changed_test_file_is_never_dropped_by_narrowing(
    changed_test_path: str,
) -> None:
    selection = compute_selection(
        changed_files=[changed_test_path],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert _is_collected_by(selection.selected_paths, changed_test_path), (
        f"{changed_test_path} not collected by {selection.selected_paths}"
    )


def test_changed_test_file_survives_alongside_a_source_change() -> None:
    # Mixed diff: the source mapping and the changed-test coverage must BOTH
    # be present -- the test file is additive, not a replacement.
    changed = [
        "src/omnibase_infra/cli/foo.py",
        "tests/integration/runtime/test_some_seam.py",
    ]
    selection = compute_selection(
        changed_files=changed,
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert "tests/unit/cli/" in selection.selected_paths
    assert _is_collected_by(
        selection.selected_paths, "tests/integration/runtime/test_some_seam.py"
    )


def test_changed_docker_integration_test_is_not_selected() -> None:
    # Documented carve-out, NOT a fail-open: tests/integration/docker/ is
    # --ignore'd by BOTH pytest steps in ci.yml (smart and full suite), so
    # selecting it can never make it run -- it would only make pytest exit 5
    # ("no tests ran") when it is the sole selected path. That family has its
    # own gate: docker-build.yml runs tests/integration/docker/ on a paths
    # filter that includes tests/integration/docker/**.
    selection = compute_selection(
        changed_files=["tests/integration/docker/test_docker_integration.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert "tests/integration/docker/" not in selection.selected_paths
    # Falls back to the conservative unit root rather than emitting nothing.
    assert selection.selected_paths == ["tests/unit/"]


def test_test_file_at_tests_root_escalates_to_full_suite() -> None:
    # A changed test path directly under tests/ cannot be narrowed below
    # "tests/" itself. Emitting "tests/" as a 1-split smart selection would run
    # the whole suite on one shard with the smart step's timeouts; escalate to
    # the real 15-split full suite instead.
    selection = compute_selection(
        changed_files=["tests/test_compose_profile_teardown_policy.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.CHANGED_TEST_UNNARROWABLE
    assert selection.split_count == 15


def test_markdown_under_tests_does_not_force_a_test_selection() -> None:
    # The coverage invariant is about executable test modules. A .md file under
    # tests/ is still provably inert (OMN-14753 docs-only exemption).
    selection = compute_selection(
        changed_files=["tests/replay/README.md"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.selected_paths == []


# ---------------------------------------------------------------------------
# OMN-15245: scripts/ family mapping
# ---------------------------------------------------------------------------


def test_scripts_change_selects_the_tests_that_exercise_scripts() -> None:
    # scripts/ in this repo holds deploy-path and governance-guard code. Its
    # tests live in tests/scripts/ and tests/unit/scripts/; before OMN-15245 a
    # scripts/ change mapped to neither (it fell through to the blanket
    # tests/unit/ fallback, which exercises none of them).
    selection = compute_selection(
        changed_files=["scripts/deploy-runtime.sh"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert "tests/scripts/" in selection.selected_paths
    assert "tests/unit/scripts/" in selection.selected_paths


def test_scripts_ci_change_still_selects_ci_process_tests() -> None:
    # No regression on the existing CI-process mapping: scripts/ci/ keeps
    # selecting tests/ci/, and now also picks up the scripts test families.
    selection = compute_selection(
        changed_files=["scripts/ci/detect_test_paths.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert "tests/ci/" in selection.selected_paths
    assert "tests/unit/scripts/" in selection.selected_paths


# ---------------------------------------------------------------------------
# OMN-15245: no-regression -- narrowing still works for pure src diffs
# ---------------------------------------------------------------------------


def test_pure_src_diff_still_narrows_to_module_tests() -> None:
    selection = compute_selection(
        changed_files=["src/omnibase_infra/cli/foo.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.full_suite_reason is None
    assert selection.selected_paths == ["tests/unit/cli/"]
    assert selection.split_count == 1


def test_pure_src_diff_with_reverse_deps_still_narrows() -> None:
    selection = compute_selection(
        changed_files=["src/omnibase_infra/services/foo.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is False
    assert selection.selected_paths == sorted(
        f"tests/unit/{m}/"
        for m in ("services", "adapters", "dlq", "handlers", "runtime")
    )
    # The whole tests/ tree is NOT selected -- narrowing is still real.
    assert "tests/" not in selection.selected_paths
    assert not any(p.startswith("tests/integration/") for p in selection.selected_paths)


def test_shared_module_escalation_unchanged_by_coverage_invariant() -> None:
    # A shared-module diff that ALSO changes a test file still escalates to the
    # full suite -- escalation outranks the additive coverage rule.
    selection = compute_selection(
        changed_files=[
            "src/omnibase_infra/models/foo.py",
            "tests/integration/runtime/test_some_seam.py",
        ],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason == EnumFullSuiteReason.SHARED_MODULE
