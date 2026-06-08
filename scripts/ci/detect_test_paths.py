# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Change-aware test path resolution for omnibase_infra CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.ci.test_selection_loader import (
    ModelAdjacencyMap,
    load_adjacency_map,
)
from scripts.ci.test_selection_models import (
    EnumFullSuiteReason,
    ModelTestSelection,
)

SRC_PREFIX = "src/omnibase_infra/"

# Repo root resolved relative to this file (scripts/ci/detect_test_paths.py),
# never a hardcoded absolute path.
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_UNIT_PREFIX = "tests/unit/"
TEST_INTEGRATION_PREFIX = "tests/integration/"

FULL_SUITE_BRANCHES = {"main"}

# Full suite uses 15 splits (infra CI split count)
_FULL_SUITE_SPLIT_COUNT = 15


def resolve_test_paths(
    changed_files: list[str],
    adjacency_path: Path,
) -> list[str]:
    """Map changed file paths to deterministic UNIT test directories.

    Behavior:
      - Source changes under src/omnibase_infra/<module>: include
        tests/unit/<module>/.
      - Test-only changes under tests/unit/: include the changed unit-test directory.
      - Test-only changes under tests/integration/: ignored (integration runs always).
      - Files outside src/ and tests/unit/: no contribution; caller decides
        whether to escalate to full suite.

    Adjacency expansion maps each changed module to its reverse dependents,
    ensuring downstream tests run when a shared module changes.
    """
    config = load_adjacency_map(adjacency_path)
    return _resolve(changed_files, config)


def _resolve(
    changed_files: list[str],
    config: ModelAdjacencyMap,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    direct_modules: set[str] = set()
    selected: set[str] = set()

    for path in changed_files:
        if path.startswith(SRC_PREFIX):
            module = path[len(SRC_PREFIX) :].split("/", 1)[0]
            if module in config.adjacency:
                direct_modules.add(module)
        elif path.startswith(TEST_UNIT_PREFIX):
            parts = path.split("/")
            if len(parts) >= 3:
                selected.add(f"{TEST_UNIT_PREFIX}{parts[2]}/")

    expanded: set[str] = set(direct_modules)
    for module in direct_modules:
        expanded.update(config.adjacency[module].reverse_deps)

    for module in expanded:
        selected.add(f"{TEST_UNIT_PREFIX}{module}/")

    # Drop selected directories that do not exist on disk. A module in the
    # adjacency map (e.g. `dlq`) may have source under src/ but no
    # corresponding tests/unit/<module>/ directory; passing a missing path to
    # pytest aborts collection with exit code 5 ("no tests ran"). Filtering to
    # existing directories keeps the gate honest for any zone whose reverse
    # dependents include a test-less module.
    return sorted(p for p in selected if (repo_root / p).is_dir())


def compute_selection(
    changed_files: list[str],
    adjacency_path: Path,
    ref_name: str,
    event_name: str = "pull_request",
    feature_flag_enabled: bool = True,
) -> ModelTestSelection:
    config = load_adjacency_map(adjacency_path)

    # 0. Feature flag short-circuit: off → legacy 15-split full suite.
    if not feature_flag_enabled:
        return _full_suite(EnumFullSuiteReason.FEATURE_FLAG_OFF)

    # 1. Branch / event escalation.
    if ref_name in FULL_SUITE_BRANCHES:
        return _full_suite(EnumFullSuiteReason.MAIN_BRANCH)
    if event_name == "merge_group":
        return _full_suite(EnumFullSuiteReason.MERGE_GROUP)
    if event_name == "schedule":
        return _full_suite(EnumFullSuiteReason.SCHEDULED)

    # 2. Test infrastructure escalation.
    for changed in changed_files:
        if any(
            changed == infra or changed.startswith(infra.rstrip("/") + "/")
            for infra in config.test_infrastructure_paths
        ):
            return _full_suite(EnumFullSuiteReason.TEST_INFRASTRUCTURE)

    # 3. Shared module escalation.
    changed_modules = {
        path[len(SRC_PREFIX) :].split("/", 1)[0]
        for path in changed_files
        if path.startswith(SRC_PREFIX)
    } & set(config.adjacency.keys())
    if changed_modules & set(config.shared_modules):
        return _full_suite(EnumFullSuiteReason.SHARED_MODULE)

    # 4. Threshold escalation: too many distinct modules.
    if len(changed_modules) >= config.thresholds.modules_changed_for_full_suite:
        return _full_suite(EnumFullSuiteReason.THRESHOLD_MODULES)

    # 5. Smart selection.
    selected = _resolve(changed_files, config)
    if not selected:
        # Conservative one-shard fallback over the full tests/unit/ tree. This
        # is NOT a no-op — it runs ~3-5 min of unit tests. It fires for changes
        # that have no unit-test mapping (doc-only, workflow-only, integration-
        # only). Per Selector Truth Boundary: safer to run something than nothing.
        selected = ["tests/unit/"]
    split_count = _split_count_for(selected)

    return ModelTestSelection(
        selected_paths=selected,
        split_count=split_count,
        is_full_suite=False,
        full_suite_reason=None,
        matrix=list(range(1, split_count + 1)),
    )


def _full_suite(reason: EnumFullSuiteReason) -> ModelTestSelection:
    return ModelTestSelection(
        selected_paths=["tests/"],
        split_count=_FULL_SUITE_SPLIT_COUNT,
        is_full_suite=True,
        full_suite_reason=reason,
        matrix=list(range(1, _FULL_SUITE_SPLIT_COUNT + 1)),
    )


def _split_count_for(selected_paths: list[str]) -> int:
    """Conservative heuristic mapping selected path count to split count.

    Thresholds keep small PRs on a single shard (cheap) while preventing
    pathologically slow runs when many paths survive selection.
    Infra has a smaller test suite than core, so the ceiling is 5 splits
    (vs core's 5 — same cap, smaller absolute counts per split).
    """
    n = len(selected_paths)
    if n <= 2:
        return 1
    if n <= 5:
        return 2
    if n <= 10:
        return 3
    if n <= 16:
        return 4
    return 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve change-aware test paths")
    parser.add_argument(
        "--changed-files-from",
        type=Path,
        required=True,
        help="Path to a file with one changed-file path per line.",
    )
    parser.add_argument("--ref-name", required=True)
    parser.add_argument("--event-name", default="pull_request")
    parser.add_argument(
        "--adjacency",
        type=Path,
        default=Path(__file__).parent / "test_selection_adjacency.yaml",
    )
    parser.add_argument(
        "--feature-flag",
        choices=("on", "off"),
        default="on",
        help="When 'off', emit a FEATURE_FLAG_OFF full-suite selection regardless of changed files.",
    )
    args = parser.parse_args(argv)

    changed = [
        line.strip()
        for line in args.changed_files_from.read_text().splitlines()
        if line.strip()
    ]
    selection = compute_selection(
        changed_files=changed,
        adjacency_path=args.adjacency,
        ref_name=args.ref_name,
        event_name=args.event_name,
        feature_flag_enabled=(args.feature_flag == "on"),
    )
    sys.stdout.write(selection.model_dump_json())
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
