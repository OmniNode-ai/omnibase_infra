# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Change-aware test path resolution for omnibase_infra CI."""

from __future__ import annotations

import argparse
import fnmatch
import math
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

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
TESTS_PREFIX = "tests/"
SCRIPTS_PREFIX = "scripts/"
# The two directories that actually exercise `scripts/`: the hermetic script
# tests (tests/scripts/) and the unit-tree mirror (tests/unit/scripts/).
#
# OMN-18833: these two are a FLOOR, not the population. They are a convention
# about where script tests are *supposed* to live, and the tree does not obey
# it: 51 modules outside these two prefixes read a `scripts/**` path off disk
# and assert its contents. omnibase_infra#3829 changed
# `scripts/deploy-runners.sh`; the three tests it broke live in
# `tests/unit/observability/runner_health/`; PR CI selected the five paths
# these prefixes produce, all green, and dev went red. The prefixes are kept
# (a script test that does live here is still selected without a reference),
# and `script_reference_test_paths` below adds the tests that reference the
# changed script, derived from the tree at selection time rather than from a
# map that would go stale the next time a test is written somewhere new.
SCRIPTS_TEST_PREFIXES = ("tests/scripts/", "tests/unit/scripts/")

# Directories the reference scan and the population walk never descend into.
# Neither can hold a test module the suite would collect from source, and both
# can carry stale copies that would corrupt the basename-uniqueness denominator.
_SCAN_SKIP_DIR_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".git", ".venv", "node_modules"}
)
CI_PROCESS_TEST_PATHS = (
    ".github/workflows/",
    "scripts/ci/",
    "config/runner_routing_policy.yaml",
)

# --- The CI-contract class (OMN-16745) -------------------------------------
#
# RULING -- what proof a `.github/workflows`-only diff actually requires.
#
# Necessary and sufficient: the CI-contract class, `tests/ci/` -- the
# workflow-shape, required-context and gate-wiring tests that read
# `.github/workflows/**` off disk and assert its contents (e.g.
# test_ci_workflow_resilience.py, test_merge_hold_gate_omn15484.py,
# test_omn_16878_enforcement_wiring.py) -- plus, when the diff also touches a
# test module, that module itself.
#
# NOT sufficient, and not even relevant: the Python unit suite. No test under
# `tests/unit/` has an outcome a workflow YAML edit can change; escalating a
# workflow diff to it is cost without proof, which is exactly what trains
# operators and agents to reach for a bypass (OMN-16346 recorded ~20 refused
# pushes and zero bypasses over a diff of this class, stranded for want of a
# host with headroom for a suite that could not have falsified the change).
#
# NOT permissible either: selecting nothing. "Workflow YAML cannot break Python
# tests, therefore skip" is the wrong inference -- workflow files break the
# ENFORCEMENT of tests, which is worse and invisible. OMN-15541 is the live
# counterexample: `ci.yml` hardcoded `pytest src/omnibase_compat/tests/` while
# the selector and pyproject named different roots, so full-suite escalation
# collected ZERO of the top-level `tests/` tree -- a fail-OPEN safety net
# produced by a workflow edit. A renamed job id silently drops a required
# status check; a changed `on:` trigger disables a gate. So this class is
# positively named and always NON-EMPTY, and its suite is asserted to be
# populated and workflow-aware by
# tests/unit/scripts/ci/test_detect_test_paths.py (Operating Rule #5: a class
# that is defined but never runs anything is a regression, not a fix).
#
# This is a CLASSIFICATION, not a weakening. Nothing here narrows a diff that
# touches any other path class: a workflow file alongside a shared module still
# escalates SHARED_MODULE, alongside test infrastructure still escalates
# TEST_INFRASTRUCTURE, and alongside an ordinary source file rides additively
# with that file's own narrowing. There is no env override, no allowlist
# mapping workflow paths to zero tests, and no bypass token anywhere in this
# change (CLAUDE.md Operating Rules #4 and #10).
CI_CONTRACT_TEST_ROOT = "tests/ci/"

# Mirrors `[tool.pytest.ini_options] python_files` in pyproject.toml, held equal
# by tests/unit/scripts/ci/test_detect_test_paths.py. Widening pytest's
# collection patterns without widening this must fail a test rather than
# silently misclassify a newly-collectable module as unnarrowable.
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")


def is_collectable_test_file_name(name: str) -> bool:
    """True when pytest would collect a file with this name (``python_files``)."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in TEST_FILE_PATTERNS)


# OMN-15336 item 4 repair follow-up: the vendored node-migration tree lives
# under neither src/, scripts/, nor tests/, so a change there (a new
# migration .sql, its _ledger row, or the FORCE-RLS fence/grandfather
# manifests) produced NO selection at all and fell through to the
# conservative tests/unit/ fallback -- which does not contain
# tests/scripts/test_node_migration_fence_parity.py. That test is the ratchet
# guarding against a future FORCE-RLS migration being laundered onto the
# grandfather snapshot; it is unreachable by the everyday change-aware
# selector on exactly the class of change that would breach it (verified:
# a grandfather-manifest + new .sql + ledger-row diff selected only
# tests/unit/ before this mapping existed).
#
# ADDITIVE, not a swap (2026-08-05 fix-forward). The first cut of this mapping
# added ONLY "tests/scripts/". Because `compute_selection`'s conservative
# fallback (`if not selected: selected = ["tests/unit/"]`) only fires when
# `_resolve()` returns nothing at all, giving migration-tree changes their own
# non-empty selection SUPPRESSED that fallback -- an ordinary migration diff
# (new .sql + ledger row, no YAML) went from selecting the whole tests/unit/
# tree to selecting tests/scripts/ ONLY. That is a real coverage regression,
# not a narrowing-to-something-equivalent swap like the scripts/ mapping
# above: tests/unit/migrations/, tests/unit/topology/, test_schema_fingerprint,
# test_db_ownership, and test_adversarial_fingerprint_drift all live under
# tests/unit/ (outside tests/unit/scripts/) and genuinely exercise migration
# .sql/ledger changes -- unlike scripts/, where tests/unit/ never covered the
# code plain-blanket-fallback was standing in for. So this branch selects
# BOTH tests/scripts/ (the fence-parity ratchet) AND tests/unit/ (the
# pre-existing real coverage) rather than trading one for the other. Any
# future prefix branch added here must make the same "does the blanket
# tests/unit/ fallback carry real coverage for this path class?" check before
# assuming a narrower, targeted selection is safe to swap in -- the
# fallback-suppression trap in `compute_selection` (a non-empty `_resolve()`
# result silently defeats the safety net for the whole diff, not just the
# part the new branch understands) is structural, not specific to migrations.
MIGRATION_TREE_PREFIX = "docker/migrations/forward/"

# OMN-15410: pytest roots that live NEXT TO the code they cover instead of
# under tests/. They are collected by the full suite (pyproject.toml
# `testpaths`), but the full suite is only one of two pytest steps — a
# NARROWED smart-selection run reaches nothing it is not explicitly told to
# reach. Without these mappings the four roots would be "collected" in the
# weakest possible sense: exercised only when something else escalated the
# job to full suite. Keys are source prefixes, values are the test roots a
# change under that prefix must run. Over-selection here is safe (extra tests
# run); under-selection is the OMN-15378 false-green class.
#
# Every value MUST also appear in pyproject.toml `testpaths`, and every
# non-`tests` testpaths entry MUST appear as a value here — both directions
# are asserted by scripts/validation/validate_test_root_collection.py.
COLLOCATED_TEST_ROOTS: dict[str, str] = {
    # Broadest first is irrelevant (all matches apply), but note scripts/tests/
    # covers the seed/keycloak scripts that live directly under scripts/, so it
    # is mapped from the whole scripts/ tree, matching SCRIPTS_TEST_PREFIXES.
    "scripts/": "scripts/tests/",
    "scripts/ci/": "scripts/ci/tests/",
    "scripts/runtime_build/": "scripts/runtime_build/tests/",
    "src/omnibase_infra/services/observability/agent_actions/": (
        "src/omnibase_infra/services/observability/agent_actions/tests/"
    ),
}

# Test families the change-aware pytest job structurally cannot run, so
# selecting one can never make it execute -- it would only make pytest exit 5
# ("no tests ran") when it is the sole selected path, reddening the gate without
# running anything. This is NOT a narrowing carve-out: the FULL suite excludes
# these identically, and each has its own dedicated gate.
#   * tests/integration/docker/ -- `--ignore`d by BOTH pytest steps in
#     .github/workflows/ci.yml; covered by docker-build.yml, whose paths filter
#     includes tests/integration/docker/**.
#   * tests/chaos/ and tests/performance/ -- deselected by the job's marker
#     expression (-m "not slow and not chaos and not kafka and not performance"),
#     which applies to the full suite too.
UNRUNNABLE_TEST_PREFIXES = (
    "tests/integration/docker/",
    "tests/chaos/",
    "tests/performance/",
)

# Positive-evidence documentation classification (OMN-14753). A path matching
# either of these can never contain executable code or fixture data, so it
# cannot influence any test outcome. This is narrower and stronger than "no
# unit-test mapping" (the conservative tests/unit/ fallback in
# `compute_selection`) -- it only exempts a diff when every changed file is
# affirmatively provable as prose/documentation, not merely unclassified.
DOCS_ONLY_SUFFIXES = (".md",)
DOCS_ONLY_PREFIXES = ("docs/",)


def _is_docs_only_path(path: str) -> bool:
    """True when `path` is documentation that cannot affect any test."""
    return path.endswith(DOCS_ONLY_SUFFIXES) or path.startswith(DOCS_ONLY_PREFIXES)


def _is_covered_by(selected: set[str] | list[str], path: str) -> bool:
    """True when pytest, given `selected`, would collect `path`."""
    return any(path.startswith(prefix) for prefix in selected)


def _changed_test_paths(changed_files: list[str]) -> list[str]:
    """Changed paths under tests/ that the selector is obliged to cover.

    Excludes documentation (provably inert, OMN-14753) and the families the
    pytest job structurally cannot run (`UNRUNNABLE_TEST_PREFIXES`).
    """
    return [
        path
        for path in changed_files
        if path.startswith(TESTS_PREFIX)
        and not _is_docs_only_path(path)
        and not path.startswith(UNRUNNABLE_TEST_PREFIXES)
    ]


def _root_level_changed_test_paths(changed_files: list[str]) -> list[str]:
    """Changed `.py` paths sitting directly in the `tests/` root."""
    return [
        path
        for path in _changed_test_paths(changed_files)
        if path.count("/") == 1 and path.endswith(".py")
    ]


def _requires_unnarrowable_full_suite(changed_files: list[str]) -> bool:
    """True when a root-level `tests/` module genuinely cannot be narrowed.

    A path sitting directly in the tests/ root has no containing directory
    other than `tests/`, and emitting `tests/` as a *smart* selection would run
    the whole suite under the smart step's split count and timeouts (OMN-15245).

    OMN-16745 splits that population in two on POSITIVE evidence, because the
    original rule conflated "has no containing directory" with "cannot be
    narrowed", and those are different claims:

      * A module pytest COLLECTS (``python_files``) is narrowable -- to itself.
        `_resolve` emits it at file grain, which is strictly narrower than the
        `tests/` directory this escalation existed to avoid emitting, and
        strictly covers the changed module. No escalation.
      * A module pytest does NOT collect (``tests/infrastructure_config.py``, a
        shared helper) is genuinely unnarrowable: handing it to pytest collects
        nothing (exit 5), and any suite in the tree may import it, so its blast
        radius really is the whole tree. Still escalates -- fail-closed.
    """
    return any(
        not is_collectable_test_file_name(path.rsplit("/", 1)[1])
        for path in _root_level_changed_test_paths(changed_files)
    )


def _uncovered_changed_test_dirs(
    changed_files: list[str],
    selected: set[str],
) -> set[str]:
    """Directories that must be added so every changed test path is collected.

    Additive only: a changed test path already covered by an existing selection
    contributes nothing. Root-level test modules are never emitted as `tests/`
    here: `_resolve` already selected the collectable ones at file grain, and
    `compute_selection` already escalated on the non-collectable ones
    (OMN-16745), so both halves are covered before this runs.
    """
    extra: set[str] = set()
    for path in _changed_test_paths(changed_files):
        parent = path.rsplit("/", 1)[0] + "/"
        if parent == TESTS_PREFIX:
            continue
        if _is_covered_by(selected | extra, path):
            continue
        extra.add(parent)
    return extra


FULL_SUITE_BRANCHES = {"main"}

# Full suite uses 15 splits (infra CI split count)
_FULL_SUITE_SPLIT_COUNT = 15

# The ceiling a NARROWED selection may reach, one below the full-suite count.
#
# This gap is load-bearing and is not a rounding choice. scripts/hooks/
# prepush_remote_verify.py's binding 3 decides whether a green CI run was the
# full suite by reading the shard DENOMINATOR out of the job names
# ("Tests (Split i/N)"), because that denominator is produced by CI from the
# pushed tree rather than supplied by the caller -- a forge-resistant witness of
# `is_full_suite`. That inference is sound only while no narrowed run can mint
# the full-suite denominator. Sizing shards by population (OMN-18542) would
# otherwise let a selection covering the whole tree reach 15 and become
# indistinguishable from a real full-suite run by job name alone.
#
# The cost of reserving 15 is bounded and measured: the widest narrowed
# selection observed (omnibase_infra#3652's ten paths, 2,178 of 2,231
# collectable modules) lands 156 modules on a shard at 14 shards against the
# full-suite path's 149 -- 4.7% denser, inside the budget's headroom by a wide
# margin. Raising this to `_FULL_SUITE_SPLIT_COUNT` therefore buys ~5% and
# silently collapses a pre-push safety binding; it is pinned in both repos'
# tests so it cannot be done by accident.
_MAX_NARROWED_SPLIT_COUNT = _FULL_SUITE_SPLIT_COUNT - 1

# Directories a filesystem walk must never descend into when counting the
# collectable population. Neither can contain a module pytest would collect from
# source, and both can carry stale copies that would inflate the count.
_UNCOUNTED_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".git"})


# The roots pytest itself collects, i.e. `[tool.pytest.ini_options] testpaths`.
# Held equal to pyproject.toml in BOTH directions by
# scripts/validation/validate_test_root_collection.py, which is why this is
# derived from COLLOCATED_TEST_ROOTS rather than re-listed here: a new collocated
# root joins the population automatically, and one that is added to only one of
# the two places fails that validator rather than silently shrinking the
# denominator below.
def _full_suite_roots() -> tuple[str, ...]:
    return (TESTS_PREFIX, *sorted(set(COLLOCATED_TEST_ROOTS.values())))


class ScriptReferenceEscalationError(RuntimeError):
    """The scripts/-reference scan cannot produce a narrowed selection.

    Carries the `EnumFullSuiteReason` the caller must escalate under, so the
    two conditions that raise it (an unusable scan, a non-collectable module in
    the tests/ root) report themselves honestly instead of sharing one reason
    whose name would misdescribe half of its occurrences.
    """

    def __init__(self, reason: EnumFullSuiteReason, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason


class ModelScriptReferenceScan(BaseModel):
    """What the reference scan found: selectable paths, and what it could not narrow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_paths: frozenset[str] = Field(default_factory=frozenset)
    unnarrowable_paths: frozenset[str] = Field(default_factory=frozenset)


def _repo_unique_basenames(repo_root: Path) -> frozenset[str]:
    """Basenames that occur exactly ONCE in the whole repository.

    A basename is only usable as a search needle when nothing else in the tree
    answers to it. `deploy-runners.sh` is unique, so a test that builds the path
    segment-wise (`REPO_ROOT / "scripts" / "deploy-runners.sh"` -- the form every
    one of the three tests #3829 broke uses, and the form a search for the joined
    repo-relative path misses entirely) is still found. `__init__.py`, `cli.py`
    and `models.py` are not unique, so they never become needles and cannot drag
    the entire tree into a selection.

    Derived from the working tree at selection time. A hand-maintained denylist
    of "generic" names is the stale-map failure this whole change exists to
    remove, one level down.
    """
    counts: Counter[str] = Counter()
    for dirpath, dirnames, filenames in repo_root.walk():
        dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP_DIR_NAMES]
        del dirpath
        counts.update(filenames)
    return frozenset(name for name, count in counts.items() if count == 1)


def _script_reference_needles(
    script_path: str, unique_basenames: frozenset[str]
) -> set[str]:
    """Literals a test would contain if it exercises `script_path`.

    Three forms, each observed in the tree:
      * the repo-relative path -- `subprocess.run(["bash", "scripts/x.sh"])`;
      * the bare basename, only when repo-unique and carrying an extension --
        `REPO_ROOT / "scripts" / "deploy-runners.sh"`, where the joined path
        never appears as a literal. An extensionless name (`scripts/onex`) is
        excluded whatever its uniqueness: it reads as an ordinary English word
        and would match on prose;
      * for a `.py` script, the dotted import form -- `from scripts.ci.foo
        import bar`, where neither of the other two appears.
    """
    needles = {script_path}
    name = script_path.rsplit("/", 1)[-1]
    if "." in name and name in unique_basenames:
        needles.add(name)
    if script_path.endswith(".py"):
        needles.add(script_path[: -len(".py")].replace("/", "."))
    return needles


def script_reference_test_paths(
    changed_files: list[str],
    repo_root: Path = REPO_ROOT,
) -> ModelScriptReferenceScan:
    """Test paths that reference a changed `scripts/` file, derived by scanning.

    Fails CLOSED. Any failure to complete the walk -- no `tests/` tree, an
    unreadable directory or module, a file that is not valid UTF-8 -- raises
    `ScriptReferenceEscalationError` rather than returning the partial result. A
    half-finished scan and a scan that found nothing are indistinguishable in
    the return value, and the difference is the whole point of the scan, so it
    is carried in the control flow instead.

    Measured cost on the 2026-09-19 tree: 2,715 modules / 36.5 MB read in
    0.13 s, plus 0.04 s for the 7,948-entry uniqueness walk.
    """
    changed_scripts = [
        path
        for path in changed_files
        if path.startswith(SCRIPTS_PREFIX) and not _is_docs_only_path(path)
    ]
    if not changed_scripts:
        return ModelScriptReferenceScan()

    tests_root = repo_root / TESTS_PREFIX
    if not tests_root.is_dir():
        raise ScriptReferenceEscalationError(
            EnumFullSuiteReason.SCRIPT_REFERENCE_SCAN_FAILED,
            f"no tests tree at {tests_root}",
        )

    selected: set[str] = set()
    unnarrowable: set[str] = set()
    try:
        unique_basenames = _repo_unique_basenames(repo_root)
        needles = set[str]()
        for script in changed_scripts:
            needles |= _script_reference_needles(script, unique_basenames)

        for dirpath, dirnames, filenames in tests_root.walk(on_error=_raise_scan_error):
            dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP_DIR_NAMES]
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                text = (dirpath / name).read_text(encoding="utf-8")
                if not any(needle in text for needle in needles):
                    continue
                rel = (dirpath / name).relative_to(repo_root).as_posix()
                if rel.startswith(UNRUNNABLE_TEST_PREFIXES):
                    # Selecting one of these can never make it run -- both
                    # pytest steps ignore or deselect them identically -- so
                    # emitting it would only risk exit 5, never add proof.
                    continue
                if is_collectable_test_file_name(name):
                    # FILE grain. A module that names a script is the module
                    # the script can break; its siblings are not, and at
                    # directory grain one reference inside `tests/ci/` costs
                    # that whole 241-module tree. Measured over the last 20
                    # merged pull requests, directory grain costs +12.3%
                    # collectable modules against this arm's +0.3%.
                    selected.add(rel)
                elif rel.rsplit("/", 1)[0] + "/" != TESTS_PREFIX:
                    # A helper, `conftest.py` or `__init__.py` that names the
                    # script. Pytest collects nothing from it, and anything in
                    # its package may import it, so the honest unit is the
                    # directory.
                    selected.add(rel.rsplit("/", 1)[0] + "/")
                else:
                    # The same shape in the `tests/` ROOT, where the directory
                    # is the whole tree. Reported, not selected -- `_resolve`
                    # escalates rather than emit `tests/` as a narrowed path.
                    unnarrowable.add(rel)
    except (OSError, UnicodeDecodeError) as exc:
        raise ScriptReferenceEscalationError(
            EnumFullSuiteReason.SCRIPT_REFERENCE_SCAN_FAILED,
            f"scan of {tests_root} failed: {exc}",
        ) from exc

    return ModelScriptReferenceScan(
        selected_paths=frozenset(selected),
        unnarrowable_paths=frozenset(unnarrowable),
    )


def _raise_scan_error(exc: OSError) -> None:
    """`Path.walk`'s default swallows an unreadable directory. This one does not."""
    raise exc


def resolve_test_paths(
    changed_files: list[str],
    adjacency_path: Path,
) -> list[str]:
    """Map changed file paths to deterministic test directories.

    Behavior:
      - Source changes under src/omnibase_infra/<module>: include
        tests/unit/<module>/.
      - Changes under scripts/: include tests/scripts/ + tests/unit/scripts/
        (scripts/ci/ additionally keeps its tests/ci/ CI-process mapping).
      - Changes under .github/workflows/ (and the other CI-process paths):
        include CI_CONTRACT_TEST_ROOT, the CI-contract class -- see the ruling
        on that constant.
      - ANY changed path under tests/ is covered by the returned selection --
        its own directory at minimum (OMN-15245), or, for a collectable module
        sitting directly in the tests/ root, the module itself (OMN-16745).
        Narrowing may add tests; it may never drop a test file the diff itself
        touched.
      - Files outside src/, scripts/ and tests/: no contribution; caller decides
        whether to escalate to full suite.

    Adjacency expansion maps each changed module to its reverse dependents,
    ensuring downstream tests run when a shared module changes.
    """
    config = load_adjacency_map(adjacency_path)
    return _resolve(changed_files, config)


def _selection_target_exists(repo_root: Path, selected_path: str) -> bool:
    """True when `selected_path` is on disk in the shape the selector emitted."""
    target = repo_root / selected_path
    return target.is_dir() if selected_path.endswith("/") else target.is_file()


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
        elif path.startswith("tests/ci/") or any(
            path == prefix.rstrip("/") or path.startswith(prefix)
            for prefix in CI_PROCESS_TEST_PATHS
        ):
            selected.add(CI_CONTRACT_TEST_ROOT)

        if path.startswith(SCRIPTS_PREFIX):
            # OMN-15245: scripts/ holds deploy-path and governance-guard code
            # whose tests live in tests/scripts/ and tests/unit/scripts/. Before
            # this mapping a scripts/ change reached neither: it produced no
            # selection at all and fell through to the blanket tests/unit/
            # fallback, which exercises none of it (recorded live on OMN-15218 /
            # omnibase_infra#2493). Note this is an `if`, not an `elif`:
            # scripts/ci/ keeps its tests/ci/ CI-process mapping AND gains these.
            selected.update(SCRIPTS_TEST_PREFIXES)

        if path.startswith(MIGRATION_TREE_PREFIX):
            # OMN-15336 item 4 repair follow-up: see MIGRATION_TREE_PREFIX's
            # own comment above. Deliberately NOT routed through
            # COLLOCATED_TEST_ROOTS -- tests/scripts/ is already collected via
            # the plain "tests" testpaths entry, so adding it as a
            # COLLOCATED_TEST_ROOTS value would trip
            # check_collocated_selector_coverage's parity assertion in
            # scripts/validation/validate_test_root_collection.py (that check
            # is scoped to roots requiring their OWN testpaths entry, which
            # tests/scripts/ does not).
            selected.add("tests/scripts/")
            # ADDITIVE fix-forward (see MIGRATION_TREE_PREFIX comment): also
            # keep the blanket tests/unit/ coverage this path class relied on
            # via the `if not selected` fallback before this mapping existed.
            # Unlike scripts/ above, tests/unit/ genuinely exercises migration
            # .sql/ledger changes (tests/unit/migrations/, tests/unit/topology/,
            # test_schema_fingerprint.py, test_db_ownership.py,
            # test_adversarial_fingerprint_drift.py), so giving this branch its
            # own non-empty selection must not silently drop that coverage by
            # suppressing the fallback.
            selected.add(TEST_UNIT_PREFIX)

        # OMN-15410: collocated roots (tests living beside their code rather
        # than under tests/). Independent of every branch above — a path can
        # legitimately map to a tests/ directory AND to its collocated root.
        for source_prefix, collocated_root in COLLOCATED_TEST_ROOTS.items():
            if path.startswith(source_prefix):
                selected.add(collocated_root)

        # OMN-18012: BOUNDARY source -> its integration proof. Also an `if`,
        # not an `elif`: a boundary file keeps every unit mapping it already
        # had and GAINS the integration suite. The unit mapping is the mocked
        # half; the point of this edge is that the mocked half is exactly what
        # was green while the boundary was broken.
        for source, targets in config.boundary_integration_tests.items():
            if path == source or (source.endswith("/") and path.startswith(source)):
                selected.update(targets)

    expanded: set[str] = set(direct_modules)
    for module in direct_modules:
        expanded.update(config.adjacency[module].reverse_deps)

    for module in expanded:
        selected.add(f"{TEST_UNIT_PREFIX}{module}/")

    # OMN-16745: a root-level test module is narrowable to ITSELF. This is the
    # file-grain half of the classification documented on
    # `_requires_unnarrowable_full_suite`; the non-collectable half never
    # reaches here, because `compute_selection` escalates on it first.
    for path in _root_level_changed_test_paths(changed_files):
        if is_collectable_test_file_name(path.rsplit("/", 1)[1]):
            selected.add(path)

    # OMN-18833: every test that REFERENCES a changed `scripts/` file, derived
    # by scanning the tree rather than read off a prefix convention the tree
    # does not obey. Additive to SCRIPTS_TEST_PREFIXES above, never a swap: a
    # script test living in one of the two prefixes without naming the script
    # keeps its selection. Raises rather than returning short when the scan
    # cannot be completed -- `compute_selection` turns that into a full suite.
    scan = script_reference_test_paths(changed_files, repo_root)
    if scan.unnarrowable_paths:
        raise ScriptReferenceEscalationError(
            EnumFullSuiteReason.SCRIPT_REFERENCE_UNNARROWABLE,
            "non-collectable tests/ root modules reference a changed script: "
            + ", ".join(sorted(scan.unnarrowable_paths)),
        )
    # Sorted so a shorter prefix is considered before anything nested under
    # it: `tests/unit/scripts/ci/` adds nothing once `tests/unit/scripts/` is
    # selected, and emitting both only makes the selection look wider than the
    # work behind it. Cost is unchanged either way -- `split_count_for_selection`
    # unions the modules -- but a selection nobody can read is how a real
    # widening goes unnoticed in review.
    for scanned in sorted(scan.selected_paths):
        if not _is_covered_by(selected, scanned):
            selected.add(scanned)

    # OMN-15245 fail-closed invariant, applied LAST so it sees everything the
    # mappings above already cover: every CHANGED path under tests/ must be
    # collected by the emitted selection.
    selected.update(_uncovered_changed_test_dirs(changed_files, selected))

    # Drop selected targets that do not exist on disk. A module in the
    # adjacency map (e.g. `dlq`) may have source under src/ but no
    # corresponding tests/unit/<module>/ directory, and a root-level test
    # module the diff DELETED no longer exists at HEAD; passing a missing path
    # to pytest aborts collection with exit code 5 ("no tests ran"). Filtering
    # to existing targets keeps the gate honest for any zone whose reverse
    # dependents include a test-less module. Directories are required to be
    # directories and file-grain entries to be files, so neither shape can
    # satisfy the check by accident.
    return sorted(p for p in selected if _selection_target_exists(repo_root, p))


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

    # 2b. Unnarrowable changed test (OMN-15245, narrowed by OMN-16745): a
    # changed module directly under tests/ that pytest would NOT collect. It
    # has no containing directory below `tests/`, pytest cannot run it, and any
    # suite may import it. A root-level module pytest DOES collect is narrowed
    # to itself at file grain in `_resolve` instead -- see the ruling above
    # CI_CONTRACT_TEST_ROOT and `_requires_unnarrowable_full_suite`.
    if _requires_unnarrowable_full_suite(changed_files):
        return _full_suite(EnumFullSuiteReason.CHANGED_TEST_UNNARROWABLE)

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

    # 5. Docs-only exemption (OMN-14753): a diff where EVERY changed file is
    # documentation cannot affect any test outcome. Select nothing rather than
    # falling through to the conservative tests/unit/ fallback below -- that
    # fallback exists for genuinely-unclassified changes (a new script
    # directory, config we have no adjacency entry for), not for a diff we can
    # positively prove is prose. A single non-doc file anywhere in the diff
    # (including one this selector doesn't otherwise recognize) disqualifies
    # the exemption and falls through to the normal smart-selection/fallback
    # path below, so ambiguous or mixed changes still escalate.
    if changed_files and all(_is_docs_only_path(p) for p in changed_files):
        return ModelTestSelection(
            selected_paths=[],
            split_count=1,
            is_full_suite=False,
            full_suite_reason=None,
            matrix=[1],
        )

    # 6. Smart selection.
    #
    # OMN-18833: the scripts/-reference scan inside `_resolve` fails CLOSED by
    # raising. A scripts/ diff whose referencing tests could not be enumerated
    # is a diff the selector cannot narrow honestly, so it escalates under the
    # reason the scan names rather than shipping a selection built from a walk
    # that did not finish.
    try:
        selected = _resolve(changed_files, config)
    except ScriptReferenceEscalationError as exc:
        return _full_suite(exc.reason)
    if not selected:
        # Conservative one-shard fallback over the full tests/unit/ tree. This
        # is NOT a no-op — it runs ~3-5 min of unit tests. It fires for changes
        # that have no test mapping at all (an unrecognised config file, a new
        # top-level directory) and are NOT provably docs-only (step 5 above
        # already exempted the pure-docs case). Per Selector Truth Boundary:
        # safer to run something than nothing. `.github/workflows/**` is NOT in
        # this population — it maps positively to CI_CONTRACT_TEST_ROOT.
        selected = ["tests/unit/"]
    split_count = split_count_for_selection(selected)

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


def _collectable_modules_under(selected_path: str, repo_root: Path) -> frozenset[Path]:
    """Every module pytest would collect under one selected path.

    Returns resolved paths so overlapping selections (`tests/unit/` alongside
    `tests/unit/docker/`) can be unioned rather than summed.
    """
    target = repo_root / selected_path
    if not selected_path.endswith("/"):
        return (
            frozenset({target})
            if target.is_file() and is_collectable_test_file_name(target.name)
            else frozenset()
        )
    if not target.is_dir():
        return frozenset()
    found: set[Path] = set()
    for dirpath, dirnames, filenames in target.walk():
        dirnames[:] = [d for d in dirnames if d not in _UNCOUNTED_DIR_NAMES]
        found.update(
            dirpath / name for name in filenames if is_collectable_test_file_name(name)
        )
    return frozenset(found)


def collectable_test_file_count(
    selected_paths: list[str], repo_root: Path = REPO_ROOT
) -> int:
    """How many modules pytest would collect for this selection, de-duplicated.

    Counted off the working tree at selection time, so it cannot go stale as the
    suite grows. A selected path that is not on disk contributes nothing: the
    selector already refuses to emit one (`_selection_target_exists`), and an
    absent path is zero work either way.
    """
    modules: set[Path] = set()
    for path in selected_paths:
        modules.update(_collectable_modules_under(path, repo_root))
    return len(modules)


def full_suite_test_file_count(repo_root: Path = REPO_ROOT) -> int:
    """The population the full-suite path hands pytest, over every testpaths root."""
    return collectable_test_file_count(list(_full_suite_roots()), repo_root)


def split_count_for_selection(
    selected_paths: list[str], repo_root: Path = REPO_ROOT
) -> int:
    """Size the shard matrix to the test population, never to the path count.

    OMN-18542. This used to be a ladder over `len(selected_paths)`:

        n <= 2 -> 1   n <= 5 -> 2   n <= 10 -> 3   n <= 16 -> 4   else 5

    Nothing in that ladder related a path to the amount of work behind it, and
    one string can be the entire unit tree. Two live populations on 2026-09-16,
    both cancelled within one second of `timeout-minutes: 15` and therefore
    neither a hang: `#3652`'s ten-path selection covered effectively the whole
    tree and was given three shards, one of which reached 59% in 15 minutes; and
    the conservative `["tests/unit/"]` fallback -- which the comment above still
    described as "~3-5 min" -- ran 28,455 cases on ONE shard at 11.9, 14.5, 14.7
    and 15.0 minutes across four runs the same day.

    The rule is PARITY, not a magic number: no narrowed selection may be denser
    per shard than the full-suite run of the whole tree, which is
    `_FULL_SUITE_SPLIT_COUNT` shards over every `testpaths` root. Both sides are
    counted from the working tree, so the ratio calibrates itself as the suite
    grows instead of ageing into the same defect.

    The result is capped at `_MAX_NARROWED_SPLIT_COUNT`, one below the
    full-suite count, so a narrowed run can never mint the full-suite shard
    denominator. See that constant for why that gap has to stay.

    Fails CLOSED. If the full-suite population reads back as zero -- a wrong
    root, a checkout that has not materialised -- the selection cannot be sized,
    and returning 1 would put an unknown amount of work on one shard. That is
    exactly the failure this function exists to remove, so it returns the
    full-suite count instead.
    """
    full_suite_files = full_suite_test_file_count(repo_root)
    if full_suite_files <= 0:
        return _MAX_NARROWED_SPLIT_COUNT
    target_per_split = math.ceil(full_suite_files / _FULL_SUITE_SPLIT_COUNT)

    selected_files = collectable_test_file_count(selected_paths, repo_root)
    if selected_files <= 0:
        return 1
    return min(_MAX_NARROWED_SPLIT_COUNT, math.ceil(selected_files / target_per_split))


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
