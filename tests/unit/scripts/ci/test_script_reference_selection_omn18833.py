# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The selector reaches tests that REFERENCE a changed script (OMN-18833).

`SCRIPTS_TEST_PREFIXES` maps every `scripts/` change to `tests/scripts/` and
`tests/unit/scripts/`. That is a convention about where script tests are
supposed to live, and the tree does not obey it. On 2026-09-19
omnibase_infra#3829 changed `scripts/deploy-runners.sh`; the three tests it
broke live in `tests/unit/observability/runner_health/`; PR CI selected the
five paths the prefixes produce, every one of them green, and dev went red --
the third dev break of that day.

The repair is a scan, not a wider map: at selection time the selector reads
`tests/` and selects the directory of every module that names the changed
script. A map would be correct on the day it was written and wrong the next
time somebody puts a script test somewhere new, which is the defect itself.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from scripts.ci.detect_test_paths import (
    _SCAN_SKIP_DIR_NAMES,
    SCRIPTS_TEST_PREFIXES,
    UNRUNNABLE_TEST_PREFIXES,
    ScriptReferenceEscalationError,
    _repo_unique_basenames,
    _resolve,
    _script_reference_needles,
    compute_selection,
    resolve_test_paths,
    script_reference_test_paths,
)
from scripts.ci.test_selection_loader import load_adjacency_map
from scripts.ci.test_selection_models import EnumFullSuiteReason

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
ADJ = REPO_ROOT / "scripts/ci/test_selection_adjacency.yaml"

# The exact changed-file list of omnibase_infra#3829, the merge that broke dev.
PR_3829_CHANGED_FILES = [
    "scripts/ci/check_runner_host_artifact_freshness.py",
    "scripts/deploy-runners.sh",
    "tests/ci/test_runner_host_hook_converge_omn18819.py",
]
# Where the three tests it broke live. Outside both SCRIPTS_TEST_PREFIXES.
BROKEN_TEST_DIR = "tests/unit/observability/runner_health/"


def _is_selected(test_module: str, selected: object) -> bool:
    """True when pytest, given `selected`, would collect `test_module`.

    The scan emits a collectable module at FILE grain and a helper's package at
    directory grain, so a caller asking "is this module covered?" has to accept
    either shape.
    """
    return any(
        test_module == path or (path.endswith("/") and test_module.startswith(path))
        for path in selected  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# (a) the replay -- RED before this change, GREEN after
# ---------------------------------------------------------------------------


def test_pr_3829_replay_selects_the_directory_that_broke_dev() -> None:
    """The regression this change exists to remove, replayed from its own diff.

    Read the pre-change bytes with `git show origin/dev:scripts/ci/
    detect_test_paths.py` to see this fail: that selector returns exactly
    ["scripts/ci/tests/", "scripts/tests/", "tests/ci/", "tests/scripts/",
    "tests/unit/scripts/"] and no runner_health path at all.
    """
    paths = resolve_test_paths(PR_3829_CHANGED_FILES, adjacency_path=ADJ)

    broke_dev = sorted(
        (REPO_ROOT / BROKEN_TEST_DIR).glob("test_*.py"),
    )
    named = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in broke_dev
        if "deploy-runners.sh" in p.read_text(encoding="utf-8")
    ]
    assert named, "no module in runner_health names the script any more"

    missing = [m for m in named if not _is_selected(m, paths)]
    assert missing == [], (
        f"these modules read scripts/deploy-runners.sh off disk, so a diff "
        f"changing that script must select them: {missing}. Got: {paths}"
    )


def test_pr_3829_replay_keeps_every_path_the_prefixes_already_gave() -> None:
    """Additive, never a swap: the two prefixes are a floor the scan builds on."""
    paths = set(resolve_test_paths(PR_3829_CHANGED_FILES, adjacency_path=ADJ))

    assert set(SCRIPTS_TEST_PREFIXES) <= paths
    assert {"scripts/ci/tests/", "scripts/tests/", "tests/ci/"} <= paths


def test_pr_3829_replay_is_still_a_narrowed_selection() -> None:
    """Closing the blind spot must not quietly become a full-suite escalation."""
    selection = compute_selection(
        changed_files=PR_3829_CHANGED_FILES,
        adjacency_path=ADJ,
        ref_name="dev",
    )

    assert selection.is_full_suite is False
    assert selection.full_suite_reason is None
    assert any(p.startswith(BROKEN_TEST_DIR) for p in selection.selected_paths)


# ---------------------------------------------------------------------------
# (b) a script no test names changes nothing
# ---------------------------------------------------------------------------


def test_script_referenced_by_no_test_selects_exactly_what_it_did_before() -> None:
    """No reference, no widening. The scan adds tests; it never adds noise.

    `scripts/` holds 594 files and only 139 of them are named by a test outside
    the two prefixes. The other 455 must select precisely the prefixes, or this
    change trades one defect for a fleet-wide cost increase.
    """
    unreferenced = _a_script_no_test_outside_the_prefixes_names()
    paths = resolve_test_paths([unreferenced], adjacency_path=ADJ)

    outside = [
        p
        for p in paths
        if not p.startswith(SCRIPTS_TEST_PREFIXES) and not p.startswith("scripts/")
    ]
    assert outside == [], (
        f"{unreferenced} is named by no test outside {SCRIPTS_TEST_PREFIXES}, "
        f"so it must not pull in {outside}"
    )


def test_a_scan_that_finds_nothing_returns_an_empty_result_not_an_escalation() -> None:
    scan = script_reference_test_paths([_a_script_no_test_outside_the_prefixes_names()])

    assert scan.unnarrowable_paths == frozenset()


def test_non_script_diff_never_scans() -> None:
    """A diff with no `scripts/` path pays nothing: the scan short-circuits."""
    scan = script_reference_test_paths(
        ["src/omnibase_infra/cli/foo.py", "docs/a.md"],
        repo_root=Path("/nonexistent-omn18833"),
    )

    assert scan.selected_paths == frozenset()
    assert scan.unnarrowable_paths == frozenset()


# ---------------------------------------------------------------------------
# (c) the scan fails CLOSED
# ---------------------------------------------------------------------------


def test_missing_tests_tree_escalates_rather_than_selecting_nothing(
    tmp_path: Path,
) -> None:
    """A scan that could not run and a scan that found nothing are different.

    Returning the empty result on failure would make them identical in the
    return value, which is the whole thing the scan exists to distinguish.
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.sh").write_text("#!/bin/bash\n", encoding="utf-8")

    with pytest.raises(ScriptReferenceEscalationError) as excinfo:
        script_reference_test_paths(["scripts/x.sh"], repo_root=tmp_path)

    assert excinfo.value.reason is EnumFullSuiteReason.SCRIPT_REFERENCE_SCAN_FAILED


def test_undecodable_test_module_escalates(tmp_path: Path) -> None:
    """An unreadable module means an incomplete answer, not a negative one."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_binary.py").write_bytes(b"\xff\xfe\x00not utf-8")

    with pytest.raises(ScriptReferenceEscalationError) as excinfo:
        script_reference_test_paths(["scripts/x.sh"], repo_root=tmp_path)

    assert excinfo.value.reason is EnumFullSuiteReason.SCRIPT_REFERENCE_SCAN_FAILED


def test_unreadable_test_directory_escalates(tmp_path: Path) -> None:
    """`Path.walk` swallows an unreadable directory by default. This one must not."""
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions, so this cannot be proven")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    locked = tmp_path / "tests" / "locked"
    locked.mkdir(parents=True)
    (locked / "test_a.py").write_text("scripts/x.sh\n", encoding="utf-8")
    locked.chmod(0o000)
    try:
        with pytest.raises(ScriptReferenceEscalationError) as excinfo:
            script_reference_test_paths(["scripts/x.sh"], repo_root=tmp_path)
    finally:
        locked.chmod(0o755)

    assert excinfo.value.reason is EnumFullSuiteReason.SCRIPT_REFERENCE_SCAN_FAILED


def test_compute_selection_turns_a_scan_escalation_into_a_full_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The raise reaches the gate as a real full suite, under the scan's own reason."""
    import scripts.ci.detect_test_paths as module

    def _boom(changed_files: list[str], repo_root: Path = REPO_ROOT) -> None:
        raise ScriptReferenceEscalationError(
            EnumFullSuiteReason.SCRIPT_REFERENCE_SCAN_FAILED, "injected"
        )

    monkeypatch.setattr(module, "script_reference_test_paths", _boom)

    selection = compute_selection(
        changed_files=["scripts/deploy-runners.sh"],
        adjacency_path=ADJ,
        ref_name="dev",
    )

    assert selection.is_full_suite is True
    assert (
        selection.full_suite_reason is EnumFullSuiteReason.SCRIPT_REFERENCE_SCAN_FAILED
    )
    assert selection.selected_paths == ["tests/"]
    assert selection.split_count == 15


def test_non_collectable_tests_root_module_escalates(tmp_path: Path) -> None:
    """A shared helper in the tests/ root has the whole tree as its blast radius.

    Same shape as `changed_test_unnarrowable`: no containing directory below
    `tests/`, pytest collects nothing from it, and any suite may import it. The
    honest answer is the real full suite, under its own reason so the escalation
    does not misreport why it happened.
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "infrastructure_config.py").write_text(
        'SCRIPT = "scripts/x.sh"\n', encoding="utf-8"
    )

    scan = script_reference_test_paths(["scripts/x.sh"], repo_root=tmp_path)
    assert scan.unnarrowable_paths == frozenset({"tests/infrastructure_config.py"})
    assert scan.selected_paths == frozenset()

    # The scan reports; `_resolve` decides. A scan that cannot COMPLETE raises,
    # because there is no honest value to return; a scan that completed and
    # found something unnarrowable returns the finding, and the escalation is
    # the caller's, one layer up where every other escalation is made.
    with pytest.raises(ScriptReferenceEscalationError) as excinfo:
        _resolve(["scripts/x.sh"], load_adjacency_map(ADJ), repo_root=tmp_path)

    assert excinfo.value.reason is EnumFullSuiteReason.SCRIPT_REFERENCE_UNNARROWABLE


def test_collectable_tests_root_module_is_narrowed_to_itself(tmp_path: Path) -> None:
    """The collectable half of the same population narrows to file grain."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_root.py").write_text(
        'SCRIPT = "scripts/x.sh"\n', encoding="utf-8"
    )

    scan = script_reference_test_paths(["scripts/x.sh"], repo_root=tmp_path)

    assert scan.selected_paths == frozenset({"tests/test_root.py"})


# ---------------------------------------------------------------------------
# needle derivation
# ---------------------------------------------------------------------------


def test_segment_wise_path_construction_is_found_by_basename(tmp_path: Path) -> None:
    """`REPO_ROOT / "scripts" / "deploy-runners.sh"` never spells the joined path.

    Every one of the three tests #3829 broke builds the path this way, so a scan
    for the repo-relative literal alone reproduces the blind spot it is fixing.
    """
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "deploy-runners.sh").write_text("x\n", encoding="utf-8")
    d = tmp_path / "tests" / "unit" / "obs"
    d.mkdir(parents=True)
    (d / "test_a.py").write_text(
        'DEPLOY = REPO_ROOT / "scripts" / "deploy-runners.sh"\n', encoding="utf-8"
    )

    scan = script_reference_test_paths(
        ["scripts/deploy-runners.sh"], repo_root=tmp_path
    )

    assert scan.selected_paths == frozenset({"tests/unit/obs/test_a.py"})


def test_dotted_import_form_is_found_for_a_python_script(tmp_path: Path) -> None:
    (tmp_path / "scripts" / "ci").mkdir(parents=True)
    (tmp_path / "scripts" / "ci" / "audit_thing.py").write_text("x\n", encoding="utf-8")
    d = tmp_path / "tests" / "unit" / "obs"
    d.mkdir(parents=True)
    (d / "test_a.py").write_text(
        "from scripts.ci.audit_thing import run\n", encoding="utf-8"
    )

    scan = script_reference_test_paths(
        ["scripts/ci/audit_thing.py"], repo_root=tmp_path
    )

    assert scan.selected_paths == frozenset({"tests/unit/obs/test_a.py"})


def test_a_basename_shared_with_another_file_is_never_a_needle() -> None:
    """A non-unique basename would drag the tree in. Uniqueness is measured, not listed.

    `__init__.py` is the obvious case and a hand-written denylist would catch it.
    The point of deriving this is the non-obvious ones -- `cli.py`, `models.py`,
    `executor.py`, `topics.py` all exist both under `scripts/` and elsewhere in
    this repo today, and the next one to appear is not knowable in advance.
    """
    unique = _repo_unique_basenames(REPO_ROOT)

    assert "__init__.py" not in unique
    needles = _script_reference_needles("scripts/deploy-agent/__init__.py", unique)
    assert "__init__.py" not in needles
    assert needles == {
        "scripts/deploy-agent/__init__.py",
        "scripts.deploy-agent.__init__",
    }


def test_an_extensionless_script_never_contributes_its_bare_name() -> None:
    """`scripts/onex` is repo-unique and still unusable: it reads as a word."""
    unique = _repo_unique_basenames(REPO_ROOT)

    assert _script_reference_needles("scripts/onex", unique) == {"scripts/onex"}


def test_a_unique_extensioned_basename_is_a_needle() -> None:
    unique = _repo_unique_basenames(REPO_ROOT)

    assert "deploy-runners.sh" in _script_reference_needles(
        "scripts/deploy-runners.sh", unique
    )


# ---------------------------------------------------------------------------
# (d) the ratchet -- a NEW blind spot is a red test
# ---------------------------------------------------------------------------

# Deliberately NOT the selector's own needle logic: a ratchet that detects
# references the same way the selector does can only ever agree with it. This
# reads the two literal forms a test can use to name a script -- a joined
# repo-relative path, and a segment-wise `"scripts" / "..." / "name.ext"`
# construction -- and keeps a hit only when it resolves to a file that is
# really on disk, so a string that merely looks like a path is not a finding.
_REFERENCE_LITERAL = re.compile(
    r"scripts/[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:sh|py)"
    r'|"scripts"\s*[,/]\s*(?:"[A-Za-z0-9_][A-Za-z0-9_./-]*"\s*[,/]\s*)*'
    r'"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:sh|py)"'
)
_QUOTED_SEGMENT = re.compile(r'"([A-Za-z0-9_][A-Za-z0-9_./-]*)"')


def _independently_detected_reference_pairs() -> set[tuple[str, str]]:
    """(script, test module) pairs found WITHOUT using the selector's needles."""
    pairs: set[tuple[str, str]] = set()
    for dirpath, dirnames, filenames in (REPO_ROOT / "tests").walk():
        dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP_DIR_NAMES]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            rel = (dirpath / name).relative_to(REPO_ROOT).as_posix()
            if rel.startswith(SCRIPTS_TEST_PREFIXES):
                continue  # already selected wholesale by the prefixes
            if rel.startswith(UNRUNNABLE_TEST_PREFIXES):
                continue  # neither pytest step can run these
            text = (dirpath / name).read_text(encoding="utf-8")
            for match in _REFERENCE_LITERAL.finditer(text):
                literal = match.group(0)
                if literal.startswith('"'):
                    segments = _QUOTED_SEGMENT.findall(literal)
                    script = "scripts/" + "/".join(segments[1:])
                else:
                    script = literal
                if (REPO_ROOT / script).is_file():
                    pairs.add((script, rel))
    return pairs


def test_the_ratchet_detector_still_finds_the_population_it_is_ratcheting() -> None:
    """A zero-row ratchet passes vacuously. Prove the detector is not returning one.

    It also has to find the pair this whole change is about, which is the
    positive control: a detector that missed `deploy-runners.sh` ->
    `runner_health` would report the fleet clean on the exact defect of record.
    """
    pairs = _independently_detected_reference_pairs()

    assert len(pairs) >= 100, f"detector found only {len(pairs)} pairs; it is broken"
    assert any(
        script == "scripts/deploy-runners.sh" and test.startswith(BROKEN_TEST_DIR)
        for script, test in pairs
    ), "the detector no longer finds the #3829 pair it was built from"


def test_every_referenced_script_reaches_its_test_through_the_selector() -> None:
    """Each independently-found pair must be reachable through the real selector.

    Asserted as a conjunction of the two halves that together ARE per-pair
    reachability, so one whole-tree scan answers for all 285 pairs instead of
    one scan per script:

      * the script's needles occur in that module's text -- so a scan given
        that script alone matches that module; and
      * the module's directory is in the scan's output -- so matching it emits
        a path pytest will collect it from.

    A test written somewhere new that names a script in a form the needles do
    not carry fails the first half. A module the scan matches but cannot emit a
    path for fails the second.
    """
    pairs = _independently_detected_reference_pairs()
    unique = _repo_unique_basenames(REPO_ROOT)
    scan = script_reference_test_paths(sorted({script for script, _ in pairs}))
    text_cache: dict[str, str] = {}

    unreachable: list[str] = []
    for script, test in sorted(pairs):
        if test not in text_cache:
            text_cache[test] = (REPO_ROOT / test).read_text(encoding="utf-8")
        needles = _script_reference_needles(script, unique)
        if not any(needle in text_cache[test] for needle in needles):
            unreachable.append(f"{test} names {script} in no form the scan searches")
            continue
        if not _is_selected(test, scan.selected_paths):
            unreachable.append(f"{test} matched but no path covering it was emitted")

    assert unreachable == [], (
        "new selector blind spot -- these tests exercise a script the selector "
        "would not select them for:\n  " + "\n  ".join(unreachable)
    )


def _tests_tree_corpus() -> list[str]:
    """Every module under `tests/`, read once."""
    corpus: list[str] = []
    for dirpath, dirnames, filenames in (REPO_ROOT / "tests").walk():
        dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP_DIR_NAMES]
        corpus.extend(
            (dirpath / name).read_text(encoding="utf-8")
            for name in filenames
            if name.endswith(".py")
        )
    return corpus


def _a_script_no_test_outside_the_prefixes_names() -> str:
    """A real script under `scripts/` that no module under `tests/` names.

    Chosen from the tree rather than hardcoded: a pinned filename would be a
    second stale map, and the first person to write a test for that script
    would get a red test with nothing wrong with their change.
    """
    corpus = _tests_tree_corpus()
    unique = _repo_unique_basenames(REPO_ROOT)
    for dirpath, dirnames, filenames in (REPO_ROOT / "scripts").walk():
        dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP_DIR_NAMES]
        for name in sorted(filenames):
            if not name.endswith((".sh", ".py")):
                continue
            rel = (dirpath / name).relative_to(REPO_ROOT).as_posix()
            if "/tests/" in rel:
                continue
            needles = _script_reference_needles(rel, unique)
            if not any(n in text for text in corpus for n in needles):
                return rel
    pytest.fail("every script under scripts/ is named by a test; cannot prove (b)")
