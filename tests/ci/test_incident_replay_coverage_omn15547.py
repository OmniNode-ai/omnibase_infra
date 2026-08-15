# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The incident-replay coverage lint must actually say no (OMN-15547).

WHAT IS UNDER TEST
    ``scripts/ci/check_incident_replay_coverage.py`` -- the module CI and
    pre-commit invoke, imported by path, never a re-implementation of its rules.

WHY EVERY RED HERE IS EXECUTED, NOT ASSERTED
    This lint exists because three guards shipped green while enforcing nothing.
    A test suite for it that only checked "the happy path passes" would be the
    same defect one level up: a lint that returns 0 unconditionally satisfies
    every green assertion and nothing else. So the load-bearing tests here all
    take the REAL repository tree, break exactly one thing about it, and require
    a non-zero exit naming the rule -- including the default-deny path, which is
    the property that makes the convention self-sustaining.

THE FIXTURE IS THE REPO
    Each RED case is built by copying the live tree's registry (and, for R1, the
    real captured artifact) into a tmp root and mutating that copy. Nothing here
    hand-writes a registry from scratch: a synthetic registry would only prove
    the lint can parse a shape this file invented, which is the exact class of
    vacuous proof the lint was written to detect.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT_PATH = REPO_ROOT / "scripts" / "ci" / "check_incident_replay_coverage.py"
REGISTRY_REL = "tests/incident_replays/registry.yaml"


def _load_lint() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_incident_replay_coverage_omn15547", LINT_PATH
    )
    assert spec and spec.loader, LINT_PATH
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LINT = _load_lint()


def _clone_tree(tmp_path: Path) -> Path:
    """Copy the parts of the real tree the lint reads, so a mutation is isolated.

    The lint reads: the registry, `.pre-commit-config.yaml`, `.github/workflows`,
    and -- to resolve R1/R4/R5 -- the guard, fixture and test files the registry
    names. Copying the whole repo would be slow; copying only what it reads keeps
    the case honest because anything the lint touches is present verbatim.
    """
    root = tmp_path / "repo"
    root.mkdir()

    shutil.copy2(
        REPO_ROOT / ".pre-commit-config.yaml", root / ".pre-commit-config.yaml"
    )
    shutil.copytree(REPO_ROOT / ".github" / "workflows", root / ".github" / "workflows")
    (root / "tests" / "incident_replays").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / REGISTRY_REL, root / REGISTRY_REL)

    registry = yaml.safe_load((REPO_ROOT / REGISTRY_REL).read_text(encoding="utf-8"))
    referenced: set[str] = set()
    for case in registry.get("cases") or []:
        referenced.add(case["guard"])
        referenced.add(case["artifact"]["fixture"])
        referenced.add(case["test"].split("::", 1)[0])
        if case.get("discriminator"):
            referenced.add(case["discriminator"].split("::", 1)[0])
    for entry in (registry.get("scope") or {}).get("debt_baseline") or []:
        referenced.add(entry)
    for entry in (registry.get("scope") or {}).get("required_guards") or []:
        referenced.add(entry)

    for rel in sorted(referenced):
        src = REPO_ROOT / rel
        if not src.is_file():
            continue  # PENDING entries legitimately do not exist yet
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return root


def _registry(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / REGISTRY_REL).read_text(encoding="utf-8"))


def _write_registry(root: Path, data: dict[str, Any]) -> None:
    (root / REGISTRY_REL).write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )


def _rules(findings: list[Any]) -> set[str]:
    return {f.rule for f in findings}


# --------------------------------------------------------------------------
# GREEN control -- on the unmutated tree the lint must pass.
# --------------------------------------------------------------------------
def test_the_live_tree_passes() -> None:
    """Without this, every RED below could be produced by a lint that always fails."""
    result = LINT.evaluate(REPO_ROOT)
    assert not result.findings, "\n".join(f.render() for f in result.findings)
    assert result.covered, "the lint reports zero covered guards -- vacuous green"


# --------------------------------------------------------------------------
# RED 1 -- a required guard loses its case (COVERAGE).
# --------------------------------------------------------------------------
def test_red_required_guard_with_no_case(tmp_path: Path) -> None:
    root = _clone_tree(tmp_path)
    data = _registry(root)
    required = data["scope"]["required_guards"]
    dropped = [
        c
        for c in data["cases"]
        if c["guard"] in required and (root / c["guard"]).exists()
    ]
    assert dropped, "no required guard is covered in the live registry -- fix the setup"
    data["cases"] = [c for c in data["cases"] if c not in dropped]
    _write_registry(root, data)

    result = LINT.evaluate(root)
    assert "COVERAGE" in _rules(result.findings), [f.render() for f in result.findings]
    assert any(dropped[0]["guard"] in f.subject for f in result.findings)


# --------------------------------------------------------------------------
# RED 2 -- R1: the captured artifact is edited after capture.
# --------------------------------------------------------------------------
def test_red_fixture_bytes_mutated_by_one_byte(tmp_path: Path) -> None:
    """A one-byte edit must break the build.

    This is the rule that separates "these are the bytes that failed" from
    "these are bytes somebody kept adjusting until the test passed".
    """
    root = _clone_tree(tmp_path)
    case = _registry(root)["cases"][0]
    fixture = root / case["artifact"]["fixture"]
    raw = fixture.read_bytes()
    fixture.write_bytes(raw + b" ")

    result = LINT.evaluate(root)
    assert "R1" in _rules(result.findings), [f.render() for f in result.findings]


# --------------------------------------------------------------------------
# RED 3 -- R2: provenance degraded to free text.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prose",
    [
        "captured from the live .201 health endpoint",
        "same shape as the production body",
        "copied off the host",
    ],
)
def test_red_free_text_provenance_is_not_a_locator(tmp_path: Path, prose: str) -> None:
    """Exactly the sentences a hand-typed fixture gets defended with.

    Each of these reads like provenance and none of them can be re-fetched. If
    the lint accepted them it would accept a reconstruction, which is the defect.
    """
    root = _clone_tree(tmp_path)
    data = _registry(root)
    data["cases"][0]["capture"]["source"] = prose
    _write_registry(root, data)

    result = LINT.evaluate(root)
    assert "R2" in _rules(result.findings), [f.render() for f in result.findings]


# --------------------------------------------------------------------------
# RED 4 -- DEFAULT-DENY: a newly wired guard with no case and no baseline row.
# --------------------------------------------------------------------------
def test_red_newly_wired_guard_defaults_to_denied(tmp_path: Path) -> None:
    """The property that makes the convention self-sustaining.

    Without it the registry is a snapshot: today's guards are covered, and every
    guard added tomorrow is exempt by default -- which is how the detection shelf
    grew faster than the proof behind it in the first place.
    """
    root = _clone_tree(tmp_path)
    new_guard = root / "scripts" / "ci" / "check_brand_new_thing_omn15547.py"
    new_guard.parent.mkdir(parents=True, exist_ok=True)
    new_guard.write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n")
    config = root / ".pre-commit-config.yaml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + "\n# OMN-15547 default-deny probe\n"
        + "#   entry: python scripts/ci/check_brand_new_thing_omn15547.py\n",
        encoding="utf-8",
    )

    result = LINT.evaluate(root)
    assert "DEFAULT-DENY" in _rules(result.findings), [
        f.render() for f in result.findings
    ]
    assert any(
        "check_brand_new_thing_omn15547.py" in f.subject for f in result.findings
    )


# --------------------------------------------------------------------------
# RED 5 -- R4: the registry claims a replay the test does not perform.
# --------------------------------------------------------------------------
def test_red_test_does_not_reference_the_fixture(tmp_path: Path) -> None:
    root = _clone_tree(tmp_path)
    data = _registry(root)
    decoy = root / "tests" / "unit" / "scripts" / "test_decoy_omn15547.py"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text("def test_nothing() -> None:\n    assert True\n")
    data["cases"][0]["test"] = "tests/unit/scripts/test_decoy_omn15547.py::test_nothing"
    _write_registry(root, data)

    result = LINT.evaluate(root)
    assert "R4" in _rules(result.findings), [f.render() for f in result.findings]


# --------------------------------------------------------------------------
# RED 6 -- R5: a false_red case with no discriminator.
# --------------------------------------------------------------------------
def test_red_false_red_case_without_a_discriminator(tmp_path: Path) -> None:
    """An accept-only proof cannot tell a working guard from a stuck-open one."""
    root = _clone_tree(tmp_path)
    data = _registry(root)
    target = next(
        (c for c in data["cases"] if c.get("regression_class") == "false_red"), None
    )
    if target is None:
        pytest.skip("no false_red case in the registry to degrade")
    target.pop("discriminator", None)
    _write_registry(root, data)

    result = LINT.evaluate(root)
    assert "R5" in _rules(result.findings), [f.render() for f in result.findings]


# --------------------------------------------------------------------------
# RED 7 -- RATCHET: the debt baseline must stay truthful.
# --------------------------------------------------------------------------
def test_red_covered_guard_left_in_the_debt_baseline(tmp_path: Path) -> None:
    root = _clone_tree(tmp_path)
    data = _registry(root)
    covered = data["cases"][0]["guard"]
    data["scope"]["debt_baseline"] = sorted({*data["scope"]["debt_baseline"], covered})
    _write_registry(root, data)

    result = LINT.evaluate(root)
    assert "RATCHET" in _rules(result.findings), [f.render() for f in result.findings]


def test_red_stale_entry_in_the_debt_baseline(tmp_path: Path) -> None:
    root = _clone_tree(tmp_path)
    data = _registry(root)
    data["scope"]["debt_baseline"] = sorted(
        {*data["scope"]["debt_baseline"], "scripts/ci/deleted_long_ago.py"}
    )
    _write_registry(root, data)

    result = LINT.evaluate(root)
    assert "RATCHET" in _rules(result.findings), [f.render() for f in result.findings]


# --------------------------------------------------------------------------
# The pre-registered requirement must ARM, not silently stay pending forever.
# --------------------------------------------------------------------------
def test_pending_requirement_arms_when_the_guard_lands(tmp_path: Path) -> None:
    """OMN-15538's guard is required before it exists; it must bind on arrival.

    A pre-registered requirement is only useful if it BINDS on arrival.
    Otherwise listing a guard before it exists would be a way to look covered
    forever: permanently PENDING, permanently green.

    This was observed live rather than imagined. ``check_pin_reachability.py``
    was pre-registered while omnibase_infra#2583 was open; when #2583 merged as
    ``1da8d3c5`` the very next run of this lint went from
    ``0 findings, 1 pending`` to ``COVERAGE: scripts/ci/check_pin_reachability
    .py`` with no edit to the registry. The test reconstructs both states from
    the real tree rather than depending on one of them still being reachable --
    a version of this test that skipped once the guard landed would prove
    nothing exactly when it started to matter.
    """
    root = _clone_tree(tmp_path)
    required = _registry(root)["scope"]["required_guards"]
    target = next((g for g in required if (root / g).exists()), None)
    assert target, "expected at least one required guard present in the clone"

    # State 1 -- the guard has not landed yet: PENDING, and NOT a failure.
    landed = root / target
    body = landed.read_bytes()
    landed.unlink()
    data = _registry(root)
    data["cases"] = [c for c in data["cases"] if c["guard"] != target]
    _write_registry(root, data)

    before = LINT.evaluate(root)
    assert target in before.pending, before.pending
    assert not before.findings, [f.render() for f in before.findings]

    # State 2 -- the guard lands with no case: the requirement must bind NOW.
    landed.parent.mkdir(parents=True, exist_ok=True)
    landed.write_bytes(body)

    after = LINT.evaluate(root)
    assert target not in after.pending
    assert "COVERAGE" in _rules(after.findings), [f.render() for f in after.findings]
    assert any(target in f.subject for f in after.findings)


# --------------------------------------------------------------------------
# The lint's own incident replay case (registry: omn15547-handtyped-fixture-
# passed-as-proof). Not a hypothetical -- these are the bytes that shipped.
# --------------------------------------------------------------------------
HANDTYPED_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn15547"
    / "test_omninode_system_slack_report.handtyped-fixture.py.captured"
)
HANDTYPED_FIXTURE_SHA256 = (
    "10cfdc48ef1a80dbc10f2c9cf1a84cfc20a68877111e15b5cbe1e56c428c5a0c"
)


def test_the_handtyped_fixture_capture_is_unmodified() -> None:
    """Provenance guard: this is the dev blob at 0f050394, verbatim."""
    digest = hashlib.sha256(HANDTYPED_FIXTURE.read_bytes()).hexdigest()
    assert digest == HANDTYPED_FIXTURE_SHA256, (
        "the captured OMN-15525 repair blob no longer matches "
        "omnibase_infra@0f05039434996594000db85cd8d3947523bfebcf; re-fetch it "
        "with `git show <sha>:<path>` rather than editing it"
    )


def test_the_shipped_handtyped_fixture_fails_r1_and_r2() -> None:
    """Replay: the artifact that WAS accepted as proof must now be refused.

    ``tests/fixtures/omn15547/test_omninode_system_slack_report.handtyped-
    fixture.py.captured`` is the verbatim blob that landed on dev as the repair
    for OMN-15525. It fixed the byte-count symptom and kept the disease: a
    literal somebody typed, defended by prose. Both halves must fail.
    """
    source = HANDTYPED_FIXTURE.read_text(encoding="utf-8")

    # R1: the health body is a literal built inside the test, not a committed
    # capture the lint can hash. That is why nothing could detect it drifting.
    assert "HEALTHY_BODY = json.dumps(" in source, (
        "the captured blob is supposed to be the LITERAL-fixture version; if "
        "this fails the capture is of the wrong revision"
    )
    assert "HEALTHY_BODY_SHA256" not in source, (
        "the captured blob must predate the sha256-pinned capture, or it is not "
        "the artifact that shipped without provenance"
    )

    # R2: harvest the provenance the file actually offers -- the comment block
    # above the literal -- and require that NONE of it resolves as a locator.
    provenance_lines = [
        line.strip().lstrip("#").strip()
        for line in source.splitlines()
        if line.strip().startswith("#")
        and any(
            token in line.lower()
            for token in ("live body", "mirrors", "realistic", ".201", "real ")
        )
    ]
    assert provenance_lines, "expected the blob's hand-written provenance comments"
    assert any("mirrors the live body" in line.lower() for line in provenance_lines), (
        "the specific sentence this case exists to refuse is missing from the capture"
    )
    for line in provenance_lines:
        assert not any(
            pattern.match(line) for pattern in LINT.LOCATOR_GRAMMARS.values()
        ), (
            f"prose accepted as a locator: {line!r} -- R2 would admit a "
            "hand-typed fixture, which is the whole defect"
        )


# --------------------------------------------------------------------------
# Locator grammar is the R2 bar -- assert both polarities on real strings.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "locator",
    [
        "gh-api:repos/OmniNode-ai/omnibase_infra/actions/runs/30574058377/jobs",
        "gh-api:repos/OmniNode-ai/omnimarket/compare/dev...879d6fc6825f876458c6d45ed670c8715de8ac95",
        "git-object:OmniNode-ai/omnimarket@879d6fc6ed6c4f6c86c2d3f0f4c1a0f8b7c6d5e4:.github/workflows/merge-hold-gate-reusable.yml",
        "host-file:omni-201-ts:/data/maintenance/bin/omninode-system-slack-report.sh",
        "live-http:omni-201-ts:8085/health",
        "ci-artifact:30574058377/coverage-shard-3",
    ],
)
def test_locator_grammars_accept_real_locators(locator: str) -> None:
    assert any(p.match(locator) for p in LINT.LOCATOR_GRAMMARS.values()), locator


@pytest.mark.parametrize(
    "locator",
    [
        "the live 201 host",
        "gh-api:repos/OmniNode-ai/omnibase_infra/actions/runs",  # no numeric id
        "git-object:OmniNode-ai/omnimarket@879d6fc6:.github/workflows/x.yml",  # short sha
        "host-file:omni-201-ts:data/maintenance/bin/x.sh",  # not absolute
        "https://example.invalid/whatever",
        "",
    ],
)
def test_locator_grammars_reject_unresolvable_provenance(locator: str) -> None:
    assert not any(p.match(locator) for p in LINT.LOCATOR_GRAMMARS.values()), locator


# --------------------------------------------------------------------------
# OMN-15536: the outage pin, replayed against the real predicate that now
# exists (registry: omn15536-unreachable-pin-outage).
# --------------------------------------------------------------------------
COMPARE_DEAD = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn15547"
    / "omnimarket-compare-dev-879d6fc6.gh-api.json.captured"
)
COMPARE_LIVE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn15547"
    / "omnimarket-compare-dev-454c429f.gh-api.json.captured"
)
COMPARE_DEAD_SHA256 = "4e957324574cf01581701cc662adba35cb092670ee67db48f31118f17239f3b5"
COMPARE_LIVE_SHA256 = "e669f0c1389440d41ba4e077fe94eea1596438d9cf866291ba298ceb2c30d5c7"


def _pin_reachability() -> Any:
    """Import the real OMN-15538 guard, by path, exactly as CI invokes it."""
    spec = importlib.util.spec_from_file_location(
        "check_pin_reachability_omn15547",
        REPO_ROOT / "scripts" / "ci" / "check_pin_reachability.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_compare_captures_are_unmodified() -> None:
    """Both captures are GitHub's own answers, byte for byte."""
    for path, expected in (
        (COMPARE_DEAD, COMPARE_DEAD_SHA256),
        (COMPARE_LIVE, COMPARE_LIVE_SHA256),
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, (
            f"{path.name} no longer matches the captured compare response; "
            "re-fetch it with gh api rather than editing it"
        )


def test_the_omn15536_outage_pin_is_unreachable_by_the_real_predicate() -> None:
    """Replay the outage against the predicate, not against a hardcoded string.

    The shape-only validators that shipped before OMN-15538 passed ``879d6fc6``
    because it is 40 hex characters. What actually mattered was GitHub's answer
    to a question they never asked, and this is that answer: ``status:
    diverged`` -- the ref is reachable from no protected branch.

    Driving ``status_is_reachable`` over the captured payload keeps the test
    hermetic while still exercising the predicate CI runs, and the ``behind``
    control below is what stops a blanket-reject implementation passing.
    """
    module = _pin_reachability()
    dead = json.loads(COMPARE_DEAD.read_text(encoding="utf-8"))

    assert dead["status"] == "diverged", (
        "the captured payload is supposed to be the UNREACHABLE case; if GitHub "
        "now answers differently the capture is stale, not the guard wrong"
    )
    assert module.status_is_reachable(dead["status"]) is False, (
        "the pin-reachability predicate accepts the exact ref that wedged every "
        "open omnibase_infra PR for ~2.5h on 2026-07-30 (OMN-15536)"
    )


def test_the_squash_that_actually_landed_is_reachable() -> None:
    """Discriminating control: reject-everything must not look correct.

    ``454c429f`` is the squash commit omnimarket#1976 really produced -- the ref
    #2577 repointed to. A guard that refused it too would break every legitimate
    pin, so the outage case above only means something alongside this one.
    """
    module = _pin_reachability()
    live = json.loads(COMPARE_LIVE.read_text(encoding="utf-8"))

    assert live["status"] == "behind"
    assert module.status_is_reachable(live["status"]) is True, (
        "the predicate rejects a ref that IS an ancestor of dev -- a "
        "blanket-reject guard is as broken as a blanket-accept one"
    )
