#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every enforcement guard must replay a REAL incident (OMN-15547).

WHY THIS EXISTS
---------------
A guard that has never been run against the real thing it exists to catch is
decorative. On 2026-07-30 that failed three times in one day, with an identical
shape every time -- *the guard was tested against a synthetic input that cannot
exhibit the failure*:

1. The workflow-pin validators are shape-only. Re-running
   ``tests/ci/test_merge_hold_gate_omn15484.py`` against ``879d6fc6`` -- the
   exact pin that wedged every ``omnibase_infra`` PR for ~2.5h (OMN-15536) --
   returns ``2 passed``. Any 40-hex string satisfies them.
2. ``tests/unit/scripts/test_omninode_system_slack_report.py`` drove the health
   reporter with a **63-byte** ``HEALTHY_BODY`` while every real ``/health`` body
   on ``.201`` is 2079-2644 bytes. The reporter truncated to 180 bytes before
   ``jq``; under the short fixture the truncation never bit, so a 654-line suite
   stayed green while the deployed artifact reported CRITICAL for all three
   lanes against a fully healthy fleet (OMN-15525).
3. A merge-hold falsifier was validated under ``bash -c``/``sh -c`` rather than
   the runner that executes it. The DoD runner's admissibility predicate reads
   *command position*, so the probe -- wrapped in ``python3 -c "..."`` -- was
   classified ``NOT_EXECUTED``: the check never ran at all (OMN-15484).

In all three the guard was green, the enforcement was zero, and nothing in CI
could tell the difference. This lint makes that difference machine-visible.

WHAT IT ENFORCES
----------------
``tests/incident_replays/registry.yaml`` declares **incident replay cases**. A
case is REAL only if all five rules hold -- these are the honest bar, written so
a machine can apply them:

  R1  COMMITTED BYTES, UNMODIFIED. ``artifact.fixture`` names a file in the
      repo (never a literal inside a test) and ``sha256(bytes)`` equals
      ``artifact.sha256``. Editing a captured artifact after the fact breaks the
      "this is what actually happened" claim, so it must break the build.
  R2  RE-FETCHABLE, NON-AUTHORED ORIGIN. ``capture.source`` matches one of the
      locator grammars in :data:`LOCATOR_GRAMMARS`. This is the discriminator
      against a hand-typed approximation: an invented payload has no locator
      that resolves. Free-text provenance ("copied from the live body", "same
      shape as prod") is REJECTED -- that prose is exactly what the OMN-15525
      fix wrote above a fixture it had typed by hand.
  R3  REAL INCIDENT. ``incident`` is an ``OMN-<n>`` ticket or an
      ``<owner>/<repo>#<n>`` pull request.
  R4  THE CASE IS ACTUALLY CONSUMED. ``test`` names a file that exists and that
      references the fixture path. A registry entry nobody reads is paperwork.
  R5  WOULD HAVE CAUGHT IT. The case pins the verdict the buggy guard got
      WRONG, and ``regression_class`` says which direction it got wrong:
        ``false_green`` -- the guard said OK on a real BAD input, so the case's
          ``guard_verdict_on_artifact`` must be ``reject`` (the pin outage: the
          validator passed ``879d6fc6``).
        ``false_red``  -- the guard said FAIL on a real GOOD input, so the
          verdict must be ``accept`` (the health alert: CRITICAL on all three
          lanes against a healthy fleet). A ``false_red`` case must also name a
          ``discriminator`` test, because an accept-only proof cannot tell a
          working guard from one that is stuck open.

Coverage is enforced by two ratchets plus a default-deny:

  * ``scope.required_guards`` -- each entry MUST have >=1 valid case. Append-only.
    An entry whose guard file does not exist yet is reported ``PENDING`` and does
    not fail: that is how a requirement is armed *before* the guard lands.
  * ``scope.debt_baseline`` -- the wired guards that have no case yet. This list
    may only shrink. A covered guard left in the baseline fails (the list must
    stay truthful); a baseline entry that is no longer wired fails (delete it).
  * DEFAULT-DENY -- any newly wired guard in neither list and with no case fails.
    **This is the load-bearing property**: a new gate cannot ship without a real
    replay case, which is what stops the shelf growing faster than the proof.

Exit codes: ``0`` every rule holds, ``1`` any violation.

Related: OMN-15547 (this), OMN-15536/OMN-15538 (pins), OMN-15525/OMN-15509
(health alert), OMN-15484/OMN-15483 (merge hold), OMN-15309 (the OCC
admissibility corpus, the closest pre-existing exemplar of the practice).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REGISTRY_REL = "tests/incident_replays/registry.yaml"

# --------------------------------------------------------------------------
# Wired-guard inventory
# --------------------------------------------------------------------------
# A guard counts as WIRED when a repo-local script under one of these roots is
# named by a pre-commit hook entry or by a workflow `run:` line. Restricting to
# these roots keeps the inventory to things that are plausibly enforcement and
# keeps it deterministic -- it reads only committed files.
GUARD_ROOTS: tuple[str, ...] = ("scripts/", "deploy/")
GUARD_SUFFIXES: tuple[str, ...] = (".py", ".sh")
# Paths inside a guard root that are not themselves guards.
GUARD_EXCLUDE_PARTS: tuple[str, ...] = ("/tests/", "/__pycache__/", "/test_")

_PATH_RE = re.compile(r"(?:scripts|deploy)/[A-Za-z0-9_.@/-]+\.(?:py|sh)")

# --------------------------------------------------------------------------
# R2 -- locator grammars. Each names an origin somebody else can re-fetch.
# --------------------------------------------------------------------------
LOCATOR_GRAMMARS: dict[str, re.Pattern[str]] = {
    # A REST path that pins ONE specific resource -- either a numeric id
    # (run/PR/check/job) or a 40-hex object. A path that only names a
    # collection ("…/actions/runs") is refused: it does not identify what was
    # captured, so it cannot be re-fetched to compare.
    #   gh-api:repos/OmniNode-ai/omnibase_infra/actions/runs/30574058377/jobs
    #   gh-api:repos/OmniNode-ai/omnimarket/compare/dev...879d6fc6825f8764…
    "gh-api": re.compile(
        r"^gh-api:[A-Za-z0-9._/-]*(?:/\d{2,}|[0-9a-f]{40})[A-Za-z0-9._/?=&,-]*$"
    ),
    # git-object:OmniNode-ai/omnimarket@879d6fc6...:.github/workflows/x.yml
    "git-object": re.compile(
        r"^git-object:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}:[A-Za-z0-9_./-]+$"
    ),
    # host-file:omni-201-ts:/data/maintenance/bin/report.sh
    "host-file": re.compile(r"^host-file:[A-Za-z0-9_.-]+:/[A-Za-z0-9_./+-]+$"),
    # live-http:omni-201-ts:8085/health
    "live-http": re.compile(r"^live-http:[A-Za-z0-9_.-]+:\d{2,5}/[A-Za-z0-9_./?=&-]*$"),
    # ci-artifact:30574058377/coverage-shard-3
    "ci-artifact": re.compile(r"^ci-artifact:\d{6,}/[A-Za-z0-9_.-]+$"),
    # container-probe:ghcr.io/omninode-ai/omnibase-infra-runtime@sha256:<64hex>:/app/config
    #
    # Added by OMN-15676, which could not otherwise be replayed at all. That
    # incident's failing artifact lives INSIDE a built image: runner_fleet.yaml
    # was tracked, valid and referenced, and absent only from the image, so
    # every locator above points at a surface that was correct at the time.
    # The image MUST be digest-pinned -- a tag is mutable, so a tag-locator
    # would let the bytes behind a case silently change and is exactly the
    # "capture that stops being the capture" R1 exists to prevent. Re-fetch:
    #   docker run --rm --entrypoint sh <ref> -c 'ls -R <path>'
    "container-probe": re.compile(
        r"^container-probe:[A-Za-z0-9._/-]+@sha256:[0-9a-f]{64}:/[A-Za-z0-9_./,+-]*$"
    ),
}

INCIDENT_RE = re.compile(r"^(?:OMN-\d+|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# R1 -- a capture must end in this suffix. Formatting hooks key off the
# trailing extension, so a fixture left as `.json`/`.yaml`/`.py` is silently
# rewritten by end-of-file-fixer or a formatter and stops being the bytes
# that failed. Learned by execution: that is exactly what happened to these
# fixtures on their first commit here, and R1 is what caught it.
CAPTURED_SUFFIX = ".captured"
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z$")
VALID_VERDICTS = frozenset({"reject", "accept"})
# R5 -- which direction the buggy guard got wrong, and the verdict that pins it.
REGRESSION_CLASSES: dict[str, str] = {
    "false_green": "reject",  # guard said OK on a real BAD input
    "false_red": "accept",  # guard said FAIL on a real GOOD input
}


@dataclass
class Finding:
    rule: str
    subject: str
    detail: str

    def render(self) -> str:
        return f"  [{self.rule}] {self.subject}\n      {self.detail}"


@dataclass
class Result:
    findings: list[Finding] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    covered: set[str] = field(default_factory=set)
    wired: set[str] = field(default_factory=set)

    def fail(self, rule: str, subject: str, detail: str) -> None:
        self.findings.append(Finding(rule, subject, detail))


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _is_guardish(rel: str) -> bool:
    if not rel.startswith(GUARD_ROOTS):
        return False
    if not rel.endswith(GUARD_SUFFIXES):
        return False
    probe = "/" + rel
    return not any(part in probe for part in GUARD_EXCLUDE_PARTS)


def wired_guards(repo_root: Path) -> set[str]:
    """Repo-relative paths of scripts wired as pre-commit or workflow enforcement."""
    found: set[str] = set()

    def consider(text: str) -> None:
        for match in _PATH_RE.finditer(text):
            rel = match.group(0)
            if not _is_guardish(rel):
                continue
            if (repo_root / rel).is_file():
                found.add(rel)

    consider(_read(repo_root / ".pre-commit-config.yaml"))
    workflows = repo_root / ".github" / "workflows"
    if workflows.is_dir():
        for wf in sorted(workflows.iterdir()):
            if wf.suffix in {".yml", ".yaml"}:
                consider(_read(wf))
    return found


# --------------------------------------------------------------------------
# Case validation (R1-R5)
# --------------------------------------------------------------------------
def _validate_case(repo_root: Path, idx: int, case: Any, result: Result) -> str | None:
    """Validate one registry case. Returns the guard path if the case is REAL."""
    label = f"cases[{idx}]"
    if not isinstance(case, dict):
        result.fail("SCHEMA", label, "case must be a mapping")
        return None

    case_id = case.get("id") or label
    label = f"case {case_id}"

    guard = case.get("guard")
    if not isinstance(guard, str) or not guard:
        result.fail("SCHEMA", label, "missing required key `guard`")
        return None
    if not (repo_root / guard).exists():
        result.fail(
            "SCHEMA",
            label,
            f"`guard: {guard}` does not exist in this repo -- a case cannot "
            "cover a guard that is not here",
        )
        return None

    ok = True

    # R3 -- real incident id.
    incident = case.get("incident")
    if not isinstance(incident, str) or not INCIDENT_RE.match(incident):
        result.fail(
            "R3",
            label,
            f"`incident: {incident!r}` must be `OMN-<n>` or `<owner>/<repo>#<n>`; "
            "a replay case has to name the failure it replays",
        )
        ok = False

    # R1 -- committed bytes, unmodified.
    artifact = case.get("artifact")
    if not isinstance(artifact, dict):
        result.fail("R1", label, "missing `artifact:` mapping")
        ok = False
    else:
        fixture = artifact.get("fixture")
        declared = artifact.get("sha256")
        if not isinstance(fixture, str) or not fixture:
            result.fail(
                "R1",
                label,
                "`artifact.fixture` must name a committed FILE. A literal typed "
                "into the test is not a capture -- that is the OMN-15525 defect.",
            )
            ok = False
        elif not fixture.endswith(CAPTURED_SUFFIX):
            result.fail(
                "R1",
                label,
                f"`artifact.fixture: {fixture}` must end in `{CAPTURED_SUFFIX}` so "
                "no formatter claims it. A capture left as `.json`/`.yaml`/`.py` "
                "gets rewritten by end-of-file-fixer or a formatter and quietly "
                "stops being the bytes that failed.",
            )
            ok = False
        else:
            fpath = repo_root / fixture
            if not fpath.is_file():
                result.fail(
                    "R1", label, f"`artifact.fixture: {fixture}` does not exist"
                )
                ok = False
            elif not isinstance(declared, str) or not SHA256_RE.match(declared):
                result.fail(
                    "R1", label, "`artifact.sha256` must be 64 lowercase hex digits"
                )
                ok = False
            else:
                actual = hashlib.sha256(fpath.read_bytes()).hexdigest()
                if actual != declared:
                    result.fail(
                        "R1",
                        label,
                        f"{fixture} has been modified since capture: "
                        f"declared {declared}, actual {actual}. An edited artifact "
                        "is no longer the thing that failed.",
                    )
                    ok = False

    # R2 -- re-fetchable, non-authored origin.
    capture = case.get("capture")
    if not isinstance(capture, dict):
        result.fail("R2", label, "missing `capture:` mapping")
        ok = False
    else:
        source = capture.get("source")
        if not isinstance(source, str) or not any(
            pattern.match(source) for pattern in LOCATOR_GRAMMARS.values()
        ):
            result.fail(
                "R2",
                label,
                f"`capture.source: {source!r}` is not a re-fetchable locator. "
                f"Allowed grammars: {', '.join(sorted(LOCATOR_GRAMMARS))}. "
                "Free-text provenance is rejected on purpose -- a hand-typed "
                "fixture has no locator that resolves.",
            )
            ok = False
        # PyYAML resolves an unquoted ISO-8601 scalar to a datetime, so accept
        # both spellings rather than failing an author for YAML's type coercion.
        captured_at = capture.get("captured_at")
        if isinstance(captured_at, datetime):
            captured_at = captured_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        if not isinstance(captured_at, str) or not TIMESTAMP_RE.match(captured_at):
            result.fail(
                "R2", label, "`capture.captured_at` must be an ISO-8601 UTC timestamp"
            )
            ok = False

    # R5 -- would-have-caught, in the direction the guard actually got wrong.
    verdict = case.get("guard_verdict_on_artifact")
    regression_class = case.get("regression_class")
    if verdict not in VALID_VERDICTS:
        result.fail(
            "R5",
            label,
            f"`guard_verdict_on_artifact` must be one of {sorted(VALID_VERDICTS)}",
        )
        ok = False
    if regression_class not in REGRESSION_CLASSES:
        result.fail(
            "R5",
            label,
            f"`regression_class` must be one of {sorted(REGRESSION_CLASSES)}: "
            "which direction did the buggy guard get wrong?",
        )
        ok = False
    elif verdict != REGRESSION_CLASSES[regression_class]:
        result.fail(
            "R5",
            label,
            f"`regression_class: {regression_class}` requires "
            f"`guard_verdict_on_artifact: {REGRESSION_CLASSES[regression_class]}`, "
            f"got {verdict!r}",
        )
        ok = False
    elif regression_class == "false_red":
        discriminator = case.get("discriminator")
        if not isinstance(discriminator, str) or not discriminator:
            result.fail(
                "R5",
                label,
                "a `false_red` case proves only that the guard ACCEPTS a real "
                "good input, which a stuck-open guard also does. Name a "
                "`discriminator:` test that proves the same guard still says NO.",
            )
            ok = False
        elif not (repo_root / discriminator.split("::", 1)[0]).is_file():
            result.fail(
                "R5",
                label,
                f"`discriminator: {discriminator}` -- file does not exist",
            )
            ok = False

    # R4 -- the case is actually consumed by a test.
    test_ref = case.get("test")
    if not isinstance(test_ref, str) or not test_ref:
        result.fail("R4", label, "missing `test:` node id")
        ok = False
    else:
        test_file = test_ref.split("::", 1)[0]
        tpath = repo_root / test_file
        if not tpath.is_file():
            result.fail(
                "R4", label, f"`test: {test_ref}` -- {test_file} does not exist"
            )
            ok = False
        elif isinstance(artifact, dict) and isinstance(artifact.get("fixture"), str):
            fixture = artifact["fixture"]
            body = _read(tpath)
            needle = Path(fixture).name
            if needle not in body and fixture not in body:
                result.fail(
                    "R4",
                    label,
                    f"{test_file} never references {needle}: the registry claims "
                    "a replay the test does not perform",
                )
                ok = False

    return guard if ok else None


# --------------------------------------------------------------------------
def evaluate(repo_root: Path) -> Result:
    result = Result()
    registry_path = repo_root / REGISTRY_REL

    if not registry_path.is_file():
        result.fail("REGISTRY", REGISTRY_REL, "registry file is missing")
        return result

    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        result.fail("REGISTRY", REGISTRY_REL, f"unparseable: {exc}")
        return result

    scope = registry.get("scope") or {}
    required = list(scope.get("required_guards") or [])
    baseline = list(scope.get("debt_baseline") or [])
    cases = list(registry.get("cases") or [])

    for idx, case in enumerate(cases):
        guard = _validate_case(repo_root, idx, case, result)
        if guard:
            result.covered.add(guard)

    result.wired = wired_guards(repo_root)

    # required_guards -- hard requirement, with a PENDING escape only for a
    # guard that does not exist yet (arming a requirement before the guard lands).
    for guard in required:
        if not (repo_root / guard).exists():
            result.pending.append(guard)
            continue
        if guard not in result.covered:
            result.fail(
                "COVERAGE",
                guard,
                "listed in scope.required_guards but has no valid incident replay "
                f"case. Add one to {REGISTRY_REL} (see the module docstring for "
                "R1-R5).",
            )

    # The baseline must stay truthful in both directions.
    baseline_set = set(baseline)
    for guard in sorted(baseline_set & result.covered):
        result.fail(
            "RATCHET",
            guard,
            "is covered by a replay case but still listed in scope.debt_baseline. "
            "Remove the line -- the baseline may only shrink, and it has to be "
            "readable as the real outstanding debt.",
        )
    for guard in sorted(baseline_set - result.wired):
        if (repo_root / guard).exists():
            result.fail(
                "RATCHET",
                guard,
                "is in scope.debt_baseline but is no longer wired as enforcement. "
                "Delete the line.",
            )
        else:
            result.fail(
                "RATCHET",
                guard,
                "is in scope.debt_baseline but does not exist. Delete the line.",
            )

    # DEFAULT-DENY -- the load-bearing rule.
    known = baseline_set | set(required) | result.covered
    for guard in sorted(result.wired - known):
        result.fail(
            "DEFAULT-DENY",
            guard,
            "is wired as enforcement but carries no incident replay case and is "
            f"not in scope.debt_baseline. A NEW guard must ship with a real "
            f"regression case sourced from an actual failure -- add it to "
            f"{REGISTRY_REL}. Baselining a genuinely new guard instead of "
            "replaying it is the behaviour this rule exists to stop.",
        )

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repository root to audit (default: this script's repo)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="ignored; accepted so the check can run as a pre-commit hook",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    result = evaluate(repo_root)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not result.findings,
                    "wired_guards": len(result.wired),
                    "covered_guards": sorted(result.covered),
                    "pending_guards": sorted(result.pending),
                    "findings": [
                        {"rule": f.rule, "subject": f.subject, "detail": f.detail}
                        for f in result.findings
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if result.findings else 0

    print(
        f"incident-replay coverage: {len(result.covered)} guard(s) covered, "
        f"{len(result.wired)} wired, {len(result.pending)} pending"
    )
    for guard in sorted(result.pending):
        print(f"  PENDING {guard} (required, guard not present yet)")
    if result.findings:
        print(f"\nFAIL -- {len(result.findings)} violation(s):\n")
        for finding in result.findings:
            print(finding.render())
        print(
            "\nA guard that has never been replayed against the real incident it "
            "exists to catch is decorative (OMN-15547)."
        )
        return 1
    print("OK -- every required guard replays a real incident.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
