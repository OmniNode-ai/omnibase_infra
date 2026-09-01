# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose proof teardown / image-retention policy gate (OMN-16367, plan item 9).

Every CI step that builds a per-run Docker Compose project (``docker compose ...
up --build``) mints images that nothing else can safely reap: they are tagged
(never dangling), uniquely named per run (never enumerable in a static keep
list), and tagged ``latest`` (hard-kept by the disk GC's first guard). The
2026-08-21 audit found ~2,986 such orphans (~2.2 TB) filling ``/data`` on
``.201`` — the fleet-wide ENOSPC root cause.

This gate enforces the two eviction-at-source invariants from
``docs/plans/2026-08-21-201-ci-image-retention-permanent-fix-plan.md``:

1. every ``up --build`` step has a later teardown step in the same job, bound
   to the same compose file, with ``if: always()``, carrying ``--volumes`` and
   ``--rmi local`` (and never ``--rmi all``, which would remove shared pulled
   bases such as ``postgres:16-alpine``);
2. every ``build:``-only service in a live-tree compose file referenced by such
   a step carries ``org.omninode.retention: disposable`` under ``build.labels``
   (the ownership claim the disk-GC classifier keys on), while pulled services
   and the runtime-lane compose files (prod / stability-test / judge / infra)
   must never carry that label.

Deliberate exceptions carry a ``# compose-image-retention-ok: <reason>``
comment within :data:`ANNOTATION_WINDOW` lines of the flagged line, following
the ``no-raw-prod-bypass`` gate's annotation convention.

Compose files referenced through pinned historical checkouts (``working-directory``
under ``.proof-dependencies/``) do not exist in the live tree; their label state
is frozen at the pinned ref, so only the teardown invariant applies to them —
the label gap there is what the plan's prefix bridge (item 3, wave 2) covers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

RETENTION_LABEL = "org.omninode.retention"
DISPOSABLE = "disposable"
ANNOTATION = "compose-image-retention-ok:"
ANNOTATION_WINDOW = 12

# Runtime-lane compose files: a per-run "disposable" claim on any of these
# would hand the GC a licence to delete a lane image. Checked only if present.
RUNTIME_LANE_COMPOSE = (
    "docker/docker-compose.prod.yml",
    "docker/docker-compose.stability-test.yml",
    "docker/docker-compose.judge.yml",
    "docker/docker-compose.infra.yml",
)

# The four live-tree proof compose files that MUST be covered by the label
# check. If workflow parsing ever stops resolving them, the coverage test
# fails rather than silently passing on an empty set.
EXPECTED_LABELED_COMPOSE = (
    "docker/application-acl-proof/compose.yml",
    "docker/application-domain-enforcement/compose.yml",
    "docker/domain-adapter-proof/compose.yml",
    "docker/legacy-rds-fixture/compose.yml",
)

_COMPOSE_REF_RE = re.compile(r"-f\s+(\S+\.ya?ml)")


def _normalize(run: str) -> str:
    """Collapse folded/piped run blocks to a single-space string."""
    return " ".join(run.replace("\\\n", " ").split())


def _annotated_lines(text: str) -> set[int]:
    return {idx for idx, line in enumerate(text.splitlines()) if ANNOTATION in line}


def _is_annotated(text: str, needle: str, annotated: set[int]) -> bool:
    """True if any raw line containing ``needle`` sits within the annotation window."""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if needle in line:
            lo = max(0, idx - ANNOTATION_WINDOW)
            hi = idx + ANNOTATION_WINDOW
            if any(lo <= a <= hi for a in annotated):
                return True
    return False


def _step_working_directory(job: dict, step: dict) -> str:
    wd = step.get("working-directory")
    if isinstance(wd, str):
        return wd
    defaults = job.get("defaults") or {}
    run_defaults = defaults.get("run") or {}
    wd = run_defaults.get("working-directory")
    return wd if isinstance(wd, str) else ""


def check_workflow_text(name: str, text: str) -> list[str]:
    """Return teardown-policy violations for one workflow file's text."""
    violations: list[str] = []
    annotated = _annotated_lines(text)

    # Global ban: --rmi all removes shared pulled base images (plan §2.7).
    for idx, line in enumerate(text.splitlines()):
        if "--rmi all" in line:
            lo = max(0, idx - ANNOTATION_WINDOW)
            hi = idx + ANNOTATION_WINDOW
            if not any(lo <= a <= hi for a in annotated):
                violations.append(
                    f"{name}: line {idx + 1} uses '--rmi all' — forbidden; "
                    f"use '--rmi local' (see plan §2.7)"
                )

    data = yaml.safe_load(text)
    jobs = (data or {}).get("jobs") or {}
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        steps = [s for s in (job.get("steps") or []) if isinstance(s, dict)]
        for idx, step in enumerate(steps):
            run = step.get("run")
            if not isinstance(run, str):
                continue
            norm = _normalize(run)
            if (
                "docker compose" not in norm
                or not re.search(r"\bup\b", norm)
                or "--build" not in norm
            ):
                continue
            refs = _COMPOSE_REF_RE.findall(norm)
            if not refs:
                continue
            if any(_is_annotated(text, ref, annotated) for ref in refs):
                continue
            step_label = step.get("name") or f"step #{idx + 1}"
            teardown = None
            for later in steps[idx + 1 :]:
                lrun = later.get("run")
                if not isinstance(lrun, str):
                    continue
                lnorm = _normalize(lrun)
                if (
                    "docker compose" in lnorm
                    and re.search(r"\bdown\b", lnorm)
                    and any(ref in lnorm for ref in refs)
                ):
                    teardown = later
                    break
            where = f"{name} / job '{job_name}' / '{step_label}'"
            if teardown is None:
                violations.append(
                    f"{where}: 'up --build' has no 'docker compose ... down' "
                    f"teardown for {refs} in the same job"
                )
                continue
            tnorm = _normalize(teardown.get("run", ""))
            if "always()" not in str(teardown.get("if", "")):
                violations.append(f"{where}: teardown step is not gated 'if: always()'")
            if "--volumes" not in tnorm and " -v " not in f" {tnorm} ":
                violations.append(f"{where}: teardown is missing '--volumes'")
            if "--rmi local" not in tnorm:
                violations.append(
                    f"{where}: teardown is missing '--rmi local' — the built "
                    f"per-run images leak onto the runner host without it"
                )
    return violations


def check_compose_text(name: str, text: str) -> list[str]:
    """Return label-policy violations for one proof compose file's text."""
    violations: list[str] = []
    annotated = _annotated_lines(text)
    data = yaml.safe_load(text)
    services = (data or {}).get("services") or {}
    for svc_name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        build = svc.get("build")
        has_image = isinstance(svc.get("image"), str)
        if isinstance(build, dict) and not has_image:
            labels = build.get("labels") or {}
            if not isinstance(labels, dict):
                labels = {}
            if labels.get(RETENTION_LABEL) != DISPOSABLE:
                if not _is_annotated(text, f"{svc_name}:", annotated):
                    violations.append(
                        f"{name}: built service '{svc_name}' is missing "
                        f"'{RETENTION_LABEL}: {DISPOSABLE}' under build.labels"
                    )
        elif has_image and not build:
            svc_labels = svc.get("labels") or {}
            if isinstance(svc_labels, list):
                carries = any(
                    str(item).startswith(f"{RETENTION_LABEL}=") for item in svc_labels
                )
            else:
                carries = RETENTION_LABEL in svc_labels
            if carries:
                violations.append(
                    f"{name}: pulled service '{svc_name}' must not carry "
                    f"'{RETENTION_LABEL}' — the label is a claim by the "
                    f"builder, and nothing built this image"
                )
    return violations


def _iter_workflow_files() -> list[Path]:
    files = sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))
    assert files, f"no workflow files found under {WORKFLOWS_DIR}"
    return files


def _referenced_live_compose_files() -> set[Path]:
    """Live-tree compose files referenced by any 'up --build' step."""
    found: set[Path] = set()
    for wf in _iter_workflow_files():
        text = wf.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        jobs = (data or {}).get("jobs") or {}
        for job in jobs.values():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                run = step.get("run")
                if not isinstance(run, str):
                    continue
                norm = _normalize(run)
                if (
                    "docker compose" not in norm
                    or not re.search(r"\bup\b", norm)
                    or "--build" not in norm
                ):
                    continue
                wd = _step_working_directory(job, step)
                for ref in _COMPOSE_REF_RE.findall(norm):
                    candidate = (REPO_ROOT / wd / ref) if wd else (REPO_ROOT / ref)
                    if candidate.is_file():
                        found.add(candidate.resolve())
    return found


@pytest.mark.unit
class TestLiveTreePolicy:
    def test_every_up_build_step_has_evicting_always_teardown(self) -> None:
        violations: list[str] = []
        for wf in _iter_workflow_files():
            violations.extend(
                check_workflow_text(wf.name, wf.read_text(encoding="utf-8"))
            )
        assert not violations, "\n".join(violations)

    def test_referenced_live_compose_files_carry_disposable_labels(
        self,
    ) -> None:
        referenced = _referenced_live_compose_files()
        expected = {(REPO_ROOT / rel).resolve() for rel in EXPECTED_LABELED_COMPOSE}
        missing = expected - referenced
        assert not missing, (
            f"expected proof compose files not resolved from workflows "
            f"(parser drift?): {sorted(str(p) for p in missing)}"
        )
        violations: list[str] = []
        for path in sorted(referenced):
            violations.extend(
                check_compose_text(
                    str(path.relative_to(REPO_ROOT)),
                    path.read_text(encoding="utf-8"),
                )
            )
        assert not violations, "\n".join(violations)

    def test_runtime_lane_compose_never_carries_disposable_label(self) -> None:
        violations: list[str] = []
        for rel in RUNTIME_LANE_COMPOSE:
            path = REPO_ROOT / rel
            if not path.is_file():
                continue
            if RETENTION_LABEL in path.read_text(encoding="utf-8"):
                violations.append(
                    f"{rel}: contains '{RETENTION_LABEL}' — runtime-lane "
                    f"images must never be claimed disposable"
                )
        assert not violations, "\n".join(violations)


FIXTURE_UP_NO_RMI = """
jobs:
  proof:
    runs-on: ubuntu-latest
    steps:
      - name: Build proof
        run: >-
          docker compose -f docker/x/compose.yml up --build
      - name: Teardown
        if: always()
        run: >-
          docker compose -f docker/x/compose.yml down --volumes --remove-orphans
"""

FIXTURE_RMI_ALL = """
jobs:
  proof:
    runs-on: ubuntu-latest
    steps:
      - name: Build proof
        run: >-
          docker compose -f docker/x/compose.yml up --build
      - name: Teardown
        if: always()
        run: >-
          docker compose -f docker/x/compose.yml down --volumes --rmi all
"""

FIXTURE_NO_TEARDOWN = """
jobs:
  proof:
    runs-on: ubuntu-latest
    steps:
      - name: Build proof
        run: >-
          docker compose -f docker/x/compose.yml up --build
"""

FIXTURE_ANNOTATED_EXCEPTION = """
jobs:
  proof:
    runs-on: ubuntu-latest
    steps:
      - name: Build proof
        # compose-image-retention-ok: fixture demonstrating a recorded exception
        run: >-
          docker compose -f docker/x/compose.yml up --build
"""

FIXTURE_COMPOSE_UNLABELED = """
services:
  proof:
    build:
      context: .
      dockerfile: Dockerfile
"""

FIXTURE_COMPOSE_LABELED = """
services:
  proof:
    build:
      context: .
      dockerfile: Dockerfile
      labels:
        org.omninode.retention: disposable
        org.omninode.run-id: "${GITHUB_RUN_ID:-local}"
        org.omninode.workflow: "${GITHUB_WORKFLOW:-local}"
  postgres:
    image: postgres:16-alpine
"""

FIXTURE_COMPOSE_PULLED_LABELED = """
services:
  postgres:
    image: postgres:16-alpine
    labels:
      org.omninode.retention: disposable
"""

FIXTURE_GOOD_WORKFLOW = """
jobs:
  proof:
    runs-on: ubuntu-latest
    steps:
      - name: Build proof
        run: >-
          docker compose -f docker/x/compose.yml up --build
      - name: Teardown
        if: always()
        run: >-
          docker compose -f docker/x/compose.yml
          down --volumes --remove-orphans --rmi local
"""


@pytest.mark.unit
class TestPolicyFixtures:
    """The four failure fixtures from the plan's item-9 acceptance criteria."""

    def test_up_build_without_rmi_local_fails(self) -> None:
        violations = check_workflow_text("fixture.yml", FIXTURE_UP_NO_RMI)
        assert any("--rmi local" in v for v in violations)

    def test_rmi_all_teardown_fails(self) -> None:
        violations = check_workflow_text("fixture.yml", FIXTURE_RMI_ALL)
        assert any("--rmi all" in v for v in violations)

    def test_missing_teardown_fails(self) -> None:
        violations = check_workflow_text("fixture.yml", FIXTURE_NO_TEARDOWN)
        assert any("no 'docker compose ... down'" in v for v in violations)

    def test_unlabeled_built_service_fails(self) -> None:
        violations = check_compose_text(
            "fixture-compose.yml", FIXTURE_COMPOSE_UNLABELED
        )
        assert any("missing" in v for v in violations)

    def test_labeled_runtime_lane_service_fails(self) -> None:
        violations = check_compose_text(
            "fixture-lane.yml", FIXTURE_COMPOSE_PULLED_LABELED
        )
        assert any("must not carry" in v for v in violations)

    def test_compliant_workflow_and_compose_pass(self) -> None:
        assert check_workflow_text("good.yml", FIXTURE_GOOD_WORKFLOW) == []
        assert check_compose_text("good-compose.yml", FIXTURE_COMPOSE_LABELED) == []

    def test_annotated_exception_passes(self) -> None:
        assert check_workflow_text("annotated.yml", FIXTURE_ANNOTATED_EXCEPTION) == []
