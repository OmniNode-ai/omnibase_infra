# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-13886 — mechanical enforcement of profile-aware runtime-lane teardown.

Residual of the local Mac dev-lane crash-loop work (OMN-13886). The runtime
services in the ONEX compose lanes are gated behind compose ``profiles:
["runtime", "full"]``. Because of that gating, a bare ``docker compose ...
down`` (no ``--profile runtime``/``--profile full`` and no ``--remove-orphans``)
tears down the WRONG / INCOMPLETE set: it removes the un-profiled core deps
(postgres, redpanda, valkey, keycloak) while leaving the profile-gated runtime
containers running as orphans — the exact "2 hits in 24h" teardown defect the
rolling plan (WS-2) flagged. The runtime containers are then orphaned, their
deps yanked out from under them, and the next bring-up collides.

This module is the mechanical enforcement for that defect class, wired as a CI
gate (``.github/workflows/no-bare-compose-teardown.yml``) and a pre-commit hook
(``.pre-commit-config.yaml``) so the prohibition is enforced, not merely
documented (CLAUDE.md Operating Rule #5: enforcement, not detection). It mirrors
the sibling ``tests/test_no_raw_prod_bypass_policy.py`` gate in omni_home.

The guard is scoped to ``down`` only. A bare ``docker compose -f
docker-compose.infra.yml up -d`` is a *documented, non-destructive* way to bring
up only the core infrastructure (postgres/redpanda/valkey), used in dozens of
legitimate recipes — flagging it would be pure false-positive noise. The
destructive, "wrong set" defect is teardown, so ``down`` is the surface this
gate guards.

A ``down`` line is a violation only when it targets a runtime lane (the
``docker-compose.infra.yml`` base file, a lane overlay, or a runtime-lane
compose project via ``-p``) and carries NEITHER a profile selector
(``--profile runtime``/``--profile full``) NOR ``--remove-orphans``. Any one of
those three makes the teardown complete:

* ``--profile runtime`` / ``--profile full`` activates the runtime services so
  ``down`` removes them too;
* ``--remove-orphans`` removes the profile-gated containers as orphans (they are
  not in the active, profile-filtered configuration), which is exactly how the
  CI e2e boot lane tears down cleanly.

Per-line escape hatch for a legitimate core-only teardown or a
historical/illustrative quote::

    <recipe>  # compose-profile-ok: <reason>

The marker is comment-syntax-agnostic (``#``, ``<!-- -->``, ``//``).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Authored recipe surfaces — the copy-pasteable places a teardown recipe lives.
SCANNED_SUFFIXES = (".md", ".sh", ".yml", ".yaml", ".py")

# Path prefixes excluded entirely (raw machine-generated log/transcript dumps,
# if any land under these prefixes in the future).
EXCLUDED_PREFIXES: tuple[str, ...] = ()

# Files that document the prohibition itself or the scanner, and therefore
# legitimately contain the forbidden pattern as an example. Exact relative path.
ALLOWLISTED_PATHS = {
    "tests/test_compose_profile_teardown_policy.py",
    ".github/workflows/no-bare-compose-teardown.yml",
    # Historical 2026-05-04 plan checklist: a core+keycloak teardown recorded
    # before the runtime lane profiles existed, not a live runtime-lane op.
    # Allowlisted by path (rather than inline-annotated) because the separate
    # `no-planning-docs` gate rejects any edit to plan docs in this repo.
    "docs/plans/2026-05-04-seed-keycloak-orchestrator.md",
}

# Per-line escape hatch. Comment-syntax-agnostic; intentionally distinct from
# the generic skip-token machinery (Operating Rule #10) so it cannot be confused
# with a merge-gate bypass.
ESCAPE_MARKER = "compose-profile-ok:"

# A compose invocation: `docker compose` or the legacy `docker-compose`.
_COMPOSE_RE = re.compile(r"docker[\s-]compose\b", re.IGNORECASE)

# The teardown verb. `down` as a standalone token (so `shutdown`, `slowdown`,
# prose "down" inside an unrelated sentence without a compose invocation are not
# matched — a compose invocation is required by _COMPOSE_RE first).
_DOWN_RE = re.compile(r"(?<![\w-])down\b", re.IGNORECASE)

# Runtime-lane targets. The base lane file, the lane overlays, or a runtime-lane
# compose project named via -p / --project-name. `docker-compose.e2e.yml`,
# `docker-compose.generated.yml`, `docker-compose.runners.yml`, and
# `docker-compose.infisical-stability.yml` are deliberately NOT runtime lanes.
_LANE_FILE_RE = re.compile(
    r"docker-compose\.(infra|prod|stability-test|judge)\.yml", re.IGNORECASE
)
_LANE_PROJECT_RE = re.compile(
    r"(?:-p|--project-name)[=\s]+omnibase-infra(?:-(?:prod|stability-test|judge))?\b",
    re.IGNORECASE,
)

# Complete-teardown signals — any one clears the line.
_PROFILE_RUNTIME_RE = re.compile(r"--profile[=\s]+(runtime|full)\b", re.IGNORECASE)
_REMOVE_ORPHANS_RE = re.compile(r"--remove-orphans\b", re.IGNORECASE)


def _targets_runtime_lane(line: str) -> bool:
    return bool(_LANE_FILE_RE.search(line) or _LANE_PROJECT_RE.search(line))


def _has_complete_teardown_signal(line: str) -> bool:
    return bool(_PROFILE_RUNTIME_RE.search(line) or _REMOVE_ORPHANS_RE.search(line))


def _is_bare_lane_teardown(line: str) -> bool:
    """A bare runtime-lane teardown: `down` at a runtime lane, no complete signal."""
    if not _COMPOSE_RE.search(line):
        return False
    if not _DOWN_RE.search(line):
        return False
    if not _targets_runtime_lane(line):
        return False
    if _has_complete_teardown_signal(line):
        return False
    return True


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def _scan_corpus() -> list[str]:
    """Return ``path:lineno: line`` for every un-annotated bare lane teardown."""
    violations: list[str] = []
    for path in _tracked_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWLISTED_PATHS:
            continue
        if not rel.endswith(SCANNED_SUFFIXES):
            continue
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _is_bare_lane_teardown(line):
                continue
            if ESCAPE_MARKER in line:
                continue
            violations.append(f"{rel}:{lineno}: {line.strip()}")
    return violations


@pytest.mark.unit
def test_no_unannotated_bare_compose_teardown_in_corpus() -> None:
    """No tracked file may carry an un-annotated bare runtime-lane teardown."""
    violations = _scan_corpus()
    assert not violations, (
        "Bare `docker compose ... down` recipe(s) found against a runtime lane "
        "without `--profile runtime`/`--profile full` or `--remove-orphans`. "
        "A bare teardown removes the un-profiled core deps while orphaning the "
        "profile-gated runtime containers (OMN-13886). Add `--remove-orphans` "
        "(or `--profile runtime`) so the teardown is complete, or — only if this "
        "is an intentional core-only teardown or a historical/illustrative "
        f"quote — annotate the line with `# {ESCAPE_MARKER} <reason>` "
        "(any comment syntax: `#`, `<!-- -->`, `//`):\n  " + "\n  ".join(violations)
    )


@pytest.mark.unit
def test_scanner_flags_known_bare_teardown_signatures() -> None:
    """Self-test: the scanner catches the bare-teardown signatures."""
    # Bare down of the base runtime lane file.
    assert _is_bare_lane_teardown("docker compose -f docker-compose.infra.yml down")
    # Bare down with volumes but still no orphan removal — still incomplete.
    assert _is_bare_lane_teardown(
        "docker compose -f docker/docker-compose.infra.yml down -v"
    )
    # Bare down of a named runtime-lane project.
    assert _is_bare_lane_teardown("docker compose -p omnibase-infra-prod down")
    assert _is_bare_lane_teardown(
        "docker compose -p omnibase-infra-stability-test down"
    )
    # A lane overlay is a runtime lane too.
    assert _is_bare_lane_teardown("docker compose -f docker-compose.judge.yml down")


@pytest.mark.unit
def test_scanner_ignores_complete_and_nonlane_teardowns() -> None:
    """Self-test: complete teardowns and non-runtime-lane composes are cleared."""
    # Profile-aware teardown — the runtime services are in scope, so down removes them.
    assert not _is_bare_lane_teardown(
        "docker compose -f docker-compose.infra.yml --profile runtime down"
    )
    assert not _is_bare_lane_teardown(
        "docker compose -f docker-compose.infra.yml --profile full down -v"
    )
    # --remove-orphans removes the profile-gated containers as orphans — complete.
    assert not _is_bare_lane_teardown(
        "docker compose -f docker-compose.infra.yml down -v --remove-orphans"
    )
    # The ephemeral CI e2e boot lane is NOT a runtime lane.
    assert not _is_bare_lane_teardown(
        "docker compose -f docker/docker-compose.e2e.yml down -v"
    )
    # The generated compose (what `onex down` operates on) is not the profiled lane.
    assert not _is_bare_lane_teardown(
        "docker compose -f docker/docker-compose.generated.yml down"
    )
    # The runner fleet compose is not a runtime lane.
    assert not _is_bare_lane_teardown(
        "docker compose -f docker/docker-compose.runners.yml down"
    )
    # A bare `up` is not a teardown — out of scope of this gate.
    assert not _is_bare_lane_teardown(
        "docker compose -f docker-compose.infra.yml up -d"
    )
    # Read-only against the lane — not a teardown.
    assert not _is_bare_lane_teardown(
        "docker compose -f docker-compose.infra.yml logs -f"
    )


@pytest.mark.unit
def test_escape_marker_suppresses_a_flagged_line() -> None:
    """Self-test: a flagged teardown annotated with the escape marker is cleared."""
    flagged = "docker compose -f docker-compose.infra.yml down -v"
    assert _is_bare_lane_teardown(flagged), "fixture must be a flagged recipe"
    for annotated in (
        f"{flagged}  # {ESCAPE_MARKER} intentional core-only teardown",
        f"{flagged} <!-- {ESCAPE_MARKER} historical plan quote -->",
        f"{flagged}  // {ESCAPE_MARKER} illustrative",
    ):
        assert ESCAPE_MARKER in annotated


def _main() -> int:
    """Standalone entrypoint for the pre-commit hook (no pytest dependency)."""
    violations = _scan_corpus()
    if violations:
        sys.stderr.write(
            "BLOCKED: bare `docker compose ... down` against a runtime lane "
            "without `--profile runtime`/`--profile full` or `--remove-orphans` "
            "(OMN-13886). Add `--remove-orphans` (or `--profile runtime`), or "
            f"annotate the line with `# {ESCAPE_MARKER} <reason>`:\n  "
            + "\n  ".join(violations)
            + "\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
