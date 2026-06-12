# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Start-period floor validator for migration-completion gates (OMN-12973).

A *migration-completion gate* is a catalog service that:

1. declares a healthcheck whose pass condition is "migrations are complete", and
2. waits on one or more sibling services via
   ``service_completed_successfully`` (the migration runners) before its own
   health can pass.

The canonical example is ``migration-gate``: a long-running sentinel
(``while true; sleep 3600``) whose healthcheck polls
``db_metadata.migrations_complete``. It is *not* a one-shot — it stays alive so
downstream runtime services can gate on it via ``service_healthy``.

Failure mode this validator prevents (observed in prod 2026-06-11, OMN-12973):
a too-short ``start_period_s`` lets the container leave Docker's ``starting``
grace window before the migrations it polls for have actually completed. Past
the grace window, the still-failing healthcheck is reported ``UNHEALTHY`` until
migrations finish (~2 min on a cold prod volume), even though nothing is wrong —
the gate is doing exactly its job. ``start_period_s`` must therefore cover the
real migration window so a still-applying gate stays in ``health: starting``.

This is a forward-only ratchet: it asserts a floor, never an exact value.
Raising the floor is a deliberate config decision; lowering a gate's
``start_period_s`` below the floor is a regression and fails CI + pre-commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibase_infra.docker.catalog.enum_depends_on_condition import (
    EnumDependsOnCondition,
)
from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest, HealthCheck

# Floor (seconds) for a migration-completion gate's healthcheck start_period.
# Derived from the observed prod migration window (OMN-12973): the gate started
# at 09:35:22 and intelligence-migration finished at 09:37:18 — ~116s. 120s is
# the enforced floor; the migration-gate manifest ships 180s for margin.
MIGRATION_GATE_START_PERIOD_FLOOR_S = 120

# Substring that identifies a healthcheck whose pass-condition is "migrations
# are complete". The canonical migration sentinel polls
# ``check_migrations_complete.sh``; its presence in the healthcheck test is the
# discriminating signal that the service's health *is* migration completion (as
# opposed to a normal app service that merely depends on a migration runner
# finishing — e.g. intelligence-api, whose healthcheck is an HTTP liveness probe).
_MIGRATION_COMPLETION_PROBE_MARKER = "check_migrations_complete"


def _healthcheck_polls_migration_completion(healthcheck: HealthCheck) -> bool:
    """True if the healthcheck command's pass-condition is migration completion.

    ``HealthCheck.test`` is either a shell string (CMD-SHELL form) or an argv
    list (CMD form); both are joined into one string before scanning for the
    migration-completion probe marker.
    """
    test = healthcheck.test
    command = test if isinstance(test, str) else " ".join(test)
    return _MIGRATION_COMPLETION_PROBE_MARKER in command


def is_migration_completion_gate(manifest: CatalogManifest) -> bool:
    """True if ``manifest`` is a sentinel whose own health *is* migration completion.

    The discriminating signature has two parts, both required:

    1. a healthcheck whose pass-condition is "migrations are complete" (its
       command invokes the migration-completion probe), and
    2. at least one dependency with
       ``condition == service_completed_successfully`` (it waits on the
       migration runner one-shots).

    Part (2) alone is NOT sufficient: ordinary application services (e.g.
    ``intelligence-api``) also depend on a migration runner completing, but
    their healthcheck is a liveness probe with a legitimately short
    ``start_period``. Only the sentinel — whose healthcheck literally polls
    migration completion — must hold ``starting`` for the full migration window,
    so only it is subject to the start_period floor.
    """
    if manifest.healthcheck is None:
        return False
    if not _healthcheck_polls_migration_completion(manifest.healthcheck):
        return False
    return any(
        dep.condition is EnumDependsOnCondition.SERVICE_COMPLETED_SUCCESSFULLY
        for dep in manifest.depends_on
    )


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class StartPeriodViolation:
    """A single migration-gate service whose start_period is below the floor."""

    service: str
    start_period_s: int
    floor_s: int

    def message(self) -> str:
        return (
            f"migration-completion gate '{self.service}' has "
            f"healthcheck.start_period_s={self.start_period_s}, below the "
            f"required floor of {self.floor_s}s. A start_period shorter than the "
            f"real migration window causes the gate to be reported UNHEALTHY "
            f"while migrations are still applying (OMN-12973). Raise "
            f"start_period_s to >= {self.floor_s}."
        )


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class StartPeriodValidationResult:
    """Result of validating migration-gate start_period floors."""

    ok: bool
    violations: list[StartPeriodViolation] = field(default_factory=list)

    def report(self) -> str:
        if self.ok:
            return "all migration-completion gates satisfy the start_period floor"
        return "\n".join(v.message() for v in self.violations)


def validate_migration_gate_start_period(
    manifests: dict[str, CatalogManifest],
    floor_s: int = MIGRATION_GATE_START_PERIOD_FLOOR_S,
) -> StartPeriodValidationResult:
    """Assert every migration-completion gate meets the start_period floor.

    Args:
        manifests: catalog manifests keyed by service name.
        floor_s: minimum allowed ``start_period_s`` for a migration gate.

    Returns:
        A result with ``ok=False`` and one violation per offending gate.
    """
    violations: list[StartPeriodViolation] = []
    for name in sorted(manifests):
        manifest = manifests[name]
        if not is_migration_completion_gate(manifest):
            continue
        # healthcheck is non-None here (guaranteed by is_migration_completion_gate)
        assert manifest.healthcheck is not None
        start_period_s = manifest.healthcheck.start_period_s
        if start_period_s < floor_s:
            violations.append(
                StartPeriodViolation(
                    service=name,
                    start_period_s=start_period_s,
                    floor_s=floor_s,
                )
            )
    return StartPeriodValidationResult(ok=len(violations) == 0, violations=violations)
