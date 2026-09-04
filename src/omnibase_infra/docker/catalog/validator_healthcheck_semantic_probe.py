# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runtime service's catalog healthcheck must be the SEMANTIC probe (OMN-17883).

OMN-15217 replaced the shallow ``curl --fail .../health`` liveness check with
``onex-container-healthcheck`` because ``curl -sf`` asserts exactly one thing:
the status code is < 400. ``/health`` returns HTTP 200 for a *running but
DEGRADED* runtime by design (``ServiceHealth._handle_health`` sets 200 for
``status="degraded"`` deliberately, so a degradation stays visible without
triggering a restart), so the shallow probe cannot fail on the exact condition
OMN-15217 exists to catch — the stability lane reported ``Up 3 hours (healthy)``
while the runtime logged ``status=DEGRADED contracts=296 errors=4``.

That conversion reached the two hand-written lane composes and never reached the
service catalog, which CLAUDE.md designates as the sanctioned generated path.
Worse than an omission: a compose-level ``healthcheck`` OVERRIDES the
image-level ``HEALTHCHECK``, and ``docker/Dockerfile.runtime`` already bakes the
semantic probe in. So a catalog entry declaring ``curl -sf`` was actively
REPLACING the deep probe the image carries.

This validator is the ratchet for that (Operating Rule 5: enforcement, not
detection). Detection was already possible — nobody was doing it, and the
regression survived from OMN-15217 until OMN-17623 surfaced it. A future manifest
cannot silently revert.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest, HealthCheck

# The layer whose services serve ``ServiceHealth`` on the runtime health port.
# Read off the manifest rather than matched by service name: a runtime service
# added tomorrow under a new name is covered without editing a list here.
RUNTIME_LAYER = EnumInfraLayer.RUNTIME

# The runtime health endpoint, kept for the violation message only. It is
# deliberately NOT the discriminator: the first version of this validator keyed
# coverage on this string appearing in the healthcheck command, which was true of
# `curl -sf http://localhost:8085/health` and FALSE of the semantic probe, whose
# argv carries no port. That validator covered zero services and reported ok
# vacuously — a gate green because it checks nothing, which is the exact class of
# defect this ticket exists to close.
RUNTIME_HEALTH_PORT = "8085"

# Substring identifying the semantic probe. Deliberately the installed path
# rather than the bare name: `Dockerfile.runtime` installs it there and invokes
# it as a FILE (not `python -m`), because importing the package chain costs ~6.8s
# against a 10s probe timeout.
_SEMANTIC_PROBE_MARKER = "onex-container-healthcheck"


def _command_text(healthcheck: HealthCheck) -> str:
    """Flatten either healthcheck form to one string.

    ``HealthCheck.test`` is a shell string (CMD-SHELL form) or an argv list (CMD
    form); both are joined before scanning, mirroring the start_period validator.
    """
    test = healthcheck.test
    return test if isinstance(test, str) else " ".join(test)


def probes_runtime_health(manifest: CatalogManifest) -> bool:
    """True if this manifest's healthcheck asserts the ONEX runtime's health.

    Three parts, all required:

    1. the service declares a healthcheck at all;
    2. it is on the ``runtime`` layer; and
    3. that healthcheck targets the runtime health endpoint — either naming the
       health PORT (the shallow ``curl`` form) or invoking the semantic probe
       (whose argv names no port because :8085 is its own default).

    Part 3 must accept BOTH forms, and getting that wrong is the trap this
    validator fell into on its first draft. Keyed on the port alone, coverage was
    correct while the catalog said ``curl -sf http://localhost:8085/health`` and
    silently dropped to ZERO the moment the fix landed — the gate then reported
    ok having examined nothing, which is the same false-green class the ticket
    exists to close.

    Layer alone is not sufficient either: 18 catalog services are on the runtime
    layer and all 18 share ``runtime:latest``, but they serve different apps on
    different ports (``intelligence-api`` on 8053, ``projection-api`` on 3002).
    Only a healthcheck aimed at the runtime health endpoint is making a claim
    about the ONEX runtime's health, so only those are subject to this rule.
    """
    if manifest.healthcheck is None:
        return False
    if manifest.layer is not RUNTIME_LAYER:
        return False
    command = _command_text(manifest.healthcheck)
    return RUNTIME_HEALTH_PORT in command or _SEMANTIC_PROBE_MARKER in command


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class SemanticProbeViolation:
    """A runtime service whose healthcheck is not the semantic probe."""

    service: str
    command: str

    def message(self) -> str:
        return (
            f"catalog service '{self.service}' is on the '{RUNTIME_LAYER.value}' layer and "
            f"probes :{RUNTIME_HEALTH_PORT}, but its healthcheck is "
            f"{self.command!r} rather than '{_SEMANTIC_PROBE_MARKER}'. "
            f"/health returns 200 for a running-but-DEGRADED runtime by design, so "
            f"a shallow check cannot fail on the condition OMN-15217 exists to "
            f"catch — and because a compose-level healthcheck OVERRIDES the "
            f"image-level HEALTHCHECK, this entry replaces the deep probe "
            f"Dockerfile.runtime already installs. Declare the argv form: "
            f"[python, /usr/local/bin/{_SEMANTIC_PROBE_MARKER}, --degraded-policy, "
            f"<fail|warn>]."
        )


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class SemanticProbeValidationResult:
    """Result of validating runtime-service healthcheck depth."""

    ok: bool
    violations: list[SemanticProbeViolation] = field(default_factory=list)

    def report(self) -> str:
        if self.ok:
            return "all runtime-layer services declare the semantic health probe"
        return "\n".join(v.message() for v in self.violations)


def validate_runtime_semantic_probe(
    manifests: dict[str, CatalogManifest],
) -> SemanticProbeValidationResult:
    """Assert every runtime-layer health probe is the semantic one.

    Args:
        manifests: catalog manifests keyed by service name.

    Returns:
        A result with ``ok=False`` and one violation per offending service.
    """
    violations: list[SemanticProbeViolation] = []
    for name in sorted(manifests):
        manifest = manifests[name]
        if not probes_runtime_health(manifest):
            continue
        # healthcheck is non-None here (guaranteed by probes_runtime_health)
        assert manifest.healthcheck is not None
        command = _command_text(manifest.healthcheck)
        if _SEMANTIC_PROBE_MARKER not in command:
            violations.append(SemanticProbeViolation(service=name, command=command))
    return SemanticProbeValidationResult(ok=len(violations) == 0, violations=violations)
