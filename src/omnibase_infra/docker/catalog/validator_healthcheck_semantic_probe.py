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

import re
import shlex
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

# Matched as an ADDRESS, not as a bare substring. `"8085" in command` also
# matches `18085`, `80853` and `HEALTH_PORT=8085`, pulling a service that never
# claimed runtime health into the ratchet and failing it for a coincidence.
#
# The leading `:` is what makes it an address (`localhost:8085`, `:8085`) rather
# than any occurrence of the digits; the trailing guard rejects `:80853`. A
# digit-boundary-only pattern is not enough -- it still accepts `HEALTH_PORT=8085`,
# which a test in this repo pins.
_RUNTIME_PORT_RE = re.compile(rf":{RUNTIME_HEALTH_PORT}(?![0-9])")

# Substring identifying the semantic probe.
#
# On the timing: the probe is fast because the MODULE is stdlib-only (its own
# docstring: "nothing here imports from omnibase_infra -- a unit test pins the
# property"), starting in ~0.12s against a 10s timeout. Invoking it by path
# rather than `python -m omnibase_infra...` avoids importing the package chain
# to REACH it, but the headroom comes from what the script imports, not from the
# invocation form -- a file that imported the chain would be just as slow.
_SEMANTIC_PROBE_MARKER = "onex-container-healthcheck"

# The probe's own default is `fail` (DEGRADED_POLICY_FAIL), so an invocation
# with no policy is not fail-open. It is still required explicitly: whether a
# DEGRADED runtime should read unhealthy is a per-lane decision, and a manifest
# that leaves it implicit silently inherits whichever default the script
# carries at the time. Enforced because the violation message prescribes it --
# a gate whose message asks for something it does not check is advice, not a
# gate.
_DEGRADED_POLICY_FLAG = "--degraded-policy"
_VALID_DEGRADED_POLICIES = ("fail",)
_URL_FLAG = "--url"


def _command_text(healthcheck: HealthCheck) -> str:
    """Flatten either healthcheck form to one string.

    ``HealthCheck.test`` is a shell string (CMD-SHELL form) or an argv list (CMD
    form); both are joined before scanning, mirroring the start_period validator.
    """
    test = healthcheck.test
    return test if isinstance(test, str) else " ".join(test)


def _command_tokens(healthcheck: HealthCheck) -> list[str]:
    test = healthcheck.test
    if isinstance(test, str):
        return shlex.split(test)
    return list(test)


def _last_option_value(tokens: list[str], flag: str) -> str | None:
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _expected_health_url(manifest: CatalogManifest) -> str | None:
    if manifest.ports is None:
        return f"http://localhost:{RUNTIME_HEALTH_PORT}/health"  # url-authority-ok: container self-probe loopback URL rendered into docker healthcheck argv
    return f"http://localhost:{manifest.ports.internal}/health"  # url-authority-ok: manifest-declared container self-probe loopback URL rendered into docker healthcheck argv


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
    return bool(_RUNTIME_PORT_RE.search(command)) or _SEMANTIC_PROBE_MARKER in command


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class SemanticProbeViolation:
    """A runtime service whose healthcheck is not a well-formed semantic probe."""

    service: str
    command: str
    reason: str = "not_semantic"

    def message(self) -> str:
        if self.reason == "policy_missing":
            return (
                f"catalog service '{self.service}' invokes "
                f"'{_SEMANTIC_PROBE_MARKER}' but declares no explicit "
                f"{_DEGRADED_POLICY_FLAG} {'|'.join(_VALID_DEGRADED_POLICIES)}. "
                f"The script defaults to 'fail', so this is not fail-open -- but "
                f"whether a DEGRADED runtime reads unhealthy is a per-lane "
                f"decision and must be declared, not inherited from whatever "
                f"default the script carries. Command: {self.command!r}"
            )
        if self.reason == "policy_invalid":
            return (
                f"catalog service '{self.service}' invokes "
                f"'{_SEMANTIC_PROBE_MARKER}' with a degraded policy other than "
                f"'fail'. A catalog runtime healthcheck must fail on DEGRADED "
                f"runtime verdicts. Command: {self.command!r}"
            )
        if self.reason == "url_missing":
            return (
                f"catalog service '{self.service}' invokes "
                f"'{_SEMANTIC_PROBE_MARKER}' but declares no explicit {_URL_FLAG} "
                f"matching its manifest health port. Command: {self.command!r}"
            )
        if self.reason == "url_mismatch":
            return (
                f"catalog service '{self.service}' invokes "
                f"'{_SEMANTIC_PROBE_MARKER}' with a {_URL_FLAG} that does not "
                f"match its manifest health port. Command: {self.command!r}"
            )
        return (
            f"catalog service '{self.service}' is on the '{RUNTIME_LAYER.value}' layer and "
            f"probes :{RUNTIME_HEALTH_PORT}, but its healthcheck is "
            f"{self.command!r} rather than '{_SEMANTIC_PROBE_MARKER}'. "
            f"/health returns 200 for a running-but-DEGRADED runtime by design, so "
            f"a shallow check cannot fail on the condition OMN-15217 exists to "
            f"catch — and because a compose-level healthcheck OVERRIDES the "
            f"image-level HEALTHCHECK, this entry replaces the deep probe "
            f"Dockerfile.runtime already installs. Declare the argv form: "
            f"[python, /usr/local/bin/{_SEMANTIC_PROBE_MARKER}, --url, "
            f"<health-url>, --degraded-policy, fail]."
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
            continue
        # The marker alone is not enough: the command must spell out both the
        # health endpoint and the degraded policy that make the semantic probe
        # enforceable from docker inspect output.
        tokens = _command_tokens(manifest.healthcheck)
        expected_url = _expected_health_url(manifest)
        if expected_url is None:
            violations.append(
                SemanticProbeViolation(
                    service=name, command=command, reason="url_missing"
                )
            )
            continue
        actual_url = _last_option_value(tokens, _URL_FLAG)
        if actual_url is None:
            violations.append(
                SemanticProbeViolation(
                    service=name, command=command, reason="url_missing"
                )
            )
            continue
        if actual_url != expected_url:
            violations.append(
                SemanticProbeViolation(
                    service=name, command=command, reason="url_mismatch"
                )
            )
            continue
        policy = _last_option_value(tokens, _DEGRADED_POLICY_FLAG)
        if policy is None:
            violations.append(
                SemanticProbeViolation(
                    service=name, command=command, reason="policy_missing"
                )
            )
            continue
        if policy not in _VALID_DEGRADED_POLICIES:
            violations.append(
                SemanticProbeViolation(
                    service=name, command=command, reason="policy_invalid"
                )
            )
    return SemanticProbeValidationResult(ok=len(violations) == 0, violations=violations)
