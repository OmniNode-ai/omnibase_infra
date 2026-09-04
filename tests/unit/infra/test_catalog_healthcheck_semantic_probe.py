# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17883: the catalog's runtime healthchecks must be the semantic probe.

OMN-15217 converted the two hand-written lane composes from ``curl -sf`` to
``onex-container-healthcheck`` and never reached the service catalog, which
CLAUDE.md designates as the sanctioned generated path. The regression survived
from 2026-07 until OMN-17623 surfaced it, because nothing compared the catalog's
declared probe against anything: ``test_catalog_completeness.py`` checks that
every compose entry HAS a manifest, never what its healthcheck says, and the two
tests that do pin the semantic probe are both lane-compose-scoped.

This is the enforcement half (Operating Rule 5). Detection was always possible.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest, HealthCheck
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.docker.catalog.validator_healthcheck_semantic_probe import (
    probes_runtime_health,
    validate_runtime_semantic_probe,
)

CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")

_SHALLOW_PROBE = "curl -sf http://localhost:8085/health"
_SEMANTIC_PROBE = [
    "python",
    "/usr/local/bin/onex-container-healthcheck",
    "--degraded-policy",
    "fail",
]

# The six runtime-family services this ticket converted. Named explicitly rather
# than derived, so a service silently dropping OUT of coverage is a failure here
# too -- the first draft of the validator lost coverage entirely and still
# reported ok, which is the defect class this file exists to close.
_EXPECTED_COVERED = {
    "dlq-replay-consumer",
    "fault-inject-fixture",
    "omninode-runtime",
    "runtime-canary",
    "runtime-effects",
    "runtime-worker",
}


def _manifest(
    *,
    name: str,
    layer: EnumInfraLayer,
    test: str | list[str],
    image: str = "runtime:latest",
) -> CatalogManifest:
    """A minimal manifest for unit testing, mirroring the start_period sibling."""
    return CatalogManifest(
        name=name,
        description="test service",
        image=image,
        layer=layer,
        required_env=[],
        hardcoded_env={},
        operational_defaults={},
        ports=None,
        healthcheck=HealthCheck(test=test),
        volumes=[],
        depends_on=[],
    )


def _all_manifests() -> dict[str, CatalogManifest]:
    bundles = [
        name
        for name in yaml.safe_load(Path(CATALOG_DIR, "bundles.yaml").read_text())
        if isinstance(name, str)
    ]
    return CatalogResolver(catalog_dir=CATALOG_DIR).resolve(bundles=bundles).manifests


@pytest.mark.unit
def test_real_catalog_declares_the_semantic_probe() -> None:
    """Every runtime health probe in the shipped catalog is the semantic one."""
    result = validate_runtime_semantic_probe(_all_manifests())
    assert result.ok, result.report()


@pytest.mark.unit
def test_coverage_is_exactly_the_runtime_family() -> None:
    """Guard the guard: assert WHICH services are checked, not just that it passes.

    A validator that silently stops matching reports ok forever. That is not
    hypothetical here -- the first version of this validator keyed coverage on
    the health PORT appearing in the healthcheck command, which the semantic
    probe's argv does not contain, so coverage fell to zero the moment the fix
    landed and the gate went green having examined nothing.
    """
    covered = {n for n, m in _all_manifests().items() if probes_runtime_health(m)}
    assert covered == _EXPECTED_COVERED, (
        f"runtime-health probe coverage changed: "
        f"missing={sorted(_EXPECTED_COVERED - covered)} "
        f"unexpected={sorted(covered - _EXPECTED_COVERED)}. A service leaving "
        f"coverage means this gate stopped checking it, which is silent."
    )


@pytest.mark.unit
def test_a_revert_to_the_shallow_probe_is_caught() -> None:
    """The negative case, exercised rather than described."""
    manifests = _all_manifests()
    original = manifests["omninode-runtime"]
    assert original.healthcheck is not None
    manifests["omninode-runtime"] = replace(
        original, healthcheck=replace(original.healthcheck, test=_SHALLOW_PROBE)
    )

    result = validate_runtime_semantic_probe(manifests)
    assert not result.ok
    assert len(result.violations) == 1
    assert result.violations[0].service == "omninode-runtime"
    assert "onex-container-healthcheck" in result.report()


@pytest.mark.unit
def test_a_non_runtime_layer_service_is_not_subject_to_the_rule() -> None:
    """Scope: only the runtime layer serves ServiceHealth on the health port.

    Eighteen catalog services sit on the runtime layer and all eighteen share
    `runtime:latest`, but they serve different apps on different ports
    (intelligence-api on 8053, projection-api on 3002). Layer alone would
    over-reach; the health-endpoint check is what narrows it.
    """
    infra_service = _manifest(
        name="postgres",
        image="postgres:16",
        layer=EnumInfraLayer.INFRASTRUCTURE,
        test=_SHALLOW_PROBE,
    )
    assert not probes_runtime_health(infra_service)
    assert validate_runtime_semantic_probe({"postgres": infra_service}).ok


@pytest.mark.unit
def test_a_runtime_service_probing_another_port_is_out_of_scope() -> None:
    """A runtime-image service serving a different app is not claiming runtime health."""
    other_app = _manifest(
        name="projection-api",
        layer=EnumInfraLayer.RUNTIME,
        test="curl -sf http://localhost:3002/health",
    )
    assert not probes_runtime_health(other_app)
    assert validate_runtime_semantic_probe({"projection-api": other_app}).ok


@pytest.mark.unit
def test_both_probe_forms_are_recognised_as_runtime_health_claims() -> None:
    """Coverage must survive the very conversion this ticket performs.

    The shallow form names the port; the semantic form names none, because
    :8085 is the probe's own default. A discriminator that accepts only one of
    them loses coverage at exactly the moment the fix lands.
    """
    shallow = _manifest(name="s", layer=EnumInfraLayer.RUNTIME, test=_SHALLOW_PROBE)
    semantic = _manifest(
        name="s", layer=EnumInfraLayer.RUNTIME, test=list(_SEMANTIC_PROBE)
    )
    assert probes_runtime_health(shallow), "shallow form must stay in scope"
    assert probes_runtime_health(semantic), "semantic form must stay in scope"
    assert not validate_runtime_semantic_probe({"s": shallow}).ok
    assert validate_runtime_semantic_probe({"s": semantic}).ok
