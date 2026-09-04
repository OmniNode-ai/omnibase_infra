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


@pytest.mark.unit
def test_the_list_form_renders_as_CMD_argv_in_generated_compose() -> None:
    """The manifest change's central mechanical claim, exercised not asserted.

    The six manifests declare `test` as a YAML list on the stated ground that it
    "renders as [CMD, ...] and needs no shell". Nothing else in this change runs
    the generator, so if it stringified the list or emitted CMD-SHELL, six
    services would ship a healthcheck that fails at container start and every
    other test here would still pass -- the validator reads manifests, not
    rendered compose.

    I verified this by hand while writing the fix and did not commit the check.
    That gap is the finding (fp=19db1f9e2463) and this is it closed.
    """
    from omnibase_infra.docker.catalog.generator import generate_compose

    resolved = CatalogResolver(catalog_dir=CATALOG_DIR).resolve(
        bundles=["runtime-core"]
    )
    compose = generate_compose(resolved)
    services = compose["services"]
    assert isinstance(services, dict)

    rendered = {
        name: svc["healthcheck"]["test"]
        for name, svc in services.items()
        if isinstance(svc, dict)
        and isinstance(svc.get("healthcheck"), dict)
        and _SEMANTIC_PROBE_MARKER_IN(svc["healthcheck"]["test"])
    }
    assert rendered, "runtime-core must render at least one semantic probe"

    for name, test in rendered.items():
        assert isinstance(test, list), f"{name}: healthcheck.test must be a list"
        assert test[0] == "CMD", (
            f"{name}: rendered as {test[0]!r}, not 'CMD'. CMD-SHELL requires "
            f"/bin/sh in the image and changes argv handling."
        )
        assert test[1:] == _SEMANTIC_PROBE, f"{name}: argv is {test[1:]!r}"


def _SEMANTIC_PROBE_MARKER_IN(test: object) -> bool:
    """True if a rendered `test` value names the semantic probe, either form."""
    flat = test if isinstance(test, str) else " ".join(map(str, test or []))
    return "onex-container-healthcheck" in flat


@pytest.mark.unit
def test_the_declared_policy_is_enforced_not_merely_prescribed() -> None:
    """The violation message asks for an explicit policy; the gate must check it.

    A probe invoked with no `--degraded-policy` is not fail-open -- the script
    defaults to `fail`. But whether a DEGRADED runtime reads unhealthy is a
    per-lane decision with measured consequences (dev PASSes today while
    lakshman and stability-test both FAIL on runtime_degraded), so a manifest
    must declare it rather than inherit whichever default the script carries.
    """
    no_policy = _manifest(
        name="s",
        layer=EnumInfraLayer.RUNTIME,
        test=["python", "/usr/local/bin/onex-container-healthcheck"],
    )
    result = validate_runtime_semantic_probe({"s": no_policy})
    assert not result.ok
    assert result.violations[0].reason == "policy_missing"

    bogus = _manifest(
        name="s",
        layer=EnumInfraLayer.RUNTIME,
        test=[
            "python",
            "/usr/local/bin/onex-container-healthcheck",
            "--degraded-policy",
            "sometimes",
        ],
    )
    assert not validate_runtime_semantic_probe({"s": bogus}).ok

    for policy in ("fail", "warn"):
        good = _manifest(
            name="s",
            layer=EnumInfraLayer.RUNTIME,
            test=[
                "python",
                "/usr/local/bin/onex-container-healthcheck",
                "--degraded-policy",
                policy,
            ],
        )
        assert validate_runtime_semantic_probe({"s": good}).ok, policy


@pytest.mark.unit
def test_an_incidental_port_digit_run_does_not_pull_a_service_into_scope() -> None:
    """`"8085" in command` also matches 18085, 80853 and HEALTH_PORT=8085.

    A runtime-layer service whose probe merely contains those digits never
    claimed runtime health, and pulling it into the ratchet would fail it for a
    coincidence (finding fp=d39830ae35e8).
    """
    for incidental in (
        "curl -sf http://localhost:18085/health",
        "curl -sf http://localhost:80853/health",
        "sh -c 'HEALTH_PORT=8085x curl -sf http://localhost:9000/health'",
    ):
        svc = _manifest(name="s", layer=EnumInfraLayer.RUNTIME, test=incidental)
        assert not probes_runtime_health(svc), incidental

    real = _manifest(
        name="s",
        layer=EnumInfraLayer.RUNTIME,
        test="curl -sf http://localhost:8085/health",
    )
    assert probes_runtime_health(real), "a real :8085 address must stay in scope"
