# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No member of one force-recreate batch may block on another member's health (OMN-18843).

The defect this ratchets, measured twice on the .201 dev lane:

``deploy-runtime.sh`` ``restart_services()`` force-recreates the WHOLE lane
runtime set in a single ``docker compose up -d --no-deps --force-recreate
<service>...`` call. ``--no-deps`` skips STARTING dependencies that were not
named, but it does not suppress a condition between two services that ARE both
named -- the script's own comment above that call says so. So a
``depends_on: condition: service_healthy`` edge pointing from one batch member
at another is not an ordering hint during a redeploy: it is a hold. Compose
creates the dependent container and leaves it in ``State=created``,
``StartedAt=0001-01-01T00:00:00Z``, until the dependency's healthcheck passes.

On 2026-09-21 that held ``omninode-runtime-effects``,
``omnibase-infra-runtime-worker-1`` and ``omninode-contract-resolver`` from
18:47:59Z to 18:53:47Z behind ``omninode-runtime``, which answered ``/health``
503 for the whole window while it auto-wired 507 contracts under a 1800 s
``start_period`` -- so compose waited rather than failing. The delegate-skill
command consumer runs in the effects container, so its consumer group was Empty
for five minutes and every ``onex delegate`` refused pre-publish. The first
occurrence, on 2026-09-19, measured 7 min 52 s.

The invariant is therefore structural and has nothing to do with which service
is slow today: if two services are recreated in the same batch, neither may
gate on the other's health, because the batch recreates the dependency at the
same moment it needs it to be healthy. An edge that must survive is expressed
as ``service_started``, which still orders the pair and still inherits the
dependency's own preconditions transitively.

Static YAML + shell parse only. No docker daemon, so this runs in unit CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
BASE_FILE = DOCKER_DIR / "docker-compose.infra.yml"
DEV_LANE_FILE = DOCKER_DIR / "docker-compose.dev-lane.yml"
CATALOG_SERVICES_DIR = DOCKER_DIR / "catalog" / "services"

# The arrays in deploy-runtime.sh whose union is recreated in ONE compose call
# on the dev lane. Named rather than globbed so a new array is a deliberate
# addition here and not a silent widening of what this test covers.
BATCH_ARRAYS = (
    "RUNTIME_SERVICES",
    "DEV_LANE_ONLY_RUNTIME_SERVICES",
    "DEV_LANE_EXTRA_BROKER_CLIENTS",
)

BLOCKING_CONDITION = "service_healthy"
RUNTIME_SERVICE = "omninode-runtime"
GENERATED_COMPOSE_NAME = "docker-compose.generated.yml"


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Passthrough for compose-only YAML tags (``!override``, ``!reset``, ``!!merge``)."""
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """Test-local loader with Docker Compose tag support."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)
_ComposeLoader.add_constructor("!reset", _construct_compose_value)
_ComposeLoader.add_multi_constructor("", _construct_compose_value)


def _load_services(path: Path) -> dict[str, dict]:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader)  # noqa: S506
    assert isinstance(data, dict), f"{path.name} did not parse to a mapping"
    services = data.get("services") or {}
    assert isinstance(services, dict)
    return services


def _merged_dev_lane_services() -> dict[str, dict]:
    """Base compose with the dev-lane overlay merged PER FIELD.

    Not ``dict.update``: compose merges an overlay stanza into the base
    service, it does not replace it. The dev-lane stanzas for the runtime
    containers carry only environment, healthcheck and labels, so a whole-value
    replace silently drops their ``depends_on`` and this file would then ratchet
    on an empty graph and pass on everything.
    """
    merged = {name: dict(cfg) for name, cfg in _load_services(BASE_FILE).items()}
    for name, overlay in _load_services(DEV_LANE_FILE).items():
        if not isinstance(overlay, dict):
            continue
        merged.setdefault(name, {}).update(overlay)
    return merged


def _batch_services() -> set[str]:
    """Union of the dev-lane restart arrays, parsed out of deploy-runtime.sh.

    Read from the script rather than restated here so the test cannot drift
    away from what the deploy agent actually names in its compose call.
    """
    source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    names: set[str] = set()
    for array in BATCH_ARRAYS:
        match = re.search(
            rf"^readonly\s+{re.escape(array)}=\((?P<body>.*?)^\)",
            source,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None, f"{array} not found in {DEPLOY_SCRIPT.name}"
        for line in match.group("body").splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                names.add(entry)
    return names


def _depends_on(service_config: dict) -> dict[str, str]:
    """Return ``{dependency: condition}`` for one service.

    Compose accepts both the short list form (implicitly ``service_started``)
    and the long mapping form. Both are normalised here so a future edge added
    in the short form cannot slip past the assertion below.
    """
    raw = service_config.get("depends_on") or {}
    if isinstance(raw, list):
        return {str(dep): "service_started" for dep in raw}
    assert isinstance(raw, dict)
    return {
        str(dep): str((cfg or {}).get("condition", "service_started"))
        for dep, cfg in raw.items()
    }


@pytest.mark.unit
def test_no_batch_member_gates_on_another_batch_members_health() -> None:
    """The OMN-18843 ratchet: no intra-batch ``service_healthy`` edge.

    Red on dev at the time of writing with exactly three findings
    (runtime-effects, runtime-worker and omninode-contract-resolver, all
    pointing at omninode-runtime, whose ``start_period`` is 1800 s).
    """
    batch = _batch_services()
    services = _merged_dev_lane_services()

    findings: list[str] = []
    for name in sorted(batch):
        config = services.get(name)
        if not isinstance(config, dict):
            # Declared in another lane's overlay; not part of this batch's graph.
            continue
        for dependency, condition in _depends_on(config).items():
            if dependency in batch and condition == BLOCKING_CONDITION:
                healthcheck = (services.get(dependency) or {}).get("healthcheck") or {}
                findings.append(
                    f"{name} -> {dependency} ({condition}, "
                    f"start_period={healthcheck.get('start_period')})"
                )

    assert not findings, (
        "A force-recreate batch member gates on another batch member's health. "
        "Compose holds the dependent in State=created for the whole healthcheck "
        "window, which is a delegation outage on every redeploy (OMN-18843). "
        "Express the edge as service_started instead -- it still orders the pair "
        "and still inherits the dependency's own preconditions. Findings: "
        + "; ".join(findings)
    )


@pytest.mark.unit
def test_deploy_script_still_recreates_the_batch_in_one_call() -> None:
    """Negative control for the premise, not a second copy of the assertion.

    The ratchet above is only meaningful while the deploy agent recreates these
    services together in a single ``up``. If that ever becomes one call per
    service, an intra-batch health edge stops being a hold and this file should
    be re-reasoned rather than silently kept passing.
    """
    source = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"restart_services\(\)\s*\{.*?^\}",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "restart_services() not found in deploy-runtime.sh"
    body = match.group(0)
    assert "up -d --no-deps --force-recreate" in body, (
        "restart_services() no longer force-recreates with --no-deps; the "
        "premise of the OMN-18843 ratchet has changed."
    )
    assert '"${lane_services[@]}"' in body, (
        "restart_services() no longer passes the whole lane service array to "
        "one compose call; re-reason the OMN-18843 ratchet."
    )


@pytest.mark.unit
def test_no_compose_file_anywhere_gates_a_consumer_on_the_runtime_health() -> None:
    """The fleet-wide form, and the reason this file has three assertions.

    The dev-lane ratchet above reads only the dev restart batch, and that is a
    real blind spot: it passed on a tree where three OTHER lane composes still
    carried the identical edge. ``docker-compose.dogfood.yml`` is a complete
    STANDALONE definition that deliberately does not include the base infra
    file, and ``judge`` and ``lakshman`` each declare their own; the catalog
    additionally declared it on two more services. Five sites, invisible to a
    dev-batch ratchet, each with a main runtime carrying a 30-minute
    start_period. Found by taking the proof to the dogfood surface, not by CI.

    The invariant is a property of the dependency, not of a lane: a container
    that consumes from the bus must not be held in ``State=created`` behind the
    main runtime's healthcheck, because the batch that recreates it recreates
    the main runtime at the same moment. Expressed as ``service_started`` the
    edge still orders the pair and still inherits the main runtime's own
    preconditions transitively.
    """
    findings: list[str] = []
    for path in sorted(DOCKER_DIR.glob("docker-compose*.yml")):
        if path.name == GENERATED_COMPOSE_NAME:
            # Generated, gitignored, and rebuilt from the catalog the assertion
            # below covers. Reading it here would grade a build artifact whose
            # content depends on which bundle was generated last.
            continue
        for name, config in _load_services(path).items():
            condition = _depends_on(config).get(RUNTIME_SERVICE)
            if condition == BLOCKING_CONDITION:
                findings.append(f"{path.name}: {name} -> {RUNTIME_SERVICE}")

    for path in sorted(CATALOG_SERVICES_DIR.glob("*.yaml")):
        catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
        for entry in (catalog or {}).get("depends_on") or []:
            if (
                entry.get("service") == RUNTIME_SERVICE
                and entry.get("condition") == BLOCKING_CONDITION
            ):
                findings.append(f"catalog/{path.name}: -> {RUNTIME_SERVICE}")

    assert not findings, (
        "A service gates on the main runtime's health. Compose holds it in "
        "State=created for the whole healthcheck window, and every lane's main "
        "runtime carries a 1800 s start_period, so a force-recreate naming both "
        "strands this container's consumer groups for minutes (OMN-18843). Use "
        "service_started. Findings: " + "; ".join(findings)
    )


@pytest.mark.unit
def test_catalog_agrees_with_the_hand_written_compose() -> None:
    """The catalog generates ``docker-compose.generated.yml``; it must not drift.

    The dependency edge is declared twice -- once in the hand-written
    ``docker-compose.infra.yml`` the dev lane deploys from, and once in
    ``docker/catalog/services/*.yaml``, which generates the compose file the
    lane manifest points at. A fix applied to only one of them leaves the other
    lane reintroducing the outage, and nothing else compares them.
    """
    base_services = _load_services(BASE_FILE)
    catalog_by_service = {
        "runtime-effects": CATALOG_SERVICES_DIR / "runtime-effects.yaml",
        "runtime-worker": CATALOG_SERVICES_DIR / "runtime-worker.yaml",
        "omninode-contract-resolver": CATALOG_SERVICES_DIR / "contract-resolver.yaml",
    }

    mismatches: list[str] = []
    for service, catalog_path in catalog_by_service.items():
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        catalog_conditions = {
            str(entry["service"]): str(entry.get("condition", "service_started"))
            for entry in (catalog.get("depends_on") or [])
        }
        compose_conditions = _depends_on(base_services[service])
        if catalog_conditions != compose_conditions:
            mismatches.append(
                f"{service}: catalog={catalog_conditions} compose={compose_conditions}"
            )

    assert not mismatches, (
        "Catalog and hand-written compose disagree on a dependency condition, "
        "so the generated compose file carries a different startup graph than "
        "the dev lane does (OMN-18843). Mismatches: " + "; ".join(mismatches)
    )
