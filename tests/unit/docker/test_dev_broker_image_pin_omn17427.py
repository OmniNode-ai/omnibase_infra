# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev broker's Redpanda image is declared once, the same in every place (OMN-17427).

On 2026-10-20 the lab dev broker (``omnibase-infra-redpanda``) crash-looped
from 15:19:18Z on a v24.2.7 startup assert (``shard_balancer.cc:102``
``topic_table unexpectedly changed``) after a pre-PR slot teardown deleted about
1,845 topics in chunks of 200. On a copy of its data volume v24.2.7 and v24.2.27
asserted the same way and v24.3.18 started healthy, so the operator ruled the
dev broker onto v24.3.18 (ledger RULING 2026-10-20T17:35:21Z).

On 2026-10-26 the pin moved to v25.1.12, the first hop of
``docker/redpanda/dev_broker_upgrade_path.yaml`` (OMN-21279), after v25.1.12,
v25.2.14 and v25.3.18 started healthy in turn on a copy of the dev data volume,
under the same ruling's rule that the dev broker moves to a version proven on
a data copy.

On 2026-10-28 the live v25.1.12 broker (cluster_version 15, read back on h201)
crashed at ``kafka::group_manager::list_groups`` +0x1cb, the same frame as the
v24.3.18 crashes: v25.1.12 is a hop, not the fix, which first ships in v25.3.16.
The pin moved to v25.2.14, the second hop, and v25.3.18 follows once v25.2.14 is
live and read back (OMN-20769).

The image is spelled in several files: the catalog services, the base compose
file (``docker-compose.infra.yml``: the broker and its partition-cap one-shot)
and the dev-lane overlay (``docker-compose.dev-lane.yml``: the SCRAM and SASL
one-shots). The deploy agent renders the two compose files, so a pin moved in
only one of them is reverted by the next deploy. These tests hold every place
to one value.

They also hold the move to the dev lane. Other lanes' compose files layer
over ``docker-compose.infra.yml`` or over both dev files, and their
``redpanda`` services used to inherit the image. A broker upgrade is one-way
on its data volume, so every other compose file declares its own image for
each broker service it runs, and the dev pin never moves another lane's
broker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]

DEV_BROKER_IMAGE = "redpandadata/redpanda:v25.2.14"

CATALOG_DIR = DOCKER_DIR / "catalog" / "services"
DEV_COMPOSE_FILES = ("docker-compose.infra.yml", "docker-compose.dev-lane.yml")
CATALOG_SERVICES = ("redpanda", "redpanda-partition-cap", "redpanda-local-sasl")


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that keeps the value under compose's ``!override``/``!reset`` tags."""


def _keep_value(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    raise AssertionError(f"unhandled YAML node kind: {type(node).__name__}")


_ComposeLoader.add_constructor("!override", _keep_value)
_ComposeLoader.add_constructor("!reset", _keep_value)


def _load(path: Path) -> dict[str, Any]:
    loaded = _ComposeLoader(path.read_text()).get_single_data()
    assert isinstance(loaded, dict), f"{path} did not parse to a mapping"
    return loaded


def _services(name: str) -> dict[str, dict[str, Any]]:
    return {k: v or {} for k, v in (_load(DOCKER_DIR / name)["services"] or {}).items()}


def _redpanda_service_names() -> set[str]:
    """Every service of the dev compose project that is a broker or runs rpk against it."""
    names: set[str] = set()
    for compose in DEV_COMPOSE_FILES:
        for service, spec in _services(compose).items():
            if "redpanda" in service or "redpanda/redpanda" in str(
                spec.get("image", "")
            ):
                names.add(service)
    return names


def _declared_dev_images() -> dict[str, str]:
    """Where the dev broker's image is declared -> the image declared there."""
    declared: dict[str, str] = {}
    for compose in DEV_COMPOSE_FILES:
        for service, spec in _services(compose).items():
            if service in _redpanda_service_names() and "image" in spec:
                declared[f"{compose}:{service}"] = str(spec["image"])
    for name in CATALOG_SERVICES:
        declared[f"catalog/services/{name}.yaml"] = str(
            _load(CATALOG_DIR / f"{name}.yaml")["image"]
        )
    return declared


def test_the_dev_broker_runs_the_declared_image() -> None:
    broker = _services("docker-compose.infra.yml")["redpanda"]
    assert broker["image"] == DEV_BROKER_IMAGE
    assert str(_load(CATALOG_DIR / "redpanda.yaml")["image"]) == DEV_BROKER_IMAGE


def test_the_dev_broker_image_is_the_same_in_every_place() -> None:
    declared = _declared_dev_images()
    # The broker, its partition-cap one-shot, the two SASL one-shots, and the
    # three catalog services: a place that stops declaring the image must fail
    # here rather than silently inherit something else.
    assert len(declared) == 7, sorted(declared)
    differing = {
        where: image for where, image in declared.items() if image != DEV_BROKER_IMAGE
    }
    assert not differing, (
        f"dev broker image differs from {DEV_BROKER_IMAGE}: {differing}"
    )


def test_every_dev_redpanda_service_declares_its_image() -> None:
    """A dev-project rpk or broker service with no image inherits from nowhere it is checked."""
    merged: dict[str, dict[str, Any]] = {}
    for compose in DEV_COMPOSE_FILES:
        for service, spec in _services(compose).items():
            merged.setdefault(service, {}).update(spec)
    missing = sorted(s for s in _redpanda_service_names() if "image" not in merged[s])
    assert not missing, f"dev redpanda services with no image: {missing}"


def _other_compose_files() -> list[Path]:
    return [
        path
        for path in sorted(DOCKER_DIR.glob("docker-compose*.yml"))
        if path.name not in DEV_COMPOSE_FILES
    ]


def test_other_compose_files_are_found() -> None:
    """Positive control: the scan below reads real files, not an empty glob."""
    running_a_broker = [
        path.name
        for path in _other_compose_files()
        if "redpanda" in (_load(path).get("services") or {})
    ]
    assert len(running_a_broker) >= 5, running_a_broker


def test_no_other_compose_file_inherits_the_dev_broker_image() -> None:
    """Moving the dev pin never moves a broker another lane inherits from the dev files.

    A service that shares its name with a dev redpanda service and starts
    (no ``profiles`` that keep it off) must declare its own image or extend
    a template that does.
    """
    dev_names = _redpanda_service_names()
    inherited: list[str] = []
    for path in _other_compose_files():
        services = _load(path).get("services") or {}
        for service in sorted(dev_names & services.keys()):
            spec = services[service] or {}
            # ``extends`` names its own source file (the broker-profile
            # template), never the dev files, so it is not inheritance here.
            if "image" in spec or "extends" in spec or spec.get("profiles"):
                continue
            inherited.append(f"{path.name}:{service}")
    assert not inherited, f"services inheriting the dev broker image: {inherited}"
