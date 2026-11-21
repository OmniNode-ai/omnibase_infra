# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``ONEX_RUNTIME_NODE_EXCLUSION`` is carried by compose and defaulted nowhere (OMN-26232).

A lane's overlay names the node it turns off. The compose passthrough anchor must
carry the key to every runtime service, and no compose file, catalog entry, lane
file, prod file or policy default may give it a non-empty value: an unset key must
leave the container's own env unset, so no node is off by default and nothing is
off on prod or stability.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = pytest.mark.unit

KEY = "ONEX_RUNTIME_NODE_EXCLUSION"
ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker"
INFRA = DOCKER / "docker-compose.infra.yml"
COMPOSE_FILES = sorted(DOCKER.glob("docker-compose*.yml"))
CATALOG_FILES = sorted((DOCKER / "catalog").rglob("*.yaml"))
PROD_AND_STABILITY = (
    DOCKER / "docker-compose.prod.yml",
    *sorted(DOCKER.glob("docker-compose.stab*.yml")),
)
# The key assigned a value in a .env, .list, .sh or compose interpolation default.
_ASSIGNED = re.compile(rf"{KEY}[ \t]*[:=][ \t]*[^\s#]|\${{{KEY}:?-[^}}]")


def _values(node: Any) -> Iterator[Any]:
    """Every value the key is given anywhere in a parsed document."""
    if isinstance(node, dict):
        for name, value in node.items():
            if name == KEY:
                yield value
            else:
                yield from _values(value)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str) and item.partition("=")[0] == KEY:
                yield item.partition("=")[2]
            else:
                yield from _values(item)


def _non_empty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def test_the_scan_covers_the_compose_and_catalog_files_it_claims_to() -> None:
    names = {path.name for path in COMPOSE_FILES}
    assert {
        "docker-compose.infra.yml",
        "docker-compose.dev-lane.yml",
        "docker-compose.prod.yml",
    } <= names
    assert len(PROD_AND_STABILITY) >= 2
    assert all(path.is_file() for path in PROD_AND_STABILITY)
    assert any(path.name == "omninode-runtime.yaml" for path in CATALOG_FILES)


def test_the_scan_sees_a_planted_value() -> None:
    # Positive control: the walker finds the key in every shape it can take.
    planted = yaml.safe_load(
        f"services: {{a: {{environment: {{{KEY}: x}}}}, b: {{environment: ['{KEY}=y']}}}}"
    )
    assert [v for v in _values(planted) if _non_empty(v)] == ["x", "y"]
    assert _ASSIGNED.search(f"export {KEY}=x")
    assert _ASSIGNED.search(f"{KEY}: ${{{KEY}:-x}}")
    assert not _ASSIGNED.search(f"  {KEY}:\n  OTHER: value\n")


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: p.name)
def test_no_compose_file_gives_the_key_a_non_empty_value(path: Path) -> None:
    given = [v for v in _values(load_compose(path)) if _non_empty(v)]
    assert not given, f"{path.name} sets {KEY}: {given}"


@pytest.mark.parametrize("path", CATALOG_FILES, ids=lambda p: p.name)
def test_no_catalog_entry_gives_the_key_a_non_empty_value(path: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    given = [v for v in _values(document) if _non_empty(v)]
    assert not given, f"{path.name} sets {KEY}: {given}"


def test_no_docker_file_assigns_the_key_anywhere() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted(DOCKER.rglob("*"))
        if path.is_file()
        and path.suffix
        in {".yml", ".yaml", ".env", ".list", ".sh", ".txt", ".manifest"}
        and _ASSIGNED.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert not offenders, offenders


@pytest.mark.parametrize("path", PROD_AND_STABILITY, ids=lambda p: p.name)
def test_prod_and_stability_files_hold_no_value_for_the_key(path: Path) -> None:
    document = load_compose(path)
    assert document.get("services"), "positive control: the file parsed with services"
    assert not [v for v in _values(document) if _non_empty(v)]
    assert KEY not in path.read_text(encoding="utf-8"), f"{path.name} mentions {KEY}"


def test_the_passthrough_anchor_carries_the_key_value_less() -> None:
    passthrough = load_compose(INFRA)["x-runtime-passthrough-env"]
    assert KEY in passthrough
    assert passthrough[KEY] is None


def test_every_service_that_merges_the_anchor_carries_the_key() -> None:
    services = load_compose(INFRA)["services"]
    anchor_keys = set(load_compose(INFRA)["x-runtime-passthrough-env"])
    merging = {
        name: service["environment"]
        for name, service in services.items()
        if isinstance(service, dict)
        and isinstance(service.get("environment"), dict)
        and anchor_keys <= set(service["environment"])
    }
    assert {
        "omninode-runtime",
        "runtime-effects",
        "runtime-worker",
        "tenant-projection-writer",
        "projection-claude-hook-events-writer",
    } <= set(merging)
    for name, environment in merging.items():
        assert KEY in environment, name
        assert environment[KEY] is None, name
