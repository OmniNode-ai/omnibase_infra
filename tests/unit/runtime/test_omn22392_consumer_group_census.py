# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22392: reconcile the recorded census and pin boot-stable instance IDs.

The dominant CI-broker producer is the per-call lab work client in omnimarket.
The census records zero consumers, independently of partition assignments.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TypedDict, cast

import pytest
import yaml

from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.utils import apply_instance_discriminator, compute_consumer_group_id

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CENSUS_PATH = (
    _REPO_ROOT / "tests/fixtures/omn22392/consumer_group_census_2026-11-03.json"
)


class _BusCensus(TypedDict):
    groups_total: int
    zero_consumer_total: int
    zero_consumer_by_template: dict[str, int]
    instance_ids_in_names: dict[str, int]


class _Census(TypedDict):
    buses: dict[str, _BusCensus]


@pytest.fixture(scope="module")
def census() -> dict[str, _BusCensus]:
    """Load the scrubbed, offline census relative to this test file."""
    document = cast("_Census", json.loads(_CENSUS_PATH.read_text(encoding="utf-8")))
    return document["buses"]


def _per_call(template: str) -> bool:
    """Detect ephemeral IDs in the group name, excluding the topic scope."""
    group_name = template.partition(".__t.")[0]
    return any(
        placeholder in group_name
        for placeholder in ("{hex8}", "{hex12}", "{hex32}", "{uuid}")
    )


def _environment_instance_ids(environment: object) -> Iterator[str]:
    """Read instance declarations from either compose environment syntax."""
    if isinstance(environment, Mapping):
        if "KAFKA_INSTANCE_ID" in environment:
            value = environment["KAFKA_INSTANCE_ID"]
            assert isinstance(value, str), f"Non-string KAFKA_INSTANCE_ID: {value!r}"
            yield value
    elif isinstance(environment, list):
        for entry in environment:
            if isinstance(entry, str):
                key, _, value = entry.partition("=")
                if key == "KAFKA_INSTANCE_ID":
                    yield value


def _declared_instance_ids(path: Path) -> Iterator[str]:
    """Read compose service environments and catalog environment mappings."""
    source = path.read_text(encoding="utf-8")
    # Compose merge tags are irrelevant when inspecting declarations in one file.
    # Strip those tags so PyYAML's safe loader can read the underlying values.
    source = re.sub(r"(?m)^(\s*[\w-]+:[ \t]*)!(?:override|reset)\b", r"\1", source)
    document: object = yaml.safe_load(source)
    if not isinstance(document, Mapping):
        return
    services = document.get("services")
    if isinstance(services, Mapping):
        for service in services.values():
            if isinstance(service, Mapping):
                yield from _environment_instance_ids(service.get("environment"))
    for key in ("hardcoded_env", "operational_defaults"):
        yield from _environment_instance_ids(document.get(key))


def test_census_counts_reconcile(census: dict[str, _BusCensus]) -> None:
    for bus, counts in census.items():
        assert (
            sum(counts["zero_consumer_by_template"].values())
            == counts["zero_consumer_total"]
        ), bus
        assert counts["zero_consumer_total"] <= counts["groups_total"], bus


def test_ci_zero_consumer_groups_are_per_call_client_groups(
    census: dict[str, _BusCensus],
) -> None:
    counts = census["ci"]
    per_call = {
        template: count
        for template, count in counts["zero_consumer_by_template"].items()
        if _per_call(template)
    }
    assert sum(per_call.values()) > 0.99 * counts["zero_consumer_total"]
    for template in per_call:
        assert template.startswith("local.omnimarket.lab_work_client.consume.v1."), (
            template
        )


def test_no_zero_consumer_group_carries_an_instance_discriminator(
    census: dict[str, _BusCensus],
) -> None:
    for bus, counts in census.items():
        for template in counts["zero_consumer_by_template"]:
            assert ".__i." not in template, (bus, template)
    assert sum(census["dev"]["instance_ids_in_names"].values()) > 100


def test_dev_instance_ids_are_declared_literals(census: dict[str, _BusCensus]) -> None:
    declared = set(
        _declared_instance_ids(_REPO_ROOT / "docker/docker-compose.infra.yml")
    )
    assert declared
    assert set(census["dev"]["instance_ids_in_names"]) <= declared


def test_every_declared_instance_id_is_a_boot_stable_literal() -> None:
    paths = sorted(_REPO_ROOT.glob("docker/docker-compose*.yml")) + sorted(
        _REPO_ROOT.glob("docker/catalog/services/*.yaml")
    )
    checked = 0
    for path in paths:
        for instance_id in _declared_instance_ids(path):
            assert re.fullmatch(r"[a-z][a-z0-9-]*", instance_id), (path, instance_id)
            checked += 1
    assert checked >= 10


def test_instance_discriminator_is_deterministic_across_boots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = ModelNodeIdentity(
        env="local", service="runtime_config", node_name="node_x", version="v1"
    )
    monkeypatch.setenv("HOSTNAME", "container-first-boot")
    first = apply_instance_discriminator(
        compute_consumer_group_id(identity), "runtime-main"
    )
    monkeypatch.setenv("HOSTNAME", "container-second-boot")
    second = apply_instance_discriminator(
        compute_consumer_group_id(identity), "runtime-main"
    )
    assert first == second
    assert first.endswith(".__i.runtime-main")


def test_per_call_detector_flags_the_producer_and_passes_infra_names() -> None:
    assert _per_call(
        "local.omnimarket.lab_work_client.consume.v1.{hex12}.t.__t.{topic}"
    )
    assert not _per_call(
        "local.runtime_config.node_x.consume.v1.__i.runtime-main.__t.{topic}"
    )
