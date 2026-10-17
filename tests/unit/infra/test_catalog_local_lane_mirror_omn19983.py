# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19983: the laptop compose stack mirrors its delegation topics to a lab lane.

The ``local-lane-mirror`` bundle is the laptop profile plus one mirror-only
forwarder (OMN-20867). Which lab lane, its address and the principal that writes
it come from two operator files outside this repo; the templates shipped here
must load through the production loader, or an operator copying them would meet
a refusal the tests never saw.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.runtime.gateway_forwarder import (
    load_gateway_forwarder_runtime_config,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_CATALOG_DIR = str(_REPO / "docker" / "catalog")
_OVERLAYS = _REPO / "docker" / "lane-overlays"
_MIRROR_OVERLAY = _OVERLAYS / "local.lane-mirror.example.yaml"
_CREDENTIAL_MAP = _OVERLAYS / "local.lane-credentials.example.yaml"
_CONTRACT = (
    _REPO
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)
_SERVICE = "lane-mirror-forwarder"
_BUNDLE = "local-lane-mirror"
_OPERATOR_FILE_VARS = {
    "ONEX_LOCAL_LANE_MIRROR_OVERLAY",
    "ONEX_LOCAL_LANE_CREDENTIAL_MAP",
}


def _services(bundle: str) -> dict[str, dict[str, object]]:
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([bundle])
    )
    services = compose["services"]
    assert isinstance(services, dict)
    return services


def _delegation_topics() -> tuple[str, ...]:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    topics = contract["config"]["gateway_forwarder"]["lane_mirror"]["topic_sets"][
        "delegation"
    ]
    assert topics
    return tuple(topics)


def test_the_mirror_bundle_adds_one_forwarder_to_the_laptop_stack() -> None:
    laptop = _services("local")
    mirror = _services(_BUNDLE)
    assert _SERVICE not in laptop
    assert sorted(set(mirror) - set(laptop)) == [_SERVICE]
    assert set(laptop) <= set(mirror)


def test_the_mirror_bundle_renders_under_the_laptop_project() -> None:
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([_BUNDLE])
    )
    assert compose["name"] == "omnibase-infra-local"


def test_the_mirror_bundle_asks_the_operator_for_exactly_the_two_files() -> None:
    laptop = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve(["local"]).required_env
    mirror = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([_BUNDLE]).required_env
    assert mirror - laptop == _OPERATOR_FILE_VARS
    assert laptop.isdisjoint(_OPERATOR_FILE_VARS)


def test_the_forwarder_runs_mirror_only_with_no_cloud_input() -> None:
    svc = _services(_BUNDLE)[_SERVICE]
    command = svc["command"]
    assert isinstance(command, list)
    assert command[0] == "onex-gateway-forwarder"
    flags = [part for part in command if str(part).startswith("--")]
    # A mirror-only deployment takes no broker-ref map and no AWS material.
    assert "--broker-ref-map" not in flags
    assert "--lane-credential-map" in flags
    assert "--ready-file" in flags
    assert "ports" not in svc


def test_the_forwarder_mounts_the_operator_files_read_only() -> None:
    svc = _services(_BUNDLE)[_SERVICE]
    volumes = svc["volumes"]
    assert isinstance(volumes, list)
    mounts = {str(v).split(":?")[0].removeprefix("${"): str(v) for v in volumes}
    for var, target in (
        ("ONEX_LOCAL_LANE_MIRROR_OVERLAY", "/app/config/lane-mirror.yaml:ro"),
        ("ONEX_LOCAL_LANE_CREDENTIAL_MAP", "/run/gateway/lane-credentials.yaml:ro"),
    ):
        assert mounts[var].endswith(f"}}:{target}")
    assert [v for v in volumes if "/dev/tpm" in str(v) or "aws" in str(v)] == []


def test_the_forwarder_starts_after_the_broker_requires_authentication() -> None:
    svc = _services(_BUNDLE)[_SERVICE]
    assert svc["depends_on"] == {
        "redpanda": {"condition": "service_healthy"},
        "redpanda-local-sasl": {"condition": "service_completed_successfully"},
    }
    assert svc["restart"] == "unless-stopped"
    # The check reads the very file the process writes once its transports start.
    command = svc["command"]
    assert isinstance(command, list)
    ready_file = command[command.index("--ready-file") + 1]
    assert svc["healthcheck"]["test"] == ["CMD", "test", "-f", ready_file]


def _credential_map(tmp_path: Path, *, drop: str | None = None) -> Path:
    entries = yaml.safe_load(_CREDENTIAL_MAP.read_text(encoding="utf-8"))
    if drop is not None:
        del entries[drop]
    path = tmp_path / "credentials.yaml"
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


def test_the_shipped_overlay_template_loads_and_mirrors_only_delegation_topics(
    tmp_path: Path,
) -> None:
    config = load_gateway_forwarder_runtime_config(
        _MIRROR_OVERLAY,
        broker_ref_map_path=None,
        lane_credential_map_path=_credential_map(tmp_path),
    )
    lane_mirror = config.forwarder.lane_mirror
    assert lane_mirror is not None
    assert lane_mirror.source_lane == "dev-local"
    assert lane_mirror.mirror_lanes == ("dev",)
    assert lane_mirror.topics == _delegation_topics()
    assert config.forwarder.cloud_bus is None
    assert config.cloud_bus is None
    assert config.local_bus is None


def test_the_overlay_selects_the_mirrored_topics_by_set_name(tmp_path: Path) -> None:
    """Falsifier: the same overlay naming ``hook_edge`` mirrors no delegation topic."""
    raw = yaml.safe_load(_MIRROR_OVERLAY.read_text(encoding="utf-8"))
    raw["forwarder"]["lane_mirror_binding"]["topic_sets"] = ["hook_edge"]
    overlay = tmp_path / "hook-edge.yaml"
    overlay.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_gateway_forwarder_runtime_config(
        overlay,
        broker_ref_map_path=None,
        lane_credential_map_path=_credential_map(tmp_path),
    )
    lane_mirror = config.forwarder.lane_mirror
    assert lane_mirror is not None
    assert set(lane_mirror.topics).isdisjoint(_delegation_topics())


def test_a_credential_map_without_the_lab_principal_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=re.escape("lane.dev.kafka.scram")):
        load_gateway_forwarder_runtime_config(
            _MIRROR_OVERLAY,
            broker_ref_map_path=None,
            lane_credential_map_path=_credential_map(
                tmp_path, drop="lane.dev.kafka.scram"
            ),
        )


def test_the_credential_template_holds_only_placeholders() -> None:
    entries = yaml.safe_load(_CREDENTIAL_MAP.read_text(encoding="utf-8"))
    assert set(entries) == {"lane.dev-local.kafka.scram", "lane.dev.kafka.scram"}
    for ref, entry in entries.items():
        assert entry["password"].startswith("REPLACE_WITH_"), ref


def test_the_env_template_leaves_the_mirror_files_commented_out() -> None:
    lines = (
        (_REPO / "docker" / "local.env.example")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    for var in _OPERATOR_FILE_VARS:
        assert [ln for ln in lines if ln.startswith(f"{var}=")] == []
        assert [ln for ln in lines if ln.startswith(f"# {var}=")] != []


def test_the_make_target_boots_the_mirror_bundle_from_the_laptop_env_file() -> None:
    makefile = (_REPO / "Makefile").read_text(encoding="utf-8")
    assert '$(ONEX_CLI) up local-lane-mirror --env-file "$(LOCAL_ENV_FILE)"' in makefile
