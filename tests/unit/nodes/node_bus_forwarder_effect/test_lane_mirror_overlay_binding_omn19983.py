# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19983: resolved deployment overlays select lane-mirror bindings.

The contract declares a closed lane set and named topic sets. Deployments
select source and mirror lanes and the topic sets to forward from those
declarations. Undeclared lanes fail when the config model is constructed;
malformed bindings and unknown topic sets fail during materialization.
The shipped deployment retains stability-test -> dev for hook topics only.
Developer delegation envelopes keep their tenant id, bytes, and headers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderRuntimeConfig,
    ModelGatewayLaneMirrorConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
    NodeLaneMirror,
)
from omnibase_infra.runtime.gateway_forwarder import (
    _materialize_contract_canary_config,
    _materialize_contract_lane_mirror,
    _materialize_contract_mirror_topics,
    _materialize_lane_broker_credentials,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACT_PATH = (
    _REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)
_RESOLVED_CONFIG_PATH = _REPO_ROOT / "docker" / "gateway" / "beta-gateway-canary.yaml"
_DELEGATION_TOPICS = (
    "onex.evt.omnimarket.delegate-skill-completed.v1",
    "onex.evt.omnimarket.delegate-skill-failed.v1",
    "onex.evt.omnibase-infra.delegation-completed.v1",
    "onex.evt.omnibase-infra.delegation-failed.v1",
    "onex.evt.omniintelligence.llm-call-completed.v1",
)


def _contract_block() -> dict[str, Any]:
    contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    return contract["config"]["gateway_forwarder"]["lane_mirror"]


def _binding() -> dict[str, Any]:
    return {
        "source_lane": "dev-local",
        "mirror_lanes": ["dev"],
        "topic_sets": ["delegation"],
    }


def _developer_runtime_raw(
    lane_credential_map: Path, *, mirror_lane: str = "dev"
) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(
        _RESOLVED_CONFIG_PATH.read_text(encoding="utf-8")
    )
    binding = _binding()
    binding["mirror_lanes"] = [mirror_lane]
    raw["forwarder"]["lane_mirror_binding"] = binding
    raw["lane_mirror_buses"] = {"dev": raw["lane_mirror_buses"]["dev"]}
    _materialize_contract_mirror_topics(raw, _CONTRACT_PATH)
    _materialize_contract_canary_config(raw, _CONTRACT_PATH)
    _materialize_contract_lane_mirror(raw, _CONTRACT_PATH)
    raw["cloud_bus"]["bootstrap_servers"] = "cloud-broker.example:9094"
    raw["lane_mirror_source_bus"]["bootstrap_servers"] = "laptop-redpanda.example:9092"
    raw["lane_mirror_buses"]["dev"]["bootstrap_servers"] = "lab-dev.example:9092"
    _materialize_lane_broker_credentials(raw, lane_credential_map)
    return raw


def test_developer_overlay_builds_with_delegation_topics(
    lane_credential_map: Path,
) -> None:
    config = ModelGatewayForwarderRuntimeConfig.model_validate(
        _developer_runtime_raw(lane_credential_map)
    )
    lane_mirror = config.forwarder.lane_mirror
    assert lane_mirror is not None
    assert lane_mirror.source_lane == "dev-local"
    assert lane_mirror.mirror_lanes == ("dev",)
    assert lane_mirror.topics == tuple(_contract_block()["topic_sets"]["delegation"])


def test_undeclared_overlay_mirror_lane_is_refused_at_model_construction(
    lane_credential_map: Path,
) -> None:
    raw = _developer_runtime_raw(lane_credential_map, mirror_lane="lab-prod")
    assert raw["forwarder"]["lane_mirror"]["mirror_lanes"] == ["lab-prod"]
    with pytest.raises(ValidationError, match="declared_lanes") as error:
        ModelGatewayForwarderRuntimeConfig.model_validate(raw)
    assert "lab-prod" in str(error.value)
    assert "dev-local" in str(error.value)


def test_undeclared_source_lane_is_refused_at_model_construction() -> None:
    with pytest.raises(ValidationError, match="declared_lanes") as error:
        ModelGatewayLaneMirrorConfig(
            source_lane="laptop-unknown",
            mirror_lanes=("dev",),
            declared_lanes=("dev-local", "dev"),
            topics=(_DELEGATION_TOPICS[0],),
        )
    assert "laptop-unknown" in str(error.value)
    assert "dev-local" in str(error.value)


def test_repeated_declared_lane_is_refused() -> None:
    with pytest.raises(ValidationError, match="declared_lanes") as error:
        ModelGatewayLaneMirrorConfig(
            source_lane="dev-local",
            mirror_lanes=("dev",),
            declared_lanes=("dev-local", "dev", "dev"),
            topics=(_DELEGATION_TOPICS[0],),
        )
    assert "repeat lanes ['dev']" in str(error.value)
    assert "('dev-local', 'dev', 'dev')" in str(error.value)


def test_unknown_topic_set_is_refused() -> None:
    binding = _binding()
    binding["topic_sets"] = ["unknown-delegation"]
    raw = {
        "forwarder": {
            "lane_mirror_set": "node_bus_forwarder_effect",
            "lane_mirror_binding": binding,
        }
    }
    with pytest.raises(ValueError, match="unknown-delegation") as error:
        _materialize_contract_lane_mirror(raw, _CONTRACT_PATH)
    assert "hook_edge" in str(error.value)
    assert "delegation" in str(error.value)


@pytest.mark.parametrize(
    ("forwarder", "message"),
    [
        (
            {"lane_mirror_binding": _binding()},
            "declares lane_mirror_binding but does not name lane_mirror_set",
        ),
        (
            {"lane_mirror_set": "node_bus_forwarder_effect"},
            "names lane_mirror_set but no lane_mirror_binding",
        ),
        (
            {
                "lane_mirror_set": "node_bus_forwarder_effect",
                "lane_mirror_binding": {**_binding(), "unexpected": True},
            },
            "extra keys.*unexpected",
        ),
        (
            {
                "lane_mirror_set": "node_bus_forwarder_effect",
                "lane_mirror_binding": {"source_lane": "dev-local"},
            },
            "missing keys.*mirror_lanes.*topic_sets",
        ),
    ],
)
def test_incomplete_or_extra_binding_is_refused(
    forwarder: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _materialize_contract_lane_mirror({"forwarder": forwarder}, _CONTRACT_PATH)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source_lane", ["dev-local"]),
        ("mirror_lanes", "dev"),
        ("mirror_lanes", [1]),
        ("topic_sets", "delegation"),
        ("topic_sets", []),
        ("topic_sets", [1]),
    ],
)
def test_binding_requires_string_lane_and_lists_of_strings(
    key: str, value: object
) -> None:
    binding = _binding()
    binding[key] = value
    with pytest.raises(ValueError, match=key):
        _materialize_contract_lane_mirror(
            {
                "forwarder": {
                    "lane_mirror_set": "node_bus_forwarder_effect",
                    "lane_mirror_binding": binding,
                }
            },
            _CONTRACT_PATH,
        )


def test_shipped_deployment_keeps_only_hook_topics(
    lane_mirror_runtime_raw: dict[str, Any],
) -> None:
    config = ModelGatewayForwarderRuntimeConfig.model_validate(lane_mirror_runtime_raw)
    lane_mirror = config.forwarder.lane_mirror
    assert lane_mirror is not None
    assert lane_mirror.source_lane == "stability-test"
    assert lane_mirror.mirror_lanes == ("dev",)
    assert lane_mirror.topics == tuple(_contract_block()["topic_sets"]["hook_edge"])
    assert len(lane_mirror.topics) == 4
    assert set(lane_mirror.topics).isdisjoint(_DELEGATION_TOPICS)


def test_contract_declares_developer_lane_and_exact_delegation_topics() -> None:
    block = _contract_block()
    assert {"dev-local", "dev"} <= set(block["declared_lanes"])
    assert tuple(block["topic_sets"]["delegation"]) == _DELEGATION_TOPICS
    assert set(block["topic_sets"]["delegation"]).isdisjoint(
        block["topic_sets"]["hook_edge"]
    )


def test_topic_sets_preserve_binding_order_and_deduplicate() -> None:
    binding = _binding()
    binding["topic_sets"] = ["delegation", "hook_edge", "delegation"]
    raw: dict[str, Any] = {
        "forwarder": {
            "lane_mirror_set": "node_bus_forwarder_effect",
            "lane_mirror_binding": binding,
        }
    }
    _materialize_contract_lane_mirror(raw, _CONTRACT_PATH)
    block = _contract_block()
    assert raw["forwarder"]["lane_mirror"]["topics"] == (
        block["topic_sets"]["delegation"] + block["topic_sets"]["hook_edge"]
    )
    assert raw["forwarder"]["lane_mirror"]["max_messages_per_poll"] == 50
    assert raw["forwarder"]["lane_mirror"]["poll_timeout_ms"] == 1000
    assert "lane_mirror_binding" not in raw["forwarder"]
    assert "lane_mirror_set" not in raw["forwarder"]


@pytest.mark.asyncio
async def test_developer_tenant_bytes_and_headers_are_kept(
    lane_mirror_harness: Any,
) -> None:
    harness = lane_mirror_harness
    harness.kwargs["config"] = ModelGatewayLaneMirrorConfig(
        source_lane="dev-local",
        mirror_lanes=("dev",),
        declared_lanes=("dev-local", "dev"),
        topics=tuple(_contract_block()["topic_sets"]["delegation"]),
    )
    template = harness.record(
        envelope_id="developer-delegation",
        topic="onex.evt.omnibase-infra.delegation-completed.v1",
    )
    tenant_id = "11111111-2222-3333-4444-555555555555"
    envelope = json.loads(template.value)
    envelope["tenant_id"] = tenant_id
    value = json.dumps(envelope).encode("utf-8")
    offered = ModelTransportMessage(
        topic=template.topic,
        partition=template.partition,
        offset=template.offset,
        key=template.key,
        value=value,
        headers=template.headers,
        ack_token=template.ack_token,
    )
    harness.source.offer(offered)

    await NodeLaneMirror(**harness.kwargs).drain_once()

    sent = harness.mirrors["dev"].sent
    assert len(sent) == 1
    assert sent[0].value == value
    assert json.loads(sent[0].value)["tenant_id"] == tenant_id
    assert sent[0].headers == offered.headers
