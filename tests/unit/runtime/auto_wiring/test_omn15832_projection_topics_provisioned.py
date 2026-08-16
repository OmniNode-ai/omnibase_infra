# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary seam test: contract-declared projection_api topics MUST be
provisioned at boot (OMN-15832, OMN-14208 seam rule).

The seam has two sides, joined by the *contract YAML string*:

* **Serving side** -- ``omnimarket.projection.discovery.build_projection_topic_map``
  gates on ``projection_api.expose: true`` (section-level) and reads
  ``projection_api.topic`` / ``projection_api.exposures[].topic`` as the topics
  it will consume from once ``SnapshotCache`` connects.
* **Provisioning side** -- ``_contract_provision_topics`` decides which topics
  the per-contract boot interleave (OMN-13237) creates + readiness-confirms
  before the consumer attaches.

Before OMN-15832 the provisioning side had zero ``projection_api`` awareness:
it unioned only ``event_bus.subscribe_topics`` / ``publish_topics`` /
``dlq_topics``. On onex-dev (managed MSK, auto-create off,
``ONEX_BOOT_UNIVERSE_PROVISION=0`` since the 2026-07-27 near-meltdown) that
meant ``onex.snapshot.projection.registration.v1`` was NEVER created, and
``SnapshotCache._wait_topics`` raised ``UnknownTopicOrPartitionError`` on
first boot of ``omnimarket-projection-api`` (live: deploy run 31389089749,
2026-08-10T12:57Z).

This test drives the real per-contract provision function off ONE on-disk
contract shaped exactly like ``node_projection_registration`` and asserts the
topic ``SnapshotCache`` will wait on is a topic the provisioner actually
creates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _contract_provision_topics,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)

SUBSCRIBE_TOPIC = "onex.evt.platform.node-heartbeat.v1"
PUBLISH_TOPIC = "onex.evt.omnimarket.projection-registration-applied.v1"
DLQ_TOPIC = "onex.dlq.omnimarket.projection-registration-malformed.v1"
# The live onex-dev name UnknownTopicOrPartitionError fired on, 2026-08-10T12:57Z.
SNAPSHOT_TOPIC = "onex.snapshot.projection.registration.v1"

CONTRACT_YAML_SERVED = f"""\
contract_version:
  major: 1
  minor: 0
  patch: 0
node_type: REDUCER_GENERIC
event_bus:
  subscribe_topics:
    - {SUBSCRIBE_TOPIC}
  publish_topics:
    - {PUBLISH_TOPIC}
  dlq_topics:
    - {DLQ_TOPIC}
projection_api:
  expose: true
  topic: "{SNAPSHOT_TOPIC}"
  table: "node_service_registry"
  schema: "public"
  bus_backed: true
  key_columns:
    - "service_name"
"""

CONTRACT_YAML_NOT_EXPOSED = CONTRACT_YAML_SERVED.replace(
    "expose: true", "expose: false"
)

CONTRACT_YAML_NOT_BUS_BACKED = CONTRACT_YAML_SERVED.replace(
    "bus_backed: true", "bus_backed: false"
)


def _write_contract(tmp_path: Path, yaml_text: str, node_name: str) -> Path:
    contract_dir = tmp_path / node_name
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / "contract.yaml"
    contract_path.write_text(yaml_text)
    return contract_path


def _discovered(contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_projection_registration",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="node_projection_registration",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(SUBSCRIBE_TOPIC,),
            publish_topics=(PUBLISH_TOPIC,),
        ),
    )


@pytest.mark.unit
def test_expose_true_bus_backed_true_projection_topic_is_provisioned(
    tmp_path: Path,
) -> None:
    """The topic SnapshotCache waits on MUST be one the provisioner creates.

    RED before OMN-15832 Phase B: ``_contract_provision_topics`` returned
    only subscribe+publish+dlq topics, so ``SNAPSHOT_TOPIC`` never reached the
    provisioner and the projection-api pod hit
    ``UnknownTopicOrPartitionError`` on the live onex-dev MSK cluster (auto-create
    off).
    """
    contract_path = _write_contract(
        tmp_path, CONTRACT_YAML_SERVED, "node_projection_registration"
    )
    contract = _discovered(contract_path)

    topics = _contract_provision_topics(contract)

    assert SNAPSHOT_TOPIC in topics, (
        "expose:true + bus_backed:true projection_api topic was never in the "
        f"boot provision set (provisioned={topics}) -- SnapshotCache._wait_topics "
        "will raise UnknownTopicOrPartitionError on a managed broker with "
        "auto-create off"
    )
    # Regression guard: existing event_bus coverage is unchanged.
    assert SUBSCRIBE_TOPIC in topics
    assert PUBLISH_TOPIC in topics
    assert DLQ_TOPIC in topics


@pytest.mark.unit
def test_expose_false_projection_topic_is_not_provisioned(tmp_path: Path) -> None:
    """``expose: false`` excludes the topic from the boot provision set even
    though ``bus_backed: true`` -- it is never served, so provisioning it
    would create a dead topic nothing reads (matches
    ``build_projection_topic_map``'s own gate)."""
    contract_path = _write_contract(
        tmp_path, CONTRACT_YAML_NOT_EXPOSED, "node_projection_registration"
    )
    contract = _discovered(contract_path)

    topics = _contract_provision_topics(contract)

    assert SNAPSHOT_TOPIC not in topics
    # event_bus topics are still provisioned regardless of projection_api state.
    assert SUBSCRIBE_TOPIC in topics
    assert PUBLISH_TOPIC in topics


@pytest.mark.unit
def test_bus_backed_false_projection_topic_is_not_provisioned(tmp_path: Path) -> None:
    """``bus_backed: false`` excludes the topic from the boot provision set --
    nothing publishes to it yet, so eagerly creating it is premature."""
    contract_path = _write_contract(
        tmp_path, CONTRACT_YAML_NOT_BUS_BACKED, "node_projection_registration"
    )
    contract = _discovered(contract_path)

    topics = _contract_provision_topics(contract)

    assert SNAPSHOT_TOPIC not in topics


@pytest.mark.unit
def test_provision_set_has_no_duplicate_projection_topic(tmp_path: Path) -> None:
    """A projection topic that also appears under ``event_bus.publish_topics``
    (unlikely but not contract-forbidden) is not double-provisioned."""
    contract_dir = tmp_path / "node_projection_registration"
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / "contract.yaml"
    contract_path.write_text(
        f"""\
event_bus:
  subscribe_topics:
    - {SUBSCRIBE_TOPIC}
  publish_topics:
    - {SNAPSHOT_TOPIC}
projection_api:
  expose: true
  topic: "{SNAPSHOT_TOPIC}"
  table: "node_service_registry"
  bus_backed: true
  key_columns:
    - "service_name"
"""
    )
    contract = _discovered(contract_path)

    topics = _contract_provision_topics(contract)

    assert topics.count(SNAPSHOT_TOPIC) == 1
