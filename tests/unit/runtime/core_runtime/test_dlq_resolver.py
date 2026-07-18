# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the S6 DLQ resolver (OMN-14758, §b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.core_runtime.dlq_resolver import (
    build_delegation_dlq_resolver,
    derive_canonical_dlq_topic,
)

ROUTING_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"
DECLARED_DLQ = "onex.dlq.omnimarket.projection-delegation-malformed.v1"


def _contract(
    *, name: str, subscribe_topics: tuple[str, ...], path: Path
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=path,
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=subscribe_topics),
    )


def test_derive_canonical_dlq_is_onex_shaped() -> None:
    dlq = derive_canonical_dlq_topic(ROUTING_TOPIC)
    assert dlq == "onex.dlq.omnibase-infra.delegation-routing-request.v1"
    assert dlq.startswith("onex.dlq.")
    assert not dlq.endswith(".v1.dlq")


def test_derive_fail_closed_on_unparseable() -> None:
    with pytest.raises(ModelOnexError, match="ONEX"):
        derive_canonical_dlq_topic("not-an-onex-topic")
    with pytest.raises(ModelOnexError, match="ONEX"):
        derive_canonical_dlq_topic("onex.cmd.producer.event")  # no version segment


def test_derived_resolver_and_provision_set() -> None:
    contract = _contract(
        name="c",
        subscribe_topics=(ROUTING_TOPIC,),
        path=Path("/nonexistent/contract.yaml"),
    )
    resolver, provision = build_delegation_dlq_resolver(
        [contract], frozenset({ROUTING_TOPIC})
    )
    resolved = resolver(ROUTING_TOPIC)
    assert resolved == "onex.dlq.omnibase-infra.delegation-routing-request.v1"
    assert resolved in provision
    assert provision == frozenset({resolved})


def test_contract_declared_dlq_wins(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        "event_bus:\n"
        f"  subscribe_topics:\n    - {ROUTING_TOPIC}\n"
        f"  dlq_topics:\n    - {DECLARED_DLQ}\n"
    )
    contract = _contract(
        name="proj", subscribe_topics=(ROUTING_TOPIC,), path=contract_path
    )
    resolver, provision = build_delegation_dlq_resolver(
        [contract], frozenset({ROUTING_TOPIC})
    )
    assert resolver(ROUTING_TOPIC) == DECLARED_DLQ
    assert DECLARED_DLQ in provision


def test_resolver_derives_for_unmapped_wellformed_topic() -> None:
    # A topic outside the allowlist still resolves (RuntimeDispatch fails an unmapped
    # topic closed to DLQ using this resolver).
    resolver, _ = build_delegation_dlq_resolver([], frozenset({ROUTING_TOPIC}))
    unmapped = "onex.cmd.omnibase-infra.some-other-command.v1"
    assert resolver(unmapped) == "onex.dlq.omnibase-infra.some-other-command.v1"
