# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Boot wiring deferral COMPUTE node tests (OMN-21318).

Small direct tests of the pure handler: which contracts boot defers, the
routing class that wraps the decision, the models' validation, and the
contract's handler route.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.nodes.node_boot_wiring_deferral_compute.handler import (
    NodeBootWiringDeferralCompute,
    decide_deferred_contracts,
)
from omnibase_infra.nodes.node_boot_wiring_deferral_compute.models import (
    ModelBootWiringCandidate,
    ModelBootWiringDeferralRequest,
    ModelBootWiringDeferralResult,
)

pytestmark = pytest.mark.unit

_HANDLER_MODULE = "omnibase_infra.nodes.node_boot_wiring_deferral_compute.handler"


def _candidate(name: str, *topics: str) -> ModelBootWiringCandidate:
    """A deferrable contract candidate with its subscribe topics."""
    return ModelBootWiringCandidate(contract_name=name, subscribe_topics=topics)


def _golden_request() -> ModelBootWiringDeferralRequest:
    """node_a shares one topic with the broker's reading; the rest idle."""
    return ModelBootWiringDeferralRequest(
        candidates=(
            _candidate("node_b", "t-b"),
            _candidate("node_a", "t-a1", "t-a2"),
            _candidate("node_c", "t-c"),
        ),
        in_use_topics=frozenset({"t-a2"}),
    )


def test_defers_only_contracts_with_no_topic_in_use() -> None:
    """A contract is deferred only when NONE of its topics is in use."""
    request = _golden_request()
    result = decide_deferred_contracts(request)
    assert isinstance(result, ModelBootWiringDeferralResult)
    # node_a is kept: one of its topics (t-a2) is in use, though t-a1 is not.
    assert result.deferred_contract_names == ("node_b", "node_c")
    # The names are sorted by the handler, not taken in the input's order.
    reordered = request.model_copy(
        update={"candidates": tuple(reversed(request.candidates))}
    )
    assert decide_deferred_contracts(reordered).deferred_contract_names == (
        "node_b",
        "node_c",
    )


def test_handle_matches_the_pure_decision() -> None:
    """The routed handler class delegates to the pure module function."""
    request = _golden_request()
    assert NodeBootWiringDeferralCompute().handle(request) == decide_deferred_contracts(
        request
    )


def test_no_candidates_defers_nothing() -> None:
    """Without candidates there is nothing to defer."""
    request = ModelBootWiringDeferralRequest(in_use_topics=frozenset({"t-b"}))
    assert decide_deferred_contracts(request).deferred_contract_names == ()


def test_every_topic_in_use_defers_nothing() -> None:
    """An unreadable broker reads as all topics in use: boot keeps every handler."""
    request = ModelBootWiringDeferralRequest(
        candidates=_golden_request().candidates,
        in_use_topics=frozenset({"t-a1", "t-a2", "t-b", "t-c"}),
    )
    assert decide_deferred_contracts(request).deferred_contract_names == ()


def test_empty_subscribe_topics_is_rejected() -> None:
    with pytest.raises(ValidationError, match="subscribe_topics"):
        ModelBootWiringCandidate(contract_name="node_x", subscribe_topics=())


def test_empty_contract_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="contract_name"):
        ModelBootWiringCandidate(contract_name="", subscribe_topics=("t-b",))


def test_unknown_field_is_rejected() -> None:
    # Unpacked rather than passed literally: the pydantic mypy plugin forbids
    # unknown kwargs in a typed constructor, and this field must only be
    # rejected at runtime by extra="forbid".
    extra_fields: dict[str, Any] = {"unknown_field": "surprise"}
    with pytest.raises(ValidationError, match="unknown_field"):
        ModelBootWiringCandidate(
            contract_name="node_x", subscribe_topics=("t-b",), **extra_fields
        )


def test_contract_yaml_routes_to_the_handler_class() -> None:
    """The contract's handler route names the class this module imports."""
    spec = importlib.util.find_spec(_HANDLER_MODULE)
    assert spec is not None and spec.origin is not None
    contract_path = Path(spec.origin).parent / "contract.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    entry = contract["handler_routing"]["handlers"][0]["handler"]
    assert entry["module"] == _HANDLER_MODULE
    assert entry["name"] == "NodeBootWiringDeferralCompute"
    resolved = getattr(importlib.import_module(entry["module"]), entry["name"])
    assert resolved is NodeBootWiringDeferralCompute
