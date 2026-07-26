# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed guards for the handler_routing flat-schema defect class (OMN-14141).

The flat ``handler_class:``/``handler_module:`` schema silently parsed to zero
handlers and then phantom-wired the subscribed topic (WIRED with no dispatcher,
Kafka offsets still committed) — the WI-14 root cause (OMN-14139/OMN-14135).
These tests pin the three enforcement surfaces:

1. Parse guard — ``discovery._parse_handler_routing`` raises on a flat/malformed
   entry, does NOT raise on the nested shape or an empty handlers list, and does
   NOT break the legacy top-level ``handler:`` fallback.
2. Wire guard — ``wire_from_manifest`` reports FAILED (not WIRED) when a contract
   subscribes to topics but registers zero dispatchers, and does NOT fire on a
   valid nested contract.
3. CI/pre-commit gate — the ``handler_routing_schema`` validator flags flat
   entries and passes nested + legacy-fallback contracts.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import (
    _parse_contract,
    _parse_handler_routing,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.validators.handler_routing_schema import main, validate_file


class TestParseGuardFlatSchema:
    """Guard A: discovery._parse_handler_routing fails closed on flat entries."""

    @pytest.mark.unit
    def test_flat_handler_class_module_entry_raises(self) -> None:
        hr_raw = {
            "routing_strategy": "operation_match",
            "handlers": [
                {
                    "operation": "x.do",
                    "handler_class": "HandlerX",
                    "handler_module": "pkg.handlers.handler_x",
                }
            ],
        }
        with pytest.raises(ValueError, match="OMN-14141"):
            _parse_handler_routing(hr_raw)

    @pytest.mark.unit
    def test_non_dict_entry_raises(self) -> None:
        hr_raw = {"routing_strategy": "operation_match", "handlers": ["not-a-dict"]}
        with pytest.raises(ValueError, match="must be a mapping"):
            _parse_handler_routing(hr_raw)

    @pytest.mark.unit
    def test_nested_handler_entry_parses(self) -> None:
        hr_raw = {
            "routing_strategy": "operation_match",
            "handlers": [
                {
                    "operation": "x.do",
                    "handler": {"name": "HandlerX", "module": "pkg.handlers.handler_x"},
                }
            ],
        }
        routing = _parse_handler_routing(hr_raw)
        assert len(routing.handlers) == 1
        assert routing.handlers[0].handler.name == "HandlerX"

    @pytest.mark.unit
    def test_empty_handlers_list_does_not_raise(self) -> None:
        # Empty/absent handlers is the precondition for the legacy fallback — it
        # must never raise (the loop simply does not run).
        routing = _parse_handler_routing(
            {"routing_strategy": "operation_match", "handlers": []}
        )
        assert routing.handlers == ()

    @pytest.mark.unit
    def test_legacy_top_level_handler_fallback_preserved(self, tmp_path: Path) -> None:
        # A contract that declares an empty handler_routing (default_handler only)
        # plus a top-level legacy handler: block must still resolve to one handler
        # and must NOT trip the flat-schema guard.
        path = tmp_path / "contract.yaml"
        path.write_text(
            dedent("""\
                name: legacy_node
                node_type: orchestrator
                handler:
                  module: omnimarket.handlers.ledger
                  class: HandlerLedger
                  input_model: omnimarket.models.ModelTick
                handler_routing:
                  default_handler: omnimarket.handlers.ledger:HandlerLedger
                event_bus:
                  subscribe_topics:
                    - onex.cmd.omnimarket.ledger-tick.v1
            """)
        )
        contract = _parse_contract(
            contract_path=path,
            entry_point_name="legacy_node",
            package_name="omnimarket",
            package_version="0.1.0",
        )
        assert contract.handler_routing is not None
        assert len(contract.handler_routing.handlers) == 1
        assert contract.handler_routing.handlers[0].handler.name == "HandlerLedger"


def _zero_handler_contract(
    *, subscribe_topics: tuple[str, ...]
) -> ModelDiscoveredContract:
    """A contract that declares routing + subscribe topics but zero handlers.

    This is the shape the flat-schema defect produced before the parse guard.
    Constructed directly (bypassing discovery) to exercise the wire-time
    backstop in isolation.
    """
    return ModelDiscoveredContract(
        name="node_phantom",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_phantom",
        package_name="local",
        event_bus=ModelEventBusWiring(subscribe_topics=subscribe_topics),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match", handlers=()
        ),
    )


class TestPhantomWiringGuard:
    """Guard B: wire_from_manifest fails closed on subscribe + zero dispatchers."""

    @pytest.mark.asyncio
    async def test_subscribe_with_zero_dispatchers_reports_failed(self) -> None:
        contract = _zero_handler_contract(
            subscribe_topics=("onex.evt.platform.test-input.v1",)
        )
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()

        report = await wire_from_manifest(
            manifest, engine, event_bus=None, subscribe_immediately=False
        )

        assert report.total_wired == 0
        result = report.results[0]
        assert result.outcome is EnumWiringOutcome.FAILED
        assert "phantom wiring" in (result.reason or "")
        assert result.dispatchers_registered == ()

    @pytest.mark.asyncio
    async def test_zero_handlers_without_subscribe_is_skipped_not_failed(self) -> None:
        # No subscribe topics → nothing is phantom-wired; the contract is a plain
        # SKIP (the guard must not fire on it).
        contract = _zero_handler_contract(subscribe_topics=())
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()

        report = await wire_from_manifest(
            manifest, engine, event_bus=None, subscribe_immediately=False
        )

        assert report.results[0].outcome is EnumWiringOutcome.SKIPPED


class TestHandlerRoutingSchemaValidator:
    """The CI/pre-commit gate flags flat entries and passes valid shapes."""

    @pytest.mark.unit
    def test_validator_flags_flat_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "contract.yaml"
        path.write_text(
            dedent("""\
                name: node_flat
                node_type: EFFECT_GENERIC
                handler_routing:
                  routing_strategy: "operation_match"
                  handlers:
                    - operation: "x.do"
                      handler_class: "HandlerX"
                      handler_module: "pkg.handlers.handler_x"
            """)
        )
        findings = validate_file(path)
        assert len(findings) == 1
        assert findings[0].operation == "x.do"
        assert main([str(path)]) == 1

    @pytest.mark.unit
    def test_validator_passes_nested_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "contract.yaml"
        path.write_text(
            dedent("""\
                name: node_nested
                node_type: EFFECT_GENERIC
                handler_routing:
                  routing_strategy: "operation_match"
                  handlers:
                    - operation: "x.do"
                      handler:
                        name: "HandlerX"
                        module: "pkg.handlers.handler_x"
            """)
        )
        assert validate_file(path) == []
        assert main([str(path)]) == 0

    @pytest.mark.unit
    def test_validator_passes_legacy_fallback_contract(self, tmp_path: Path) -> None:
        # Top-level legacy handler: with an empty handler_routing.handlers list —
        # no offending handlers[] entry, so the gate stays silent.
        path = tmp_path / "contract.yaml"
        path.write_text(
            dedent("""\
                name: legacy_node
                node_type: orchestrator
                handler:
                  module: omnimarket.handlers.ledger
                  class: HandlerLedger
                handler_routing:
                  default_handler: omnimarket.handlers.ledger:HandlerLedger
            """)
        )
        assert validate_file(path) == []
        assert main([str(path)]) == 0

    @pytest.mark.unit
    def test_validator_clean_on_repo_tree(self) -> None:
        # Regression fence: the shipped omnibase_infra contract tree must stay
        # free of flat-schema entries so the gate is green at HEAD.
        root = Path(__file__).resolve()
        for parent in root.parents:
            candidate = parent / "src" / "omnibase_infra"
            if candidate.is_dir():
                assert main([str(candidate)]) == 0
                return
        pytest.skip("src/omnibase_infra not found from test location")
