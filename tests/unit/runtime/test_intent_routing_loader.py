# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for service_intent_routing_loader.

Tests the load_intent_routing_table function which reads
intent_consumption.intent_routing_table from contract YAML files.

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - service_intent_routing_loader: Implementation under test
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.service_intent_routing_loader import (
    load_intent_routing_table,
)

pytestmark = [pytest.mark.unit]


class TestLoadIntentRoutingTable:
    """Tests for load_intent_routing_table function."""

    def test_loads_valid_routing_table(self, tmp_path: Path) -> None:
        """Should load a valid intent routing table from contract YAML."""
        contract = {
            "intent_consumption": {
                "subscribed_intents": [
                    "consul.register",
                    "postgres.upsert_registration",
                ],
                "intent_routing_table": {
                    "consul.register": "node_registry_effect",
                    "consul.deregister": "node_registry_effect",
                    "postgres.upsert_registration": "node_registry_effect",
                },
            }
        }
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(yaml.dump(contract))

        result = load_intent_routing_table(contract_path)

        assert result == {
            "consul.register": "node_registry_effect",
            "consul.deregister": "node_registry_effect",
            "postgres.upsert_registration": "node_registry_effect",
        }

    def test_returns_empty_dict_for_missing_file(self, tmp_path: Path) -> None:
        """Should return empty dict when contract file does not exist."""
        contract_path = tmp_path / "nonexistent.yaml"

        result = load_intent_routing_table(contract_path)

        assert result == {}

    def test_returns_empty_dict_for_empty_file(self, tmp_path: Path) -> None:
        """Should return empty dict when contract file is empty."""
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text("")

        result = load_intent_routing_table(contract_path)

        assert result == {}

    def test_returns_empty_dict_for_no_intent_consumption(self, tmp_path: Path) -> None:
        """Should return empty dict when no intent_consumption section."""
        contract = {"event_bus": {"version": {"major": 1, "minor": 0, "patch": 0}}}
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(yaml.dump(contract))

        result = load_intent_routing_table(contract_path)

        assert result == {}

    def test_returns_empty_dict_for_no_routing_table(self, tmp_path: Path) -> None:
        """Should return empty dict when intent_consumption has no routing table."""
        contract = {
            "intent_consumption": {
                "subscribed_intents": ["consul.register"],
            }
        }
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(yaml.dump(contract))

        result = load_intent_routing_table(contract_path)

        assert result == {}

    def test_returns_empty_dict_for_invalid_yaml(self, tmp_path: Path) -> None:
        """Should return empty dict for invalid YAML content."""
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text("invalid: yaml: content: [")

        result = load_intent_routing_table(contract_path)

        assert result == {}

    def test_skips_non_string_entries(self, tmp_path: Path) -> None:
        """Should skip entries with non-string keys or values."""
        contract = {
            "intent_consumption": {
                "intent_routing_table": {
                    "consul.register": "node_registry_effect",
                    123: "bad_key",
                    "bad_value": 456,
                },
            }
        }
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(yaml.dump(contract))

        result = load_intent_routing_table(contract_path)

        assert result == {"consul.register": "node_registry_effect"}

    def test_returns_empty_dict_when_routing_table_is_not_dict(
        self, tmp_path: Path
    ) -> None:
        """Should return empty dict when routing table is a list instead of dict."""
        contract = {
            "intent_consumption": {
                "intent_routing_table": ["consul.register"],
            }
        }
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(yaml.dump(contract))

        result = load_intent_routing_table(contract_path)

        assert result == {}

    def test_loads_real_contract(self) -> None:
        """Should successfully load the actual registration orchestrator contract."""
        contract_path = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_registration_orchestrator"
            / "contract.yaml"
        )
        if not contract_path.exists():
            pytest.skip("Real contract.yaml not available")

        result = load_intent_routing_table(contract_path)

        # Should have at least the three declared intent types
        assert "consul.register" in result
        assert "postgres.upsert_registration" in result
        assert len(result) >= 2
