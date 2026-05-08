# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for contract-driven routing in HandlerDelegationRouting (OMN-10615).

Tests cover:
    - Graceful degradation when no contract file is present
    - cloud_routing_policy=blocked skips non-local tiers
    - pricing_ceiling_per_1k_tokens blocks expensive tiers
    - escalation_policy.tier_order reorders tier iteration
    - Default contract file is loaded and parses correctly
    - TASK_CLASS_CONTRACT_PATH env var overrides default path

Related:
    - OMN-10615: Wire routing reducer to read task-class contracts
    - OMN-10657: Contract-driven endpoint resolution
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

import omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing as _handler_mod
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing import (
    _task_class_entry,
    _tier_allowed_by_contract,
    _tier_order_from_contract,
    delta,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_tier import (
    ModelRoutingTier,
)

pytestmark = [pytest.mark.unit]

_DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[5]
    / "src"
    / "omnibase_infra"
    / "configs"
    / "task_class_contracts.v1.yaml"
)


def _request(
    task_type: str = "test",
    prompt: str = "Write unit tests for auth.py",
    **kwargs: object,
) -> ModelDelegationRequest:
    return ModelDelegationRequest(
        prompt=prompt,
        task_type=task_type,  # type: ignore[arg-type]
        correlation_id=uuid4(),
        emitted_at=datetime.now(tz=UTC),
        **kwargs,  # type: ignore[arg-type]
    )


def _make_tier(name: str) -> ModelRoutingTier:
    return ModelRoutingTier(name=name, models=())


def _write_bifrost(tmp_path: Path, backends: dict[str, str]) -> str:
    """Write a bifrost contract YAML with given backend_id -> endpoint_url mappings."""
    entries = []
    for bid, url in backends.items():
        entries.append(
            f"  - backend_id: {bid}\n"
            f'    endpoint_url: "{url}"\n'
            f"    model_name: "
            "\n"
            f"    tier: local\n"
            f"    timeout_ms: 30000\n"
            f"    capabilities: []"
        )
    backends_block = "\n".join(entries) if entries else "  []"
    content = (
        "config_version: '1.1.0'\n"
        "schema_version: bifrost_delegation.v1\n"
        "backends:\n" + backends_block + "\n"
    )
    path = tmp_path / "bifrost.yaml"
    path.write_text(content)
    return str(path)


@pytest.fixture(autouse=True)
def reset_singletons():  # type: ignore[no-untyped-def]
    """Reset module-level singletons before each test."""
    _handler_mod._config = None
    _handler_mod._get_task_class_contract.cache_clear()
    _handler_mod._load_bifrost_endpoints.cache_clear()
    yield
    _handler_mod._config = None
    _handler_mod._get_task_class_contract.cache_clear()
    _handler_mod._load_bifrost_endpoints.cache_clear()


class TestGracefulDegradation:
    """No contract file -> tier routing unchanged."""

    def test_absent_contract_path_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = str(tmp_path / "nonexistent.yaml")
        monkeypatch.setenv("TASK_CLASS_CONTRACT_PATH", missing)
        contract = _handler_mod._get_task_class_contract()
        assert contract is None

    def test_no_env_var_and_default_missing_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TASK_CLASS_CONTRACT_PATH", raising=False)
        monkeypatch.setattr(
            _handler_mod,
            "_DEFAULT_TASK_CLASS_CONTRACT_PATH",
            Path("/nonexistent/path/contract.yaml"),
        )
        contract = _handler_mod._get_task_class_contract()
        assert contract is None

    def test_routing_works_without_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        monkeypatch.delenv("TASK_CLASS_CONTRACT_PATH", raising=False)
        monkeypatch.setattr(
            _handler_mod,
            "_DEFAULT_TASK_CLASS_CONTRACT_PATH",
            Path("/nonexistent/path/contract.yaml"),
        )
        req = _request(task_type="test", prompt="x" * 200000)
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"


class TestCloudBlockedPolicy:
    """cloud_routing_policy=blocked must skip non-local tiers."""

    def test_tier_allowed_local_when_blocked(self) -> None:
        entry = {"cloud_routing_policy": "blocked"}
        assert _tier_allowed_by_contract(_make_tier("local"), entry) is True

    def test_tier_blocked_cheap_cloud_when_blocked(self) -> None:
        entry = {"cloud_routing_policy": "blocked"}
        assert _tier_allowed_by_contract(_make_tier("cheap_cloud"), entry) is False

    def test_tier_blocked_claude_when_blocked(self) -> None:
        entry = {"cloud_routing_policy": "blocked"}
        assert _tier_allowed_by_contract(_make_tier("claude"), entry) is False

    def test_tier_allowed_cli_agents_when_blocked(self) -> None:
        entry = {"cloud_routing_policy": "blocked"}
        assert _tier_allowed_by_contract(_make_tier("cli_agents"), entry) is True

    def test_delta_skips_cloud_when_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_yaml = textwrap.dedent("""\
            version: "1.0"
            task_classes:
              test:
                required_capabilities: []
                pricing_ceiling_per_1k_tokens: 0.002
                latency_sla_p99_ms: 30000
                cloud_routing_policy: blocked
                definition_of_done:
                  deterministic: []
                  heuristic: []
                escalation_policy:
                  max_escalations: 0
                  tier_order: []
        """)
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(contract_yaml)
        monkeypatch.setenv("TASK_CLASS_CONTRACT_PATH", str(contract_path))

        # No local tier endpoints + cloud blocked -> should raise
        bifrost = _write_bifrost(
            tmp_path, {"cloud-sonnet": "https://api.anthropic.com"}
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)

        req = _request(task_type="test")
        with pytest.raises(ProtocolConfigurationError, match="No tier"):
            delta(req)

    def test_delta_uses_local_when_cloud_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_yaml = textwrap.dedent("""\
            version: "1.0"
            task_classes:
              test:
                required_capabilities: []
                pricing_ceiling_per_1k_tokens: 0.002
                latency_sla_p99_ms: 30000
                cloud_routing_policy: blocked
                definition_of_done:
                  deterministic: []
                  heuristic: []
                escalation_policy:
                  max_escalations: 0
                  tier_order: []
        """)
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(contract_yaml)
        monkeypatch.setenv("TASK_CLASS_CONTRACT_PATH", str(contract_path))
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)

        req = _request(task_type="test", prompt="x" * 200000)
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"
        assert decision.cost_tier == "low"


class TestPricingCeiling:
    """pricing_ceiling_per_1k_tokens must block tiers exceeding the ceiling."""

    def test_ceiling_allows_local(self) -> None:
        entry = {"pricing_ceiling_per_1k_tokens": 0.001}
        assert _tier_allowed_by_contract(_make_tier("local"), entry) is True

    def test_ceiling_blocks_claude(self) -> None:
        entry = {"pricing_ceiling_per_1k_tokens": 0.001}
        assert _tier_allowed_by_contract(_make_tier("claude"), entry) is False

    def test_ceiling_blocks_cheap_cloud_when_tight(self) -> None:
        entry = {"pricing_ceiling_per_1k_tokens": 0.001}
        assert _tier_allowed_by_contract(_make_tier("cheap_cloud"), entry) is False

    def test_ceiling_allows_cheap_cloud_when_generous(self) -> None:
        entry = {"pricing_ceiling_per_1k_tokens": 0.005}
        assert _tier_allowed_by_contract(_make_tier("cheap_cloud"), entry) is True

    def test_no_ceiling_allows_all(self) -> None:
        entry: dict[str, object] = {}
        for tier_name in ("local", "cheap_cloud", "claude"):
            assert _tier_allowed_by_contract(_make_tier(tier_name), entry) is True

    def test_delta_skips_claude_when_ceiling_too_low(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_yaml = textwrap.dedent("""\
            version: "1.0"
            task_classes:
              test:
                required_capabilities: []
                pricing_ceiling_per_1k_tokens: 0.001
                latency_sla_p99_ms: 30000
                cloud_routing_policy: allowed
                definition_of_done:
                  deterministic: []
                  heuristic: []
                escalation_policy:
                  max_escalations: 0
                  tier_order: []
        """)
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(contract_yaml)
        monkeypatch.setenv("TASK_CLASS_CONTRACT_PATH", str(contract_path))
        # Only cloud-sonnet available but ceiling blocks it
        bifrost = _write_bifrost(
            tmp_path, {"cloud-sonnet": "https://api.anthropic.com"}
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)

        req = _request(task_type="test")
        with pytest.raises(ProtocolConfigurationError, match="No tier"):
            delta(req)


class TestEscalationTierOrder:
    """escalation_policy.tier_order reorders tier iteration."""

    def test_tier_order_reorders(self) -> None:
        from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_delegation_config import (
            ModelDelegationConfig,
        )

        local = ModelRoutingTier(name="local", models=())
        cheap = ModelRoutingTier(name="cheap_cloud", models=())
        claude = ModelRoutingTier(name="claude", models=())

        config = ModelDelegationConfig(tiers=(local, cheap, claude))
        entry = {
            "escalation_policy": {"tier_order": ["claude", "local", "cheap_cloud"]}
        }

        ordered = _tier_order_from_contract(config, entry)
        assert [t.name for t in ordered] == ["claude", "local", "cheap_cloud"]

    def test_tier_order_appends_unlisted(self) -> None:
        from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_delegation_config import (
            ModelDelegationConfig,
        )

        local = ModelRoutingTier(name="local", models=())
        cheap = ModelRoutingTier(name="cheap_cloud", models=())
        claude = ModelRoutingTier(name="claude", models=())
        cli = ModelRoutingTier(name="cli_agents", models=())

        config = ModelDelegationConfig(tiers=(local, cheap, claude, cli))
        entry = {"escalation_policy": {"tier_order": ["claude"]}}

        ordered = _tier_order_from_contract(config, entry)
        names = [t.name for t in ordered]
        assert names[0] == "claude"
        assert set(names) == {"local", "cheap_cloud", "claude", "cli_agents"}

    def test_empty_tier_order_uses_default(self) -> None:
        from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_delegation_config import (
            ModelDelegationConfig,
        )

        local = ModelRoutingTier(name="local", models=())
        cheap = ModelRoutingTier(name="cheap_cloud", models=())
        config = ModelDelegationConfig(tiers=(local, cheap))
        entry: dict[str, object] = {"escalation_policy": {"tier_order": []}}

        ordered = _tier_order_from_contract(config, entry)
        assert ordered == config.tiers

    def test_none_entry_uses_default(self) -> None:
        from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_delegation_config import (
            ModelDelegationConfig,
        )

        local = ModelRoutingTier(name="local", models=())
        config = ModelDelegationConfig(tiers=(local,))

        ordered = _tier_order_from_contract(config, None)
        assert ordered == config.tiers


class TestTaskClassEntry:
    """_task_class_entry returns the right entry or None."""

    def test_returns_entry_when_present(self) -> None:
        contract = {"task_classes": {"test": {"cloud_routing_policy": "allowed"}}}
        assert _task_class_entry(contract, "test") == {
            "cloud_routing_policy": "allowed"
        }

    def test_returns_none_when_task_missing(self) -> None:
        contract = {"task_classes": {"code_generation": {}}}
        assert _task_class_entry(contract, "test") is None

    def test_returns_none_when_contract_none(self) -> None:
        assert _task_class_entry(None, "test") is None


class TestDefaultContractFile:
    """Verify the shipped task_class_contracts.v1.yaml is valid and parseable."""

    def test_default_contract_parses(self) -> None:
        assert _DEFAULT_CONTRACT_PATH.exists(), (
            f"Default contract file missing: {_DEFAULT_CONTRACT_PATH}"
        )
        raw = yaml.safe_load(_DEFAULT_CONTRACT_PATH.read_text())
        assert isinstance(raw, dict)
        assert "version" in raw
        assert "task_classes" in raw
        assert isinstance(raw["task_classes"], dict)

    def test_default_contract_has_expected_task_classes(self) -> None:
        raw = yaml.safe_load(_DEFAULT_CONTRACT_PATH.read_text())
        task_classes = raw["task_classes"]
        for expected in ("code_generation", "documentation", "research", "test"):
            assert expected in task_classes, f"Missing task class: {expected}"

    def test_default_contract_entries_have_required_fields(self) -> None:
        raw = yaml.safe_load(_DEFAULT_CONTRACT_PATH.read_text())
        for class_name, entry in raw["task_classes"].items():
            assert "cloud_routing_policy" in entry, (
                f"{class_name} missing cloud_routing_policy"
            )
            assert "pricing_ceiling_per_1k_tokens" in entry, (
                f"{class_name} missing pricing_ceiling_per_1k_tokens"
            )
            assert "escalation_policy" in entry, (
                f"{class_name} missing escalation_policy"
            )

    def test_default_contract_loads_via_get_task_class_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TASK_CLASS_CONTRACT_PATH", raising=False)
        contract = _handler_mod._get_task_class_contract()
        assert contract is not None
        assert "task_classes" in contract


class TestRationaleContractAnnotation:
    """Contract-driven routing appends policy annotation to rationale."""

    def test_rationale_includes_contract_info_when_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract_yaml = textwrap.dedent("""\
            version: "1.0"
            task_classes:
              test:
                required_capabilities: []
                pricing_ceiling_per_1k_tokens: 0.015
                latency_sla_p99_ms: 30000
                cloud_routing_policy: allowed
                definition_of_done:
                  deterministic: []
                  heuristic: []
                escalation_policy:
                  max_escalations: 0
                  tier_order: []
        """)
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(contract_yaml)
        monkeypatch.setenv("TASK_CLASS_CONTRACT_PATH", str(contract_path))
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)

        req = _request(task_type="test", prompt="x" * 200000)
        decision = delta(req)
        assert "Contract-driven" in decision.rationale
        assert "allowed" in decision.rationale
