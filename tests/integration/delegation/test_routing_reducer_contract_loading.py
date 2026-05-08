# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: routing reducer reads on-disk task-class contracts (OMN-10615).

Exercises the full contract-loading path against the real
`configs/task_class_contracts.v1.yaml` shipped in the repo. Unit tests already
cover individual enforcement helpers with synthetic contracts; this module
proves the reducer wires to the canonical contract file at its installed
location and that contract-driven enforcement (cloud_routing_policy,
pricing_ceiling_per_1k_tokens, escalation_policy.tier_order) remains correct
when the file is loaded from disk rather than constructed in a tmp_path.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

import omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing as _handler_mod
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing import (
    delta,
)

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT_PATH = (
    REPO_ROOT / "src" / "omnibase_infra" / "configs" / "task_class_contracts.v1.yaml"
)


def _request(
    task_type: str, prompt: str = "Write unit tests for auth.py"
) -> ModelDelegationRequest:
    return ModelDelegationRequest(
        prompt=prompt,
        task_type=task_type,
        correlation_id=uuid4(),
        emitted_at=datetime.now(tz=UTC),
    )


@pytest.fixture(autouse=True)
def reset_singletons() -> Iterator[None]:
    _handler_mod._config = None
    _handler_mod._get_task_class_contract.cache_clear()
    yield
    _handler_mod._config = None
    _handler_mod._get_task_class_contract.cache_clear()


def test_default_contract_file_exists_at_canonical_path() -> None:
    """The default contract file must exist on disk for production wiring."""
    assert DEFAULT_CONTRACT_PATH.exists(), (
        f"Default task-class contract missing at {DEFAULT_CONTRACT_PATH}"
    )


def test_default_contract_file_parses_to_expected_shape() -> None:
    """Loaded contract must declare task_classes for every routed task type."""
    raw = yaml.safe_load(DEFAULT_CONTRACT_PATH.read_text())
    assert isinstance(raw, dict)
    task_classes = raw.get("task_classes")
    assert isinstance(task_classes, dict)
    for required in (
        "code_generation",
        "documentation",
        "research",
        "test",
        "reasoning",
        "summarization",
        "agent_delegation",
        "escalation",
    ):
        assert required in task_classes, f"Missing task class: {required}"


def test_reducer_loads_default_contract_from_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no env override, the reducer reads the bundled contract YAML from disk."""
    monkeypatch.delenv("TASK_CLASS_CONTRACT_PATH", raising=False)
    contract = _handler_mod._get_task_class_contract()
    assert contract is not None
    assert isinstance(contract, dict)
    assert "task_classes" in contract


def test_reducer_honors_env_override_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK_CLASS_CONTRACT_PATH env var redirects loading to the chosen file."""
    monkeypatch.setenv("TASK_CLASS_CONTRACT_PATH", str(DEFAULT_CONTRACT_PATH))
    contract = _handler_mod._get_task_class_contract()
    assert contract is not None
    task_classes = contract.get("task_classes")
    assert isinstance(task_classes, dict)
    assert "test" in task_classes


def test_routing_with_contract_loaded_picks_local_tier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end: loaded task-class contract + bifrost endpoint → routes to local model.

    Endpoint URLs are now resolved from the bifrost contract (OMN-10657), not env vars.
    This test writes a temp bifrost YAML with a populated endpoint for local-qwen-coder-30b
    and verifies the reducer picks the local tier.
    """
    # Write a minimal bifrost contract with the local backend endpoint populated.
    bifrost_yaml = tmp_path / "bifrost.yaml"
    bifrost_yaml.write_text(
        "config_version: '1.1.0'\n"
        "schema_version: bifrost_delegation.v1\n"
        "backends:\n"
        "  - backend_id: local-qwen-coder-30b\n"
        '    endpoint_url: "http://192.168.86.201:8000"\n'
        "    model_name: cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit\n"
        "    tier: local\n"
        "    timeout_ms: 30000\n"
        "    capabilities: []\n"
    )
    monkeypatch.setenv("TASK_CLASS_CONTRACT_PATH", str(DEFAULT_CONTRACT_PATH))
    monkeypatch.setenv("BIFROST_CONTRACT_PATH", str(bifrost_yaml))

    decision = delta(_request(task_type="test", prompt="x" * 200000))
    assert decision.selected_model == "cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
    assert decision.cost_tier == "low"
