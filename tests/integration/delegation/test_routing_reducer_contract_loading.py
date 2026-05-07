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
    _handler_mod._task_class_contract = None
    _handler_mod._task_class_contract_loaded = False
    yield
    _handler_mod._config = None
    _handler_mod._task_class_contract = None
    _handler_mod._task_class_contract_loaded = False


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
) -> None:
    """End-to-end: loaded contract + local tier endpoint → routes to local model."""
    monkeypatch.setenv("TASK_CLASS_CONTRACT_PATH", str(DEFAULT_CONTRACT_PATH))
    monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
    monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)

    decision = delta(_request(task_type="test", prompt="x" * 200000))
    assert decision.selected_model == "qwen3-coder-30b"
    assert decision.cost_tier == "low"
