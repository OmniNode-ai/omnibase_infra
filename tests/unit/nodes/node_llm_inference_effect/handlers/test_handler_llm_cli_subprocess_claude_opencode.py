# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for claude and opencode CLI backend registration in omnibase_infra.

OMN-10137: Register claude and opencode as CLI subprocess backends.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


@pytest.mark.unit
def test_claude_handler_is_registered_in_contract() -> None:
    contract_path = (
        Path(__file__).parents[5]
        / "src/omnibase_infra/nodes/node_llm_inference_effect/contract.yaml"
    )
    data = yaml.safe_load(contract_path.read_text())
    ops = [h["operation"] for h in data["handler_routing"]["handlers"]]
    assert "inference.claude_cli" in ops


@pytest.mark.unit
def test_opencode_handler_is_registered_in_contract() -> None:
    contract_path = (
        Path(__file__).parents[5]
        / "src/omnibase_infra/nodes/node_llm_inference_effect/contract.yaml"
    )
    data = yaml.safe_load(contract_path.read_text())
    ops = [h["operation"] for h in data["handler_routing"]["handlers"]]
    assert "inference.opencode_cli" in ops


@pytest.mark.unit
def test_claude_cli_tier_is_registered_in_routing_tiers() -> None:
    tiers_path = (
        Path(__file__).parents[5] / "src/omnibase_infra/configs/routing_tiers.yaml"
    )
    data = yaml.safe_load(tiers_path.read_text())
    tier_names = [t["name"] for t in data["tiers"]]
    assert "cli_agents" in tier_names


@pytest.mark.unit
def test_cli_agents_tier_contains_claude_cli_model() -> None:
    tiers_path = (
        Path(__file__).parents[5] / "src/omnibase_infra/configs/routing_tiers.yaml"
    )
    data = yaml.safe_load(tiers_path.read_text())
    cli_tier = next(t for t in data["tiers"] if t["name"] == "cli_agents")
    model_ids = [m["id"] for m in cli_tier["models"]]
    assert "claude-cli" in model_ids


@pytest.mark.unit
def test_cli_agents_tier_contains_opencode_cli_model() -> None:
    tiers_path = (
        Path(__file__).parents[5] / "src/omnibase_infra/configs/routing_tiers.yaml"
    )
    data = yaml.safe_load(tiers_path.read_text())
    cli_tier = next(t for t in data["tiers"] if t["name"] == "cli_agents")
    model_ids = [m["id"] for m in cli_tier["models"]]
    assert "opencode-cli" in model_ids


@pytest.mark.unit
def test_claude_cli_subprocess_unavailable_when_binary_missing() -> None:
    from omnibase_infra.models.llm.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )
    from omnibase_infra.models.llm.model_llm_message import ModelLlmMessage
    from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
        EnumCliBackendStatus,
        HandlerLlmCliSubprocess,
    )

    handler = HandlerLlmCliSubprocess(
        cli="claude", cli_args=["-p", "--output-format", "json"]
    )
    req = ModelLlmInferenceRequest(
        base_url="http://localhost:1",
        messages=[ModelLlmMessage(role="user", content="hi")],
        model="claude-cli",
    )
    with patch("shutil.which", return_value=None):
        _, status, _ = handler.execute_cli_inference(req)
    assert status == EnumCliBackendStatus.UNAVAILABLE


@pytest.mark.unit
def test_opencode_cli_subprocess_unavailable_when_binary_missing() -> None:
    from omnibase_infra.models.llm.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )
    from omnibase_infra.models.llm.model_llm_message import ModelLlmMessage
    from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
        EnumCliBackendStatus,
        HandlerLlmCliSubprocess,
    )

    handler = HandlerLlmCliSubprocess(
        cli="opencode", cli_args=["run", "--format", "json", "--pure"]
    )
    req = ModelLlmInferenceRequest(
        base_url="http://localhost:1",
        messages=[ModelLlmMessage(role="user", content="hi")],
        model="opencode-cli",
    )
    with patch("shutil.which", return_value=None):
        _, status, _ = handler.execute_cli_inference(req)
    assert status == EnumCliBackendStatus.UNAVAILABLE


@pytest.mark.unit
def test_existing_gemini_cli_entry_still_present() -> None:
    contract_path = (
        Path(__file__).parents[5]
        / "src/omnibase_infra/nodes/node_llm_inference_effect/contract.yaml"
    )
    data = yaml.safe_load(contract_path.read_text())
    ops = [h["operation"] for h in data["handler_routing"]["handlers"]]
    assert "inference.gemini_cli" in ops


@pytest.mark.unit
def test_existing_codex_cli_entry_still_present() -> None:
    contract_path = (
        Path(__file__).parents[5]
        / "src/omnibase_infra/nodes/node_llm_inference_effect/contract.yaml"
    )
    data = yaml.safe_load(contract_path.read_text())
    ops = [h["operation"] for h in data["handler_routing"]["handlers"]]
    assert "inference.codex_cli" in ops
