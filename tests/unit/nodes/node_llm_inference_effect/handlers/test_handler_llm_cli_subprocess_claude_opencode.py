# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for OAuth/headless CLI backend registration in omnibase_infra.

OMN-10137: Register claude and opencode as CLI subprocess backends.
OMN-13158: Register Codex as the primary headless CLI subprocess backend.
"""

from __future__ import annotations

import subprocess
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
def test_cli_agents_tier_contains_codex_cli_model() -> None:
    tiers_path = (
        Path(__file__).parents[5] / "src/omnibase_infra/configs/routing_tiers.yaml"
    )
    data = yaml.safe_load(tiers_path.read_text())
    cli_tier = next(t for t in data["tiers"] if t["name"] == "cli_agents")
    model_ids = [m["id"] for m in cli_tier["models"]]
    assert model_ids[0] == "codex-cli"


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
def test_codex_cli_subprocess_resolves_from_request_model() -> None:
    from omnibase_infra.models.llm.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )
    from omnibase_infra.models.llm.model_llm_message import ModelLlmMessage
    from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
        EnumCliBackendStatus,
        HandlerLlmCliSubprocess,
    )

    handler = HandlerLlmCliSubprocess()
    req = ModelLlmInferenceRequest(
        base_url="http://localhost:1",
        messages=[ModelLlmMessage(role="user", content="hi")],
        model="codex-cli",
    )
    with patch("shutil.which", return_value=None):
        _, status, detail = handler.execute_cli_inference(req)
    assert status == EnumCliBackendStatus.UNAVAILABLE
    assert detail == "codex not found on PATH"


@pytest.mark.unit
def test_codex_cli_subprocess_uses_headless_oauth_safe_argv() -> None:
    from omnibase_infra.models.llm.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )
    from omnibase_infra.models.llm.model_llm_message import ModelLlmMessage
    from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
        EnumCliBackendStatus,
        HandlerLlmCliSubprocess,
    )

    handler = HandlerLlmCliSubprocess()
    req = ModelLlmInferenceRequest(
        base_url="http://localhost:1",
        messages=[ModelLlmMessage(role="user", content="hi")],
        model="codex-cli",
    )
    completed = subprocess.CompletedProcess(
        args=["codex"],
        returncode=0,
        stdout="codex response\n",
        stderr="",
    )
    with (
        patch("shutil.which", return_value="/opt/homebrew/bin/codex"),
        patch("subprocess.run", return_value=completed) as mock_run,
    ):
        response, status, detail = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.SUCCESS
    assert detail == ""
    assert response is not None
    assert response.generated_text == "codex response"
    assert mock_run.call_args.args[0] == [
        "codex",
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
        "exec",
        "hi",
    ]


@pytest.mark.unit
def test_claude_cli_subprocess_resolves_headless_oauth_argv_from_request_model() -> (
    None
):
    from omnibase_infra.models.llm.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )
    from omnibase_infra.models.llm.model_llm_message import ModelLlmMessage
    from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
        EnumCliBackendStatus,
        HandlerLlmCliSubprocess,
    )

    handler = HandlerLlmCliSubprocess()
    req = ModelLlmInferenceRequest(
        base_url="http://localhost:1",
        messages=[ModelLlmMessage(role="user", content="hi")],
        model="claude-cli",
    )
    completed = subprocess.CompletedProcess(
        args=["claude"],
        returncode=0,
        stdout="claude response\n",
        stderr="",
    )
    with (
        patch("shutil.which", return_value="/opt/homebrew/bin/claude"),
        patch("subprocess.run", return_value=completed) as mock_run,
    ):
        response, status, detail = handler.execute_cli_inference(req)

    assert status == EnumCliBackendStatus.SUCCESS
    assert detail == ""
    assert response is not None
    assert response.generated_text == "claude response"
    assert mock_run.call_args.args[0] == [
        "claude",
        "-p",
        "--output-format",
        "text",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "hi",
    ]


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
