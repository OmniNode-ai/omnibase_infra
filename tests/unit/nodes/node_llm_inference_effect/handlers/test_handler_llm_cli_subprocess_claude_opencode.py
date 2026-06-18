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
def test_cli_agents_tier_omits_codex_cli_model() -> None:
    """OMN-13215: the shelled codex-cli model was removed from cli_agents.

    The delegation ceiling executes over the canonical HTTP path; no codex
    subprocess model remains in any routing tier.
    """
    tiers_path = (
        Path(__file__).parents[5] / "src/omnibase_infra/configs/routing_tiers.yaml"
    )
    data = yaml.safe_load(tiers_path.read_text())
    cli_tier = next(t for t in data["tiers"] if t["name"] == "cli_agents")
    model_ids = [m["id"] for m in cli_tier["models"]]
    assert "codex-cli" not in model_ids
    # No tier anywhere may declare a codex shell-out model.
    for tier in data["tiers"]:
        assert "codex-cli" not in [m["id"] for m in tier["models"]]


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
def test_codex_cli_model_no_longer_resolves_a_subprocess_config() -> None:
    """OMN-13215: the codex-cli shell-out config was removed.

    A request whose model is ``codex-cli`` must no longer resolve a CLI config —
    the handler reports UNAVAILABLE (cli not configured) instead of spawning the
    codex binary. The delegation ceiling runs over the canonical HTTP path.
    """
    from omnibase_infra.models.llm.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )
    from omnibase_infra.models.llm.model_llm_message import ModelLlmMessage
    from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
        _CLI_CONFIG_BY_MODEL,
        EnumCliBackendStatus,
        HandlerLlmCliSubprocess,
    )

    # The codex model has no CLI config mapping anymore.
    assert "codex-cli" not in _CLI_CONFIG_BY_MODEL

    handler = HandlerLlmCliSubprocess()
    req = ModelLlmInferenceRequest(
        base_url="http://localhost:1",
        messages=[ModelLlmMessage(role="user", content="hi")],
        model="codex-cli",
    )
    # No subprocess is spawned: with no resolved CLI, the handler fails closed.
    with patch("subprocess.run") as mock_run:
        _, status, detail = handler.execute_cli_inference(req)
    assert status == EnumCliBackendStatus.UNAVAILABLE
    assert "cli not configured" in detail
    mock_run.assert_not_called()


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
def test_codex_cli_operation_removed_from_contract() -> None:
    """OMN-13215: the inference.codex_cli operation was removed from the contract.

    The codex shell-out is gone; the ceiling executes over the canonical HTTP
    inference path (HandlerLlmOpenaiCompatible).
    """
    contract_path = (
        Path(__file__).parents[5]
        / "src/omnibase_infra/nodes/node_llm_inference_effect/contract.yaml"
    )
    data = yaml.safe_load(contract_path.read_text())
    ops = [h["operation"] for h in data["handler_routing"]["handlers"]]
    assert "inference.codex_cli" not in ops
