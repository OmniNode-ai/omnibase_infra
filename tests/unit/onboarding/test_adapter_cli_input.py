# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for CLI input adapter."""

import asyncio

import pytest

pytestmark = pytest.mark.unit


def _make_step(
    step_id: str, step_type: str, options: list[str] | None = None
) -> object:
    from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep

    return ModelInteractiveStep(
        id=step_id,
        prompt="Select an option:",
        type=step_type,  # type: ignore[arg-type]
        options=options or [],
    )


def test_collect_choice_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnibase_infra.onboarding.adapter_cli_input import AdapterCliInput

    step = _make_step("choose_mode", "choice", ["local", "cloud", "hybrid"])
    monkeypatch.setattr("builtins.input", lambda _: "local")
    adapter = AdapterCliInput()
    result = asyncio.run(adapter.collect_choice(step))  # type: ignore[arg-type]
    assert result == "local"


def test_collect_choice_trims_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnibase_infra.onboarding.adapter_cli_input import AdapterCliInput

    step = _make_step("choose_mode", "choice", ["local", "cloud"])
    responses = iter(["  local  "])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    adapter = AdapterCliInput()
    result = asyncio.run(adapter.collect_choice(step))  # type: ignore[arg-type]
    assert result == "local"


def test_collect_choice_rejects_unknown_then_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnibase_infra.onboarding.adapter_cli_input import AdapterCliInput

    step = _make_step("choose_mode", "choice", ["local", "cloud"])
    responses = iter(["bogus", "cloud"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    adapter = AdapterCliInput()
    result = asyncio.run(adapter.collect_choice(step))  # type: ignore[arg-type]
    assert result == "cloud"


def test_collect_multi_choice_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnibase_infra.onboarding.adapter_cli_input import AdapterCliInput

    step = _make_step(
        "choose_services", "multi_choice", ["kafka", "postgres", "llm_inference"]
    )
    monkeypatch.setattr("builtins.input", lambda _: "kafka, postgres")
    adapter = AdapterCliInput()
    result = asyncio.run(adapter.collect_multi_choice(step))  # type: ignore[arg-type]
    assert result == ["kafka", "postgres"]


def test_collect_multi_choice_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnibase_infra.onboarding.adapter_cli_input import AdapterCliInput

    step = _make_step("choose_services", "multi_choice", ["kafka", "postgres"])
    responses = iter(["kafka, bogus_service", "kafka"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    adapter = AdapterCliInput()
    result = asyncio.run(adapter.collect_multi_choice(step))  # type: ignore[arg-type]
    assert result == ["kafka"]


def test_collect_text_returns_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnibase_infra.onboarding.adapter_cli_input import AdapterCliInput

    step = _make_step("enter_endpoint", "text")
    monkeypatch.setattr("builtins.input", lambda _: "  http://localhost:8000  ")
    adapter = AdapterCliInput()
    result = asyncio.run(adapter.collect_text(step))  # type: ignore[arg-type]
    assert result == "http://localhost:8000"


def test_notify_action_does_not_raise() -> None:
    from omnibase_infra.onboarding.adapter_cli_input import AdapterCliInput

    step = _make_step("write_config", "action")
    adapter = AdapterCliInput()
    asyncio.run(adapter.notify_action(step))  # type: ignore[arg-type]
