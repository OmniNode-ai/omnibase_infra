# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two old import paths installed consumers load resolve to the moved models (OMN-22478)."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import SecretStr

from omnibase_infra.handlers.models.infisical import ModelInfisicalHandlerConfig
from omnibase_infra.handlers.models.model_retry_state import (
    ModelRetryState as RetryStateAtOldPath,
)
from omnibase_infra.models.resilience.model_retry_state import ModelRetryState
from omnibase_infra.models.security.model_infisical_config import ModelInfisicalConfig

_HANDLERS_MODELS = Path(__file__).resolve().parents[3] / (
    "src/omnibase_infra/handlers/models"
)


@pytest.mark.unit
def test_old_infisical_name_is_the_moved_model() -> None:
    assert ModelInfisicalHandlerConfig is ModelInfisicalConfig


@pytest.mark.unit
def test_old_retry_state_path_is_the_moved_model() -> None:
    assert RetryStateAtOldPath is ModelRetryState


@pytest.mark.unit
def test_consumer_call_shape_builds_without_reading_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """omnimarket's bootstrap passes every value by keyword, the host included."""
    monkeypatch.delenv("INFISICAL_ADDR", raising=False)
    config = ModelInfisicalHandlerConfig(
        host="http://infisical.invalid",
        client_id=SecretStr("id"),
        client_secret=SecretStr("value"),
        project_id=UUID(int=1),
        environment_slug="dev",
        secret_path="/shared/env/",
    )
    assert config.host == "http://infisical.invalid"


@pytest.mark.unit
@pytest.mark.parametrize("module", ["infisical/__init__.py", "model_retry_state.py"])
def test_old_paths_hold_no_definition(module: str) -> None:
    """They stay re-exports: the move to the shared models stands."""
    source = (_HANDLERS_MODELS / module).read_text(encoding="utf-8")
    assert "class " not in source
