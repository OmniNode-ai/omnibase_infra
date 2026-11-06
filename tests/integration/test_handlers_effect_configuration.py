# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the handler package and its relocated configuration boundary."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import pytest
from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_core.validation.validator_no_io_outside_effects import (
    ValidatorNoIoOutsideEffects,
)
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.handler_http import HandlerHttpRest
from omnibase_infra.handlers.handler_infisical import HandlerInfisical
from omnibase_infra.models.security import ModelInfisicalConfig
from omnibase_infra.validation.infra_validators import validate_infra_patterns

if TYPE_CHECKING:
    from pytest_httpserver import HTTPServer

_SECURITY_MODELS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "omnibase_infra"
    / "models"
    / "security"
)
_IDENTITY = {
    "client_id": "synthetic-client",
    "client_secret": "synthetic-secret",
    "project_id": UUID("11111111-1111-1111-1111-111111111111"),
    "environment_slug": "dev",
}


@pytest.mark.parametrize("ambient_host", [None, "https://ambient.invalid"])
def test_infisical_model_requires_explicit_host(
    monkeypatch: pytest.MonkeyPatch, ambient_host: str | None
) -> None:
    if ambient_host is None:
        monkeypatch.delenv("INFISICAL_ADDR", raising=False)
    else:
        monkeypatch.setenv("INFISICAL_ADDR", ambient_host)

    with pytest.raises(ValidationError) as exc:
        ModelInfisicalConfig.model_validate(_IDENTITY)

    assert exc.value.errors()[0]["loc"] == ("host",)
    assert exc.value.errors()[0]["type"] == "missing"


def test_explicit_infisical_config_preserves_credentials_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFISICAL_ADDR", "https://ambient.invalid")
    config = ModelInfisicalConfig.model_validate(
        {**_IDENTITY, "host": "https://explicit.invalid"}
    )

    assert config.host == "https://explicit.invalid"
    assert config.client_id.get_secret_value() == _IDENTITY["client_id"]
    assert config.client_secret.get_secret_value() == _IDENTITY["client_secret"]
    assert config.cache_ttl_seconds == 300.0
    assert config.circuit_breaker_enabled


@pytest.mark.asyncio
async def test_handler_reports_missing_host_as_configuration_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INFISICAL_ADDR", raising=False)
    handler = HandlerInfisical(ModelONEXContainer())

    with pytest.raises(RuntimeHostError) as exc:
        await handler.initialize(dict(_IDENTITY))

    assert isinstance(exc.value.__cause__, ValidationError)
    assert exc.value.__cause__.errors()[0]["loc"] == ("host",)


def test_relocated_security_models_obey_model_naming() -> None:
    result = validate_infra_patterns(directory=_SECURITY_MODELS)

    assert result.is_valid, result.errors


@pytest.mark.asyncio
async def test_effect_handler_is_validated_and_executes_http(
    httpserver: HTTPServer,
) -> None:
    module = Path(inspect.getfile(HandlerHttpRest))
    result = ValidatorNoIoOutsideEffects().validate(
        [module.parent / "contract.yaml", module]
    )
    assert result.is_valid, result.issues
    httpserver.expect_request("/probe").respond_with_json({"effect": "observed"})
    handler = HandlerHttpRest(ModelONEXContainer())
    await handler.initialize({})
    try:
        output = await handler.execute(
            {
                "operation": "http.get",
                "payload": {"url": httpserver.url_for("/probe")},
            }
        )
        assert output.result["status"] == "success"
        assert output.result["payload"]["body"] == {"effect": "observed"}
        httpserver.check_assertions()
    finally:
        await handler.shutdown()
