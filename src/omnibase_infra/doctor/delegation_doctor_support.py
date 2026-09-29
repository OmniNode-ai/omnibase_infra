# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared, secret-safe support for delegation doctor checks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Final

import yaml

from omnibase_core.enums.enum_doctor_category import EnumDoctorCategory
from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_core.models.doctor.model_doctor_check_result import (
    ModelDoctorCheckResult,
)
from omnibase_core.protocols.http.protocol_http_client import ProtocolHttpResponse
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)
from omnibase_infra.gateway.client.gateway_identity_verifier import (
    GATEWAY_WHOAMI_PATH,
    ProtocolWhoamiTransport,
)
from omnibase_infra.gateway.models.model_gateway_api_key import (
    ModelGatewayApiKeyCredential,
)

GATEWAY_BLOCK_VALID: Final[str] = "valid"
GATEWAY_BLOCK_MISSING: Final[str] = "missing"
GATEWAY_BLOCK_INVALID: Final[str] = "invalid"
DOCTOR_HTTP_TIMEOUT_SECONDS: Final[float] = 5.0


def default_onex_home() -> Path:
    """Return the standard ONEX user configuration directory."""
    return Path.home() / ".onex"


def read_gateway_block(onex_home: Path) -> tuple[str, dict[str, object] | None]:
    """Read only the non-secret gateway block from the ONEX config."""
    path = onex_home / "config.yaml"
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return (GATEWAY_BLOCK_MISSING, None)
    except (OSError, UnicodeError, yaml.YAMLError):
        return (GATEWAY_BLOCK_INVALID, None)

    if not isinstance(raw, dict):
        return (GATEWAY_BLOCK_INVALID, None)
    block = raw.get("gateway")
    if block is None:
        return (GATEWAY_BLOCK_MISSING, None)
    if not isinstance(block, dict):
        return (GATEWAY_BLOCK_INVALID, None)
    return (GATEWAY_BLOCK_VALID, {str(key): value for key, value in block.items()})


def gateway_text(block: dict[str, object], key: str) -> str | None:
    """Return a stripped, non-empty string from a gateway block."""
    value = block.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def run_async[T](operation: Callable[[], Coroutine[object, object, T]]) -> T:
    """Run one bounded async doctor probe from the synchronous doctor API."""
    return asyncio.run(operation())


async def request_whoami_status(
    *,
    transport: ProtocolWhoamiTransport,
    credential: ModelGatewayApiKeyCredential,
) -> int:
    """Return the raw whoami status without exposing credential material."""
    response = await transport.get(
        f"{credential.base_url.rstrip('/')}{GATEWAY_WHOAMI_PATH}",
        timeout=DOCTOR_HTTP_TIMEOUT_SECONDS,
        headers={
            "x-api-key": credential.api_key.get_secret_value(),
            "Accept": "application/json",
        },
    )
    return response.status


async def request_models(
    *,
    transport: ProtocolWhoamiTransport,
    url: str,
) -> ProtocolHttpResponse:
    """Fetch one OpenAI-compatible models document."""
    return await transport.get(
        url,
        timeout=DOCTOR_HTTP_TIMEOUT_SECONDS,
        headers={"Accept": "application/json"},
    )


def render_diagnosis(
    *,
    name: str,
    diagnosis: ModelDelegationDiagnosis,
    judged: bool,
) -> ModelDoctorCheckResult:
    """Render a typed diagnosis into the core doctor result contract."""
    if not judged:
        status = EnumHealthStatusValue.UNKNOWN
        message = diagnosis.detail
    elif diagnosis.fault is None:
        status = EnumHealthStatusValue.HEALTHY
        message = diagnosis.detail
    else:
        status = EnumHealthStatusValue.UNHEALTHY
        message = f"[{diagnosis.fault.value}] {diagnosis.detail} Fix: {diagnosis.fix}"
    return ModelDoctorCheckResult(
        name=name,
        category=EnumDoctorCategory.SERVICES,
        status=status,
        message=message,
    )


def unknown_result(*, name: str, dependency: str) -> ModelDoctorCheckResult:
    """Return the total-function fallback used by every check's run method."""
    return ModelDoctorCheckResult(
        name=name,
        category=EnumDoctorCategory.SERVICES,
        status=EnumHealthStatusValue.UNKNOWN,
        message=f"Delegation {dependency} was not judged because the check failed safely.",
    )


__all__ = [
    "DOCTOR_HTTP_TIMEOUT_SECONDS",
    "GATEWAY_BLOCK_INVALID",
    "GATEWAY_BLOCK_MISSING",
    "GATEWAY_BLOCK_VALID",
    "default_onex_home",
    "gateway_text",
    "read_gateway_block",
    "render_diagnosis",
    "request_models",
    "request_whoami_status",
    "run_async",
    "unknown_result",
]
