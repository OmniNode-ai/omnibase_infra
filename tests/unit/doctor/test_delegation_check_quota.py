# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline quota probes with injected credential and HTTP stubs."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import SecretStr

from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.doctor.checks import check_delegation_quota as module
from omnibase_infra.doctor.checks.check_delegation_quota import CheckDelegationQuota
from omnibase_infra.doctor.enum_delegation_doctor_fault import EnumDelegationDoctorFault
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.gateway.client.gateway_identity_verifier import (
    ProtocolWhoamiTransport,
)
from omnibase_infra.gateway.models.model_gateway_api_key import (
    ModelGatewayApiKeyCredential,
)
from omnibase_infra.gateway.models.model_gateway_credential_base import (
    ModelGatewayCredentialBase,
)


@pytest.fixture
def transport() -> Mock:
    stub = Mock(spec=ProtocolWhoamiTransport)
    stub.get = AsyncMock(return_value=Mock(status=200))
    return stub


@pytest.fixture
def credential_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mock:
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  tenant_slug: acme\n  base_url: https://gateway.invalid\n"
        "  api_key_ref: doctor-test\n",
        encoding="utf-8",
    )
    store = Mock()
    store.load_read_credential.return_value = ModelGatewayApiKeyCredential(
        tenant_slug="acme",
        base_url="https://gateway.invalid/",
        api_key=SecretStr("doctor-test-value"),
        api_key_ref="doctor-test",
    )
    monkeypatch.setattr(module, "StoreGatewayCredential", Mock(return_value=store))
    return store


@pytest.mark.parametrize(
    ("status", "expected_status", "reason"),
    [
        (200, EnumHealthStatusValue.HEALTHY, "probe was accepted"),
        (429, EnumHealthStatusValue.UNHEALTHY, "quota is exhausted"),
        (401, EnumHealthStatusValue.UNKNOWN, "delegation_key was refused"),
        (403, EnumHealthStatusValue.UNKNOWN, "delegation_key was refused"),
        (500, EnumHealthStatusValue.UNKNOWN, "HTTP 500"),
        (204, EnumHealthStatusValue.UNKNOWN, "HTTP 204"),
    ],
)
def test_quota_status_and_probe_contract(
    tmp_path: Path,
    transport: Mock,
    credential_store: Mock,
    status: int,
    expected_status: EnumHealthStatusValue,
    reason: str,
) -> None:
    transport.get.return_value.status = status
    check = CheckDelegationQuota(onex_home=tmp_path, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert result.status is expected_status
    assert reason in result.message
    if status == 429:
        assert diagnosis.fault is EnumDelegationDoctorFault.QUOTA_EXHAUSTED
        assert diagnosis.fix == (
            "Wait for the tenant quota window to reset, then re-run this check."
        )
        assert "[quota_exhausted]" in result.message
        assert diagnosis.fix in result.message
    else:
        assert diagnosis.fault is None
        assert diagnosis.fix == ""
    assert credential_store.load_read_credential.call_count == 2
    assert transport.get.await_count == 2
    transport.get.assert_awaited_with(
        "https://gateway.invalid/v1/whoami",
        timeout=5.0,
        headers={"x-api-key": "doctor-test-value", "Accept": "application/json"},
    )
    assert "doctor-test-value" not in diagnosis.detail + result.message


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (None, "delegation_identity is not configured"),
        ("{}", "delegation_identity is not configured"),
        ("gateway: [", "config.yaml is unreadable or invalid"),
        ("gateway: {}", "delegation_key is not configured"),
        ("gateway:\n  api_key_ref: ' '", "delegation_key is not configured"),
        ("gateway:\n  api_key_ref: 42", "delegation_key is not configured"),
    ],
)
def test_config_prerequisites_never_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Mock,
    config: str | None,
    reason: str,
) -> None:
    if config is not None:
        (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    store_factory = Mock(side_effect=AssertionError("must not load credentials"))
    monkeypatch.setattr(module, "StoreGatewayCredential", store_factory)
    check = CheckDelegationQuota(onex_home=tmp_path, transport=transport)

    assert reason in check.diagnose().detail
    assert check.diagnose().fault is None
    assert check.run().status is EnumHealthStatusValue.UNKNOWN
    store_factory.assert_not_called()
    transport.get.assert_not_awaited()


@pytest.mark.parametrize("kind", ["unreadable", "not-api-key"])
def test_unusable_credential_never_probes(
    tmp_path: Path, transport: Mock, credential_store: Mock, kind: str
) -> None:
    if kind == "unreadable":
        credential_store.load_read_credential.side_effect = ModelOnexError(
            "credential-store-private-detail"
        )
        reason = "delegation_key is not readable"
    else:
        credential_store.load_read_credential.return_value = ModelGatewayCredentialBase(
            tenant_slug="acme", base_url="https://gateway.invalid"
        )
        reason = "delegation_key is not an API key"
    check = CheckDelegationQuota(onex_home=tmp_path, transport=transport)

    assert reason in check.diagnose().detail
    assert check.diagnose().fault is None
    assert check.run().status is EnumHealthStatusValue.UNKNOWN
    transport.get.assert_not_awaited()


def test_gateway_unavailable_does_not_claim_quota_exhaustion(
    tmp_path: Path, transport: Mock, credential_store: Mock
) -> None:
    transport.get.side_effect = InfraUnavailableError(
        "probe-private-detail",
        context=ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.HTTP, operation="doctor_probe"
        ),
    )
    check = CheckDelegationQuota(onex_home=tmp_path, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is None
    assert "delegation_gateway could not be reached" in diagnosis.detail
    assert result.status is EnumHealthStatusValue.UNKNOWN
    assert "probe-private-detail" not in diagnosis.detail + result.message


@pytest.mark.parametrize("seam", ["reader", "credential", "transport"])
def test_unexpected_failures_are_total_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Mock,
    credential_store: Mock,
    seam: str,
) -> None:
    sensitive = "quota-probe-private-detail"
    failure = RuntimeError(sensitive)
    if seam == "reader":
        monkeypatch.setattr(module, "read_gateway_block", Mock(side_effect=failure))
    elif seam == "credential":
        credential_store.load_read_credential.side_effect = failure
    else:
        transport.get.side_effect = failure
    check = CheckDelegationQuota(onex_home=tmp_path, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is None
    assert diagnosis.fix == ""
    assert "check failed safely" in diagnosis.detail
    assert result.status is EnumHealthStatusValue.UNKNOWN
    assert "check failed safely" in result.message
    assert sensitive not in diagnosis.detail + result.message


def test_default_dependencies_are_bounded_and_use_default_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Mock,
    credential_store: Mock,
) -> None:
    monkeypatch.setattr(module, "default_onex_home", lambda: tmp_path)
    factory = Mock(return_value=transport)
    monkeypatch.setattr(module, "GatewayTransportHttpx", factory)

    assert CheckDelegationQuota().run().status is EnumHealthStatusValue.HEALTHY
    factory.assert_called_once_with(timeout_seconds=5.0)
