# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Direct, offline tests for the shared delegation doctor helpers."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import SecretStr

from omnibase_core.enums.enum_doctor_category import EnumDoctorCategory
from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.doctor.delegation_doctor_support import (
    default_onex_home,
    gateway_text,
    read_gateway_block,
    render_diagnosis,
    request_models,
    request_whoami_status,
    run_async,
    unknown_result,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.doctor.model_delegation_diagnosis import ModelDelegationDiagnosis
from omnibase_infra.gateway.models.model_gateway_api_key import (
    ModelGatewayApiKeyCredential,
)

pytestmark = pytest.mark.unit


@dataclass
class StubResponse:
    status: int
    body: str = "not a JSON document"
    text_reads: int = 0

    async def text(self) -> str:
        self.text_reads += 1
        return self.body


class StubTransport:
    """Record requests and return or raise immediately without network access."""

    def __init__(self, response: StubResponse, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, float | None, dict[str, str] | None]] = []

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> StubResponse:
        self.calls.append((url, timeout, headers))
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def credential() -> ModelGatewayApiKeyCredential:
    return ModelGatewayApiKeyCredential(
        tenant_slug="doctor-test",
        base_url="https://gateway.invalid///",
        api_key=SecretStr("doctor-test-key"),
        api_key_ref="doctor-test",
    )


def test_default_onex_home_uses_the_current_user_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert default_onex_home() == tmp_path / ".onex"
    assert not (tmp_path / ".onex").exists()


def test_read_gateway_block_missing_file(tmp_path: Path) -> None:
    assert read_gateway_block(tmp_path) == ("missing", None)


@pytest.mark.parametrize("document", ["{}", "other: configured", "gateway: null"])
def test_read_gateway_block_missing_gateway(tmp_path: Path, document: str) -> None:
    (tmp_path / "config.yaml").write_text(document, encoding="utf-8")

    assert read_gateway_block(tmp_path) == ("missing", None)


@pytest.mark.parametrize(
    "document",
    [
        pytest.param("gateway: [", id="unparseable-yaml"),
        pytest.param("", id="empty-document"),
        pytest.param("null", id="null-document"),
        pytest.param("[]", id="sequence-document"),
        pytest.param("scalar", id="scalar-document"),
        pytest.param("gateway: []", id="sequence-gateway"),
        pytest.param("gateway: scalar", id="string-gateway"),
        pytest.param("gateway: 42", id="numeric-gateway"),
        pytest.param("gateway: false", id="boolean-gateway"),
    ],
)
def test_read_gateway_block_invalid_document(tmp_path: Path, document: str) -> None:
    (tmp_path / "config.yaml").write_text(document, encoding="utf-8")

    assert read_gateway_block(tmp_path) == ("invalid", None)


def test_read_gateway_block_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_bytes(b"gateway: \xff")

    assert read_gateway_block(tmp_path) == ("invalid", None)


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_read_gateway_block_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError],
) -> None:
    paths: list[Path] = []

    def fail_read(path: Path, *, encoding: str) -> str:
        paths.append(path)
        assert encoding == "utf-8"
        raise error_type("config is unreadable")

    monkeypatch.setattr(Path, "read_text", fail_read)

    assert read_gateway_block(tmp_path) == ("invalid", None)
    assert paths == [tmp_path / "config.yaml"]


def test_read_gateway_block_returns_only_gateway_with_values_unchanged(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.yaml").write_text(
        'other: ignored\ngateway:\n  tenant_slug: "  café  "\n'
        "  retries: 3\n  enabled: false\n  options: [one, two]\n",
        encoding="utf-8",
    )

    assert read_gateway_block(tmp_path) == (
        "valid",
        {
            "tenant_slug": "  café  ",
            "retries": 3,
            "enabled": False,
            "options": ["one", "two"],
        },
    )


def test_read_gateway_block_empty_mapping_is_valid(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("gateway: {}", encoding="utf-8")

    assert read_gateway_block(tmp_path) == ("valid", {})


def test_current_behavior_gateway_key_stringification_collision_keeps_last_value(
    tmp_path: Path,
) -> None:
    """Distinct YAML keys can silently collide when converted to strings."""
    (tmp_path / "config.yaml").write_text(
        'gateway:\n  7: numeric key\n  "7": text key\n', encoding="utf-8"
    )

    assert read_gateway_block(tmp_path) == ("valid", {"7": "text key"})


def test_gateway_text_missing_key() -> None:
    assert gateway_text({"other": "value"}, "tenant_slug") is None


@pytest.mark.parametrize("value", [None, 7, False, [], {}, b"text", "", " \t\n"])
def test_gateway_text_rejects_non_text_or_blank_values(value: object) -> None:
    assert gateway_text({"tenant_slug": value}, "tenant_slug") is None


@pytest.mark.parametrize("value", ["tenant", " \t tenant \n", "\u2003café\u2003"])
def test_gateway_text_strips_text_without_mutating_block(value: str) -> None:
    block: dict[str, object] = {"tenant_slug": value}

    assert gateway_text(block, "tenant_slug") == value.strip()
    assert block == {"tenant_slug": value}


def test_run_async_calls_operation_once_and_returns_its_result() -> None:
    calls: list[str] = []
    result = object()

    async def operation() -> object:
        calls.append("called")
        return result

    assert run_async(operation) is result
    assert calls == ["called"]


def test_run_async_propagates_operation_error() -> None:
    error = RuntimeError("probe failed")

    async def operation() -> None:
        raise error

    with pytest.raises(RuntimeError, match="probe failed") as caught:
        run_async(operation)

    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 204, 301, 401, 403, 429, 500, 503])
async def test_request_whoami_status_returns_raw_status_with_bounded_request(
    credential: ModelGatewayApiKeyCredential, status: int
) -> None:
    response = StubResponse(status)
    transport = StubTransport(response)

    assert (
        await request_whoami_status(transport=transport, credential=credential)
        == status
    )
    assert transport.calls == [
        (
            "https://gateway.invalid/v1/whoami",
            5.0,
            {"x-api-key": "doctor-test-key", "Accept": "application/json"},
        )
    ]
    assert response.text_reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 204, 301, 401, 403, 429, 500, 503])
async def test_request_models_returns_response_without_parsing_or_authentication(
    status: int,
) -> None:
    response = StubResponse(status)
    transport = StubTransport(response)
    url = "https://models.invalid/v1/models"

    assert await request_models(transport=transport, url=url) is response
    assert transport.calls == [(url, 5.0, {"Accept": "application/json"})]
    assert response.text_reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", ["whoami", "models"])
@pytest.mark.parametrize("error_type", [TimeoutError, OSError])
async def test_http_helpers_propagate_transport_errors_without_retry(
    credential: ModelGatewayApiKeyCredential,
    probe: str,
    error_type: type[Exception],
) -> None:
    error = error_type("transport failed")
    response = StubResponse(200)
    transport = StubTransport(response, error)

    with pytest.raises(error_type, match="transport failed") as caught:
        if probe == "whoami":
            await request_whoami_status(transport=transport, credential=credential)
        else:
            await request_models(
                transport=transport, url="https://models.invalid/v1/models"
            )

    assert caught.value is error
    assert len(transport.calls) == 1
    assert transport.calls[0][1] == 5.0
    assert response.text_reads == 0


@pytest.mark.parametrize("judged", [False, True], ids=["unjudged", "judged"])
@pytest.mark.parametrize("fault", [None, *EnumDelegationDoctorFault])
def test_render_diagnosis_status_message_and_category(
    judged: bool, fault: EnumDelegationDoctorFault | None
) -> None:
    diagnosis = ModelDelegationDiagnosis(
        fault=fault,
        detail="Diagnostic detail.",
        fix="Repair the dependency." if fault is not None else "",
    )

    result = render_diagnosis(
        name="delegation_test", diagnosis=diagnosis, judged=judged
    )

    assert result.name == "delegation_test"
    assert result.category is EnumDoctorCategory.SERVICES
    if not judged:
        assert result.status is EnumHealthStatusValue.UNKNOWN
        assert result.message == "Diagnostic detail."
    elif fault is None:
        assert result.status is EnumHealthStatusValue.HEALTHY
        assert result.message == "Diagnostic detail."
    else:
        assert result.status is EnumHealthStatusValue.UNHEALTHY
        assert result.message == (
            f"[{fault.value}] Diagnostic detail. Fix: Repair the dependency."
        )


def test_unknown_result_identifies_the_unjudged_dependency() -> None:
    result = unknown_result(name="delegation_test", dependency="gateway identity")

    assert result.name == "delegation_test"
    assert result.category is EnumDoctorCategory.SERVICES
    assert result.status is EnumHealthStatusValue.UNKNOWN
    assert result.message == (
        "Delegation gateway identity was not judged because the check failed safely."
    )
