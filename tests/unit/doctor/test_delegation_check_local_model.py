# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline local-model diagnosis and models-endpoint contract tests."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.doctor.checks import check_delegation_local_model as module
from omnibase_infra.doctor.checks.check_delegation_local_model import (
    CheckDelegationLocalModel,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import EnumDelegationDoctorFault
from omnibase_infra.doctor.protocol_models_transport import ProtocolModelsTransport
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext

_ENDPOINT = "https://model.invalid/proxy/v1/chat/completions"
_MODEL = "local/test-model"


@pytest.fixture
def transport() -> Mock:
    response = Mock(status=200)
    response.text = AsyncMock(return_value=json.dumps({"data": [{"id": _MODEL}]}))
    stub = Mock(spec=ProtocolModelsTransport)
    stub.get = AsyncMock(return_value=response)
    return stub


@pytest.fixture
def overlay(tmp_path: Path) -> Path:
    path = tmp_path / "overlay.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {"tier": "local", "endpoint_url": _ENDPOINT, "model_name": _MODEL}
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "config",
    [
        None,
        "{}",
        "backends: {}",
        "backends: []",
        "backends:\n  - ignored\n  - tier: cloud\n  - tier: local",
        "backends:\n  - tier: local\n    endpoint_url: http://embed.invalid/v1/embeddings",
    ],
)
def test_absent_chat_backend_never_probes(
    tmp_path: Path, transport: Mock, config: str | None
) -> None:
    path = tmp_path / "overlay.yaml"
    if config is not None:
        path.write_text(config, encoding="utf-8")
    check = CheckDelegationLocalModel(overlay_path=path, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is EnumDelegationDoctorFault.NO_LOCAL_MODEL
    assert str(path) in diagnosis.fix
    assert "tier: local" in diagnosis.fix
    assert result.status is EnumHealthStatusValue.UNHEALTHY
    assert "[no_local_model]" in result.message
    transport.get.assert_not_awaited()


@pytest.mark.parametrize("config", ["[]", "", "backends: ["])
def test_invalid_overlay_is_unknown(
    tmp_path: Path, transport: Mock, config: str
) -> None:
    path = tmp_path / "overlay.yaml"
    path.write_text(config, encoding="utf-8")
    check = CheckDelegationLocalModel(overlay_path=path, transport=transport)

    assert check.diagnose().fault is None
    assert "overlay is unreadable or invalid" in check.diagnose().detail
    assert check.run().status is EnumHealthStatusValue.UNKNOWN
    transport.get.assert_not_awaited()


@pytest.mark.parametrize("kind", ["directory", "invalid-utf8"])
def test_unreadable_overlay_is_unknown(
    tmp_path: Path, transport: Mock, kind: str
) -> None:
    path = tmp_path / "overlay.yaml"
    if kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"\xff")
    check = CheckDelegationLocalModel(overlay_path=path, transport=transport)

    assert check.diagnose().fault is None
    assert check.run().status is EnumHealthStatusValue.UNKNOWN
    transport.get.assert_not_awaited()


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://model.invalid/v1/chat/completions",
        "/v1/chat/completions",
        "http:///v1/chat/completions",
        "https://model.invalid/chat/completions",
    ],
)
def test_invalid_endpoint_is_unknown_and_never_probes(
    overlay: Path, transport: Mock, endpoint: str
) -> None:
    overlay.write_text(
        yaml.safe_dump({"backends": [{"tier": "local", "endpoint_url": endpoint}]}),
        encoding="utf-8",
    )
    check = CheckDelegationLocalModel(overlay_path=overlay, transport=transport)

    assert "local endpoint URL is invalid" in check.diagnose().detail
    assert check.diagnose().fault is None
    assert check.run().status is EnumHealthStatusValue.UNKNOWN
    transport.get.assert_not_awaited()


@pytest.mark.parametrize("status", [503, 401, 404])
def test_http_failure_reports_not_serving_without_reading_body(
    overlay: Path, transport: Mock, status: int
) -> None:
    transport.get.return_value.status = status
    check = CheckDelegationLocalModel(overlay_path=overlay, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is EnumDelegationDoctorFault.LOCAL_MODEL_NOT_SERVING
    assert f"HTTP {status}" in diagnosis.detail
    assert diagnosis.fix == "Start the model server at https://model.invalid."
    assert result.status is EnumHealthStatusValue.UNHEALTHY
    assert "[local_model_not_serving]" in result.message
    transport.get.return_value.text.assert_not_awaited()


def test_unreachable_model_server_reports_not_serving(
    overlay: Path, transport: Mock
) -> None:
    transport.get.side_effect = InfraUnavailableError(
        "transport-private-detail",
        context=ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.HTTP, operation="doctor_probe"
        ),
    )
    check = CheckDelegationLocalModel(overlay_path=overlay, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is EnumDelegationDoctorFault.LOCAL_MODEL_NOT_SERVING
    assert "could not be reached" in diagnosis.detail
    assert diagnosis.fix == "Start the model server at https://model.invalid."
    assert result.status is EnumHealthStatusValue.UNHEALTHY
    assert "transport-private-detail" not in diagnosis.detail + result.message


@pytest.mark.parametrize("body", ["not-json", "[]", "{}", '{"data": {}}'])
def test_invalid_models_response_is_unknown(
    overlay: Path, transport: Mock, body: str
) -> None:
    transport.get.return_value.text.return_value = body
    check = CheckDelegationLocalModel(overlay_path=overlay, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is None
    assert diagnosis.fix == ""
    assert "models response was unreadable or invalid" in diagnosis.detail
    assert result.status is EnumHealthStatusValue.UNKNOWN
    assert result.message == diagnosis.detail


@pytest.mark.parametrize(
    ("data", "reported"),
    [
        (
            [{"id": " z/model "}, {"id": "a/model"}, {"id": "a/model"}],
            "a/model, z/model",
        ),
    ],
)
def test_model_id_mismatch_lists_sorted_valid_ids(
    overlay: Path, transport: Mock, data: list[object], reported: str
) -> None:
    transport.get.return_value.text.return_value = json.dumps({"data": data})
    check = CheckDelegationLocalModel(overlay_path=overlay, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is EnumDelegationDoctorFault.LOCAL_MODEL_ID_MISMATCH
    assert diagnosis.detail == (
        f"The overlay declares '{_MODEL}', but the server reports {reported}."
    )
    assert str(overlay) in diagnosis.fix
    assert "set model_name" in diagnosis.fix
    assert result.status is EnumHealthStatusValue.UNHEALTHY
    assert "[local_model_id_mismatch]" in result.message


@pytest.mark.parametrize(
    "model_fields",
    [
        {"model_name": f" {_MODEL} ", "served_model_id": "ignored"},
        {"model_name": " ", "served_model_id": f" {_MODEL} "},
        {"model_name": 42, "served_model_id": _MODEL},
        {},
    ],
)
def test_healthy_model_and_bounded_models_request(
    overlay: Path, transport: Mock, model_fields: dict[str, object]
) -> None:
    backend = {
        "tier": "local",
        "endpoint_url": f" {_ENDPOINT}/ ",
        **model_fields,
    }
    overlay.write_text(
        yaml.safe_dump({"backends": [{"tier": "cloud"}, backend]}), encoding="utf-8"
    )
    transport.get.return_value.text.return_value = json.dumps(
        {"data": [{"id": f" {_MODEL} "}]}
    )
    check = CheckDelegationLocalModel(overlay_path=overlay, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is None
    assert diagnosis.fix == ""
    assert result.status is EnumHealthStatusValue.HEALTHY
    assert result.message == diagnosis.detail
    if model_fields:
        assert f"Local model '{_MODEL}' is serving" in diagnosis.detail
    else:
        assert "A local model server is answering" in diagnosis.detail
    assert transport.get.await_count == 2
    transport.get.assert_awaited_with(
        "https://model.invalid/proxy/v1/models",
        timeout=5.0,
        headers={"Accept": "application/json"},
    )


@pytest.mark.parametrize(
    "failure", [TypeError("invalid response"), RuntimeError("private")]
)
def test_transport_exceptions_are_total(
    overlay: Path, transport: Mock, failure: Exception
) -> None:
    transport.get.side_effect = failure
    check = CheckDelegationLocalModel(overlay_path=overlay, transport=transport)

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is None
    assert diagnosis.fix == ""
    assert result.status is EnumHealthStatusValue.UNKNOWN
    if isinstance(failure, TypeError):
        assert "models response was unreadable or invalid" in diagnosis.detail
    else:
        assert "check failed safely" in diagnosis.detail
        assert "check failed safely" in result.message
    assert str(failure) not in diagnosis.detail + result.message


@pytest.mark.parametrize(
    "binding", ["explicit", "environment", "default", "os-environ"]
)
def test_overlay_resolution_and_default_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Mock,
    binding: str,
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    factory = Mock(return_value=transport)
    monkeypatch.setattr(module, "GatewayTransportHttpx", factory)
    path = tmp_path / ".omninode" / "delegation" / "bifrost_overrides.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump({"backends": [{"tier": "local", "endpoint_url": _ENDPOINT}]}),
        encoding="utf-8",
    )
    if binding == "explicit":
        check = CheckDelegationLocalModel(
            overlay_path=path, environ={"BIFROST_OVERLAY_PATH": "/ignored.yaml"}
        )
    elif binding == "environment":
        check = CheckDelegationLocalModel(
            environ={
                "BIFROST_OVERLAY_PATH": " ~/.omninode/delegation/bifrost_overrides.yaml "
            }
        )
    elif binding == "os-environ":
        monkeypatch.setenv("BIFROST_OVERLAY_PATH", str(path))
        check = CheckDelegationLocalModel()
    else:
        check = CheckDelegationLocalModel(environ={"BIFROST_OVERLAY_PATH": " "})

    assert check.run().status is EnumHealthStatusValue.HEALTHY
    factory.assert_called_once_with(timeout_seconds=5.0)
