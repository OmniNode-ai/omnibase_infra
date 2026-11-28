# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26616 AC4: a local model is optional; doctor fails only when no model is set up."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.doctor.checks.check_delegation_local_model import (
    CheckDelegationLocalModel,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import EnumDelegationDoctorFault
from omnibase_infra.doctor.protocol_models_transport import ProtocolModelsTransport
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.handlers import (
    handler_check_delegation_local_model as handler_module,
)

_ENDPOINT = "https://model.invalid/proxy/v1/chat/completions"
_MODEL = "local/test-model"


def _transport() -> Mock:
    response = Mock(status=200)
    response.text = AsyncMock(return_value=json.dumps({"data": [{"id": _MODEL}]}))
    stub = Mock(spec=ProtocolModelsTransport)
    stub.get = AsyncMock(return_value=response)
    return stub


def _declared_overlay(tmp_path: Path) -> Path:
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


def test_hosted_provider_and_no_local_model_does_not_fail_doctor(
    tmp_path: Path,
) -> None:
    transport = _transport()
    check = CheckDelegationLocalModel(
        overlay_path=tmp_path / "absent-overlay.yaml",
        environ={},
        transport=transport,
        hosted_models_reader=lambda: ("gemini", "openrouter"),
    )

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is None
    assert result.status is EnumHealthStatusValue.HEALTHY
    assert "gemini" in result.message
    assert "optional" in result.message
    assert ".yaml" not in result.message
    transport.get.assert_not_awaited()


@pytest.mark.parametrize("config", [None, "backends: []"])
def test_no_model_at_all_fails_with_a_cli_fix(
    tmp_path: Path, config: str | None
) -> None:
    path = tmp_path / "overlay.yaml"
    if config is not None:
        path.write_text(config, encoding="utf-8")
    transport = _transport()
    check = CheckDelegationLocalModel(
        overlay_path=path,
        environ={},
        transport=transport,
        hosted_models_reader=lambda: (),
    )

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is EnumDelegationDoctorFault.NO_LOCAL_MODEL
    assert "onex" in diagnosis.fix
    assert "/config" in diagnosis.fix
    for hidden in ("onex models", "onex setup", ".yaml"):
        assert hidden not in diagnosis.fix
    assert str(path) not in diagnosis.fix
    assert result.status is EnumHealthStatusValue.UNHEALTHY
    transport.get.assert_not_awaited()


def test_unreadable_model_list_is_not_judged(tmp_path: Path) -> None:
    check = CheckDelegationLocalModel(
        overlay_path=tmp_path / "absent-overlay.yaml",
        environ={},
        transport=_transport(),
        hosted_models_reader=lambda: None,
    )

    assert check.diagnose().fault is None
    assert check.run().status is EnumHealthStatusValue.UNKNOWN


def test_declared_but_unreachable_local_model_still_fails_beside_a_hosted_one(
    tmp_path: Path,
) -> None:
    transport = Mock(spec=ProtocolModelsTransport)
    transport.get = AsyncMock(
        side_effect=InfraUnavailableError(
            "down",
            context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP, operation="get"
            ),
        )
    )
    check = CheckDelegationLocalModel(
        overlay_path=_declared_overlay(tmp_path),
        environ={},
        transport=transport,
        hosted_models_reader=lambda: ("gemini",),
    )

    diagnosis = check.diagnose()

    assert diagnosis.fault is EnumDelegationDoctorFault.LOCAL_MODEL_NOT_SERVING
    assert check.run().status is EnumHealthStatusValue.UNHEALTHY
    transport.get.assert_awaited()


def _rule_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rule: str | None,
    hosted: tuple[str, ...],
) -> CheckDelegationLocalModel:
    monkeypatch.setattr(handler_module, "read_model_rule", lambda: rule)
    return CheckDelegationLocalModel(
        overlay_path=tmp_path / "absent-overlay.yaml",
        environ={},
        transport=_transport(),
        hosted_models_reader=lambda: hosted,
    )


def test_local_only_rule_with_only_hosted_providers_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = _rule_check(tmp_path, monkeypatch, "local", ("gemini",))

    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is EnumDelegationDoctorFault.NO_LOCAL_MODEL
    assert result.status is EnumHealthStatusValue.UNHEALTHY
    assert "local models only" in diagnosis.detail
    assert "uses your hosted" not in diagnosis.detail
    assert "uses your hosted" not in result.message
    assert "onex" in diagnosis.fix
    assert "/config" in diagnosis.fix
    for hidden in ("onex models", "onex setup", ".yaml"):
        assert hidden not in diagnosis.fix


@pytest.mark.parametrize("rule", [None, "free"])
def test_other_rules_keep_a_hosted_provider_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rule: str | None
) -> None:
    check = _rule_check(tmp_path, monkeypatch, rule, ("gemini",))

    assert check.diagnose().fault is None
    assert check.run().status is EnumHealthStatusValue.HEALTHY
