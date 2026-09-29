# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Acceptance tests for named customer delegation faults (OMN-19453)."""

import importlib
import inspect
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.doctor.doctor_check_base import DoctorCheckBase
from omnibase_core.doctor.doctor_registry import DoctorRegistry
from omnibase_core.enums.enum_doctor_category import EnumDoctorCategory
from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_core.models.doctor.model_doctor_check_result import (
    ModelDoctorCheckResult,
)
from omnibase_infra.doctor.checks.check_delegation_gateway import (
    CheckDelegationGateway,
)
from omnibase_infra.doctor.checks.check_delegation_identity import (
    CheckDelegationIdentity,
)
from omnibase_infra.doctor.checks.check_delegation_key import CheckDelegationKey
from omnibase_infra.doctor.checks.check_delegation_local_model import (
    CheckDelegationLocalModel,
)
from omnibase_infra.doctor.checks.check_delegation_quota import CheckDelegationQuota
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)

pytestmark = pytest.mark.unit

_BASE_URL = "https://gateway.invalid"
_API_KEY = "onxk_omn19453-never-log-this"  # pragma: allowlist secret
_MODEL_ID = "local/test-model"
_OVERLAY_ENDPOINT = "http://model.invalid/v1/chat/completions"
_EXPECTED_ENTRY_POINTS = {
    "delegation_identity": (
        "omnibase_infra.doctor.checks.check_delegation_identity:CheckDelegationIdentity"
    ),
    "delegation_key": (
        "omnibase_infra.doctor.checks.check_delegation_key:CheckDelegationKey"
    ),
    "delegation_gateway": (
        "omnibase_infra.doctor.checks.check_delegation_gateway:CheckDelegationGateway"
    ),
    "delegation_quota": (
        "omnibase_infra.doctor.checks.check_delegation_quota:CheckDelegationQuota"
    ),
    "delegation_local_model": (
        "omnibase_infra.doctor.checks.check_delegation_local_model:"
        "CheckDelegationLocalModel"
    ),
}


@dataclass(frozen=True)
class FakeResponse:
    status: int
    body: str = "{}"

    async def text(self) -> str:
        return self.body


class FakeTransport:
    def __init__(
        self,
        *,
        status: int = 200,
        body: str = "{}",
        unavailable: bool = False,
    ) -> None:
        self.status = status
        self.body = body
        self.unavailable = unavailable
        self.urls: list[str] = []

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeResponse:
        del timeout, headers
        self.urls.append(url)
        if self.unavailable:
            raise InfraUnavailableError(
                "no route to host",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="doctor_probe",
                ),
            )
        return FakeResponse(self.status, self.body)


class DiagnosticCheck(Protocol):
    def diagnose(self) -> ModelDelegationDiagnosis: ...

    def run(self) -> ModelDoctorCheckResult: ...


def _save_gateway(home: Path) -> None:
    StoreGatewayCredential(onex_home=home).save_api_key(
        tenant_slug="acme",
        api_key=_API_KEY,
        base_url=_BASE_URL,
    )


def _write_identity_without_key(home: Path) -> None:
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "gateway:\n  tenant_slug: acme\n  base_url: https://gateway.invalid\n"
    )


def _write_overlay(path: Path, *, model_name: str | None = _MODEL_ID) -> None:
    model_line = f"    model_name: {model_name}\n" if model_name else ""
    path.write_text(
        "backends:\n"
        "  - backend_id: local-test\n"
        "    tier: local\n"
        f"    endpoint_url: {_OVERLAY_ENDPOINT}\n"
        f"{model_line}"
    )


def _identity_body() -> str:
    return json.dumps({"tenant_id": str(uuid4()), "tenant_slug": "acme"})


@pytest.fixture
def fault_checks(tmp_path: Path) -> tuple[DiagnosticCheck, ...]:
    wrong_key_home = tmp_path / "wrong-key"
    gateway_down_home = tmp_path / "gateway-down"
    quota_home = tmp_path / "quota"
    for home in (wrong_key_home, gateway_down_home, quota_home):
        _save_gateway(home)

    no_key_home = tmp_path / "no-key"
    _write_identity_without_key(no_key_home)

    not_serving_overlay = tmp_path / "not-serving.yaml"
    mismatch_overlay = tmp_path / "mismatch.yaml"
    _write_overlay(not_serving_overlay)
    _write_overlay(mismatch_overlay)

    return (
        CheckDelegationIdentity(onex_home=tmp_path / "no-identity"),
        CheckDelegationKey(onex_home=no_key_home),
        CheckDelegationKey(
            onex_home=wrong_key_home,
            transport=FakeTransport(status=401),
        ),
        CheckDelegationGateway(
            onex_home=gateway_down_home,
            transport=FakeTransport(unavailable=True),
        ),
        CheckDelegationQuota(
            onex_home=quota_home,
            transport=FakeTransport(status=429),
        ),
        CheckDelegationLocalModel(
            overlay_path=tmp_path / "absent-overlay.yaml",
            environ={},
        ),
        CheckDelegationLocalModel(
            overlay_path=not_serving_overlay,
            environ={},
            transport=FakeTransport(unavailable=True),
        ),
        CheckDelegationLocalModel(
            overlay_path=mismatch_overlay,
            environ={},
            transport=FakeTransport(
                status=200,
                body=json.dumps({"data": [{"id": "some/other-model"}]}),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("case_index", "expected_fault"),
    [
        pytest.param(0, EnumDelegationDoctorFault.NO_IDENTITY, id="no-identity"),
        pytest.param(1, EnumDelegationDoctorFault.NO_KEY, id="no-key"),
        pytest.param(2, EnumDelegationDoctorFault.WRONG_KEY, id="wrong-key"),
        pytest.param(3, EnumDelegationDoctorFault.GATEWAY_DOWN, id="gateway-down"),
        pytest.param(4, EnumDelegationDoctorFault.QUOTA_EXHAUSTED, id="quota"),
        pytest.param(5, EnumDelegationDoctorFault.NO_LOCAL_MODEL, id="no-local"),
        pytest.param(
            6,
            EnumDelegationDoctorFault.LOCAL_MODEL_NOT_SERVING,
            id="local-not-serving",
        ),
        pytest.param(
            7,
            EnumDelegationDoctorFault.LOCAL_MODEL_ID_MISMATCH,
            id="local-id-mismatch",
        ),
    ],
)
def test_each_fault_is_named_with_one_distinct_fix(
    fault_checks: tuple[DiagnosticCheck, ...],
    tmp_path: Path,
    case_index: int,
    expected_fault: EnumDelegationDoctorFault,
) -> None:
    diagnoses = [check.diagnose() for check in fault_checks]
    diagnosis = diagnoses[case_index]

    assert diagnosis.fault is expected_fault
    assert diagnosis.fix.strip()
    assert {item.fault for item in diagnoses} == set(EnumDelegationDoctorFault)
    assert len({item.fix for item in diagnoses}) == 8
    if expected_fault is EnumDelegationDoctorFault.NO_LOCAL_MODEL:
        assert str(tmp_path / "absent-overlay.yaml") in diagnosis.fix
        assert "tier: local" in diagnosis.fix
        assert "endpoint_url" in diagnosis.fix
    if expected_fault is EnumDelegationDoctorFault.LOCAL_MODEL_ID_MISMATCH:
        assert _MODEL_ID in diagnosis.detail
        assert "some/other-model" in diagnosis.detail

    result = fault_checks[case_index].run()
    assert result.status is EnumHealthStatusValue.UNHEALTHY
    assert result.message.startswith(f"[{expected_fault.value}] ")
    assert f"Fix: {diagnosis.fix}" in result.message


def test_healthy_delegation_passes_all_five_checks(tmp_path: Path) -> None:
    home = tmp_path / "onex"
    overlay = tmp_path / "overlay.yaml"
    _save_gateway(home)
    _write_overlay(overlay)
    whoami = FakeTransport(body=_identity_body())
    gateway = FakeTransport()
    quota = FakeTransport()
    models = FakeTransport(body=json.dumps({"data": [{"id": _MODEL_ID}]}))
    checks: tuple[DiagnosticCheck, ...] = (
        CheckDelegationIdentity(onex_home=home),
        CheckDelegationKey(onex_home=home, transport=whoami),
        CheckDelegationGateway(onex_home=home, transport=gateway),
        CheckDelegationQuota(onex_home=home, transport=quota),
        CheckDelegationLocalModel(
            overlay_path=overlay,
            environ={},
            transport=models,
        ),
    )

    assert [check.diagnose().fault for check in checks] == [None] * 5
    assert all(check.run().status is EnumHealthStatusValue.HEALTHY for check in checks)
    assert whoami.urls == [f"{_BASE_URL}/v1/whoami"] * 2
    assert gateway.urls == [f"{_BASE_URL}/v1/whoami"] * 2
    assert quota.urls == [f"{_BASE_URL}/v1/whoami"] * 2
    assert models.urls == ["http://model.invalid/v1/models"] * 2


@pytest.mark.parametrize(
    ("transport", "dependency"),
    [
        pytest.param(FakeTransport(unavailable=True), "delegation_gateway"),
        pytest.param(FakeTransport(status=429), "delegation_quota"),
    ],
)
def test_key_does_not_false_pass_when_a_dependency_is_unknown(
    tmp_path: Path, transport: FakeTransport, dependency: str
) -> None:
    _save_gateway(tmp_path)
    check = CheckDelegationKey(onex_home=tmp_path, transport=transport)

    assert check.diagnose().fault is None
    result = check.run()
    assert result.status is EnumHealthStatusValue.UNKNOWN
    assert dependency in result.message


def test_no_doctor_message_leaks_the_api_key(tmp_path: Path) -> None:
    home = tmp_path / "onex"
    overlay = tmp_path / "overlay.yaml"
    _save_gateway(home)
    _write_overlay(overlay)
    checks: tuple[DiagnosticCheck, ...] = (
        CheckDelegationIdentity(onex_home=home),
        CheckDelegationKey(onex_home=home, transport=FakeTransport(status=401)),
        CheckDelegationGateway(onex_home=home, transport=FakeTransport(status=401)),
        CheckDelegationQuota(onex_home=home, transport=FakeTransport(status=401)),
        CheckDelegationLocalModel(
            overlay_path=overlay,
            environ={},
            transport=FakeTransport(body=json.dumps({"data": [{"id": _MODEL_ID}]})),
        ),
    )

    assert all(_API_KEY not in check.run().message for check in checks)


def test_empty_machine_is_total_and_reports_actionable_results(tmp_path: Path) -> None:
    missing_overlay = tmp_path / "missing-overlay.yaml"
    checks: tuple[DiagnosticCheck, ...] = (
        CheckDelegationIdentity(onex_home=tmp_path),
        CheckDelegationKey(onex_home=tmp_path),
        CheckDelegationGateway(onex_home=tmp_path),
        CheckDelegationQuota(onex_home=tmp_path),
        CheckDelegationLocalModel(overlay_path=missing_overlay, environ={}),
    )

    results = [check.run() for check in checks]
    diagnoses = [check.diagnose() for check in checks]

    assert len(results) == 5
    assert diagnoses[0].fault is EnumDelegationDoctorFault.NO_IDENTITY
    assert diagnoses[1].fault in {
        EnumDelegationDoctorFault.NO_KEY,
        None,
    }
    assert diagnoses[2].fault is None
    assert diagnoses[3].fault is None
    assert diagnoses[4].fault is EnumDelegationDoctorFault.NO_LOCAL_MODEL
    assert results[2].status is EnumHealthStatusValue.UNKNOWN
    assert results[3].status is EnumHealthStatusValue.UNKNOWN


def test_fault_enum_values_and_diagnosis_invariants() -> None:
    assert [fault.value for fault in EnumDelegationDoctorFault] == [
        "no_identity",
        "no_key",
        "wrong_key",
        "gateway_down",
        "quota_exhausted",
        "no_local_model",
        "local_model_not_serving",
        "local_model_id_mismatch",
    ]
    healthy = ModelDelegationDiagnosis(fault=None, detail="ready", fix="")
    with pytest.raises(ValidationError):
        healthy.detail = "changed"
    with pytest.raises(ValidationError):
        ModelDelegationDiagnosis(
            fault=EnumDelegationDoctorFault.NO_KEY,
            detail="missing key",
            fix="",
        )


def test_pyproject_registers_importable_doctor_checks() -> None:
    root = Path(__file__).parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    configured: Mapping[str, str] = project["project"]["entry-points"]["onex.doctor"]

    assert {name: configured.get(name) for name in _EXPECTED_ENTRY_POINTS} == (
        _EXPECTED_ENTRY_POINTS
    )
    for check_id, target in _EXPECTED_ENTRY_POINTS.items():
        module_name, _, attribute = target.partition(":")
        check_class = getattr(importlib.import_module(module_name), attribute)
        assert issubclass(check_class, DoctorCheckBase)
        assert check_class.check_id == check_id
        assert check_class.category is EnumDoctorCategory.SERVICES
        assert all(
            parameter.default is not inspect.Parameter.empty
            for parameter in inspect.signature(check_class).parameters.values()
        )


def test_refreshed_metadata_is_discoverable_by_core_registry() -> None:
    installed = {point.name for point in entry_points(group="onex.doctor")}
    if not _EXPECTED_ENTRY_POINTS.keys() <= installed:
        pytest.skip("editable package metadata has not been refreshed")

    registry = DoctorRegistry()
    registry.discover()

    discovered = {check.check_id for check in registry.list_all()}
    assert _EXPECTED_ENTRY_POINTS.keys() <= discovered


def test_overlay_is_bound_through_the_routing_authority_env_key(
    tmp_path: Path,
) -> None:
    """The doctor reads the overlay the router reads: BIFROST_OVERLAY_PATH."""
    overlay = tmp_path / "bound-overlay.yaml"
    overlay.write_text(
        "backends:\n"
        "  - backend_id: local-coder\n"
        "    tier: local\n"
        f"    endpoint_url: {_OVERLAY_ENDPOINT}\n"
        f"    model_name: {_MODEL_ID}\n"
    )
    bound = CheckDelegationLocalModel(
        environ={"BIFROST_OVERLAY_PATH": str(overlay)},
        transport=FakeTransport(
            status=200, body=json.dumps({"data": [{"id": _MODEL_ID}]})
        ),
    )
    assert bound.diagnose().fault is None


def test_embedding_backend_is_not_the_declared_chat_model(tmp_path: Path) -> None:
    """A local embedding backend never stands in for the chat model, and a
    chat backend without an overlay model_name is judged on reachability."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "backends:\n"
        "  - backend_id: local-embedding\n"
        "    tier: local\n"
        "    endpoint_url: http://embed.invalid/v1/embeddings\n"
        "  - backend_id: local-coder\n"
        "    tier: local\n"
        f"    endpoint_url: {_OVERLAY_ENDPOINT}\n"
    )
    check = CheckDelegationLocalModel(
        overlay_path=overlay,
        environ={},
        transport=FakeTransport(
            status=200, body=json.dumps({"data": [{"id": "anything"}]})
        ),
    )
    assert check.diagnose().fault is None

    embeddings_only = tmp_path / "embeddings-only.yaml"
    embeddings_only.write_text(
        "backends:\n"
        "  - backend_id: local-embedding\n"
        "    tier: local\n"
        "    endpoint_url: http://embed.invalid/v1/embeddings\n"
    )
    assert (
        CheckDelegationLocalModel(overlay_path=embeddings_only, environ={})
        .diagnose()
        .fault
        is EnumDelegationDoctorFault.NO_LOCAL_MODEL
    )
