# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pre-publish classification and refusal diagnostics without a runtime or bus."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, RootModel, field_validator

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli import delegate_pre_publish_failure as failure
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
)
from omnibase_infra.cli.model_delegate_run_addressing import ModelDelegateRunAddressing
from omnibase_infra.cli.model_delegate_transport_refusal import (
    ModelDelegateTransportRefusal,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

_RUN_ID = "11111111-1111-4111-8111-111111111111"
_CORRELATION_ID = "22222222-2222-4222-8222-222222222222"
_CONNECTION = "postgresql://fixture_user:fixture_password@db.example:5432/app"
_MODEL_PATH = "fixture_models.Request"


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int


class ModelNestedRequest(BaseModel):
    request: ModelRequest


class ModelSensitiveRequest(BaseModel):
    connection: str

    @field_validator("connection")
    @classmethod
    def refuse_connection(cls, value: str) -> str:
        raise ValueError(f"cannot connect to {value}")


def _envelope(**summary: object) -> dict[str, object]:
    return {
        "run_id": _RUN_ID,
        "correlation_id": _CORRELATION_ID,
        "duration_ms": 1250,
        "result_model": (
            "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
        ),
        "result": {
            "workflow_result": "failed",
            "wire_correlation_id": None,
            "runtime_error_is_transport": False,
            "terminal_payload": None,
            "handler_result": None,
            **summary,
        },
    }


@pytest.fixture
def payload(tmp_path: Path) -> Path:
    path = tmp_path / "payload.json"
    path.write_text('{"count": 1}', encoding="utf-8")
    return path


@pytest.fixture
def contract(tmp_path: Path) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(f"input_model: {_MODEL_PATH}\n", encoding="utf-8")
    return path


@pytest.fixture
def request_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        failure.importlib,
        "import_module",
        lambda name: SimpleNamespace(Request=ModelRequest),
    )


@pytest.mark.parametrize("result", [None, [], "failed", 1])
def test_non_summary_result_is_not_pre_publish(result: object) -> None:
    envelope = _envelope()
    envelope["result"] = result
    assert failure.pre_publish_failure_from_receipt(envelope) is None


@pytest.mark.parametrize("result_model", [None, "", "other.ModelResult"])
def test_wrong_result_model_is_not_pre_publish(result_model: object) -> None:
    envelope = _envelope()
    envelope["result_model"] = result_model
    assert failure.pre_publish_failure_from_receipt(envelope) is None


@pytest.mark.parametrize(
    "summary",
    [
        {"workflow_result": "completed"},
        {"wire_correlation_id": _CORRELATION_ID},
        {"runtime_error_is_transport": True},
        {"terminal_payload": {}},
        {"terminal_payload": False},
        {"handler_result": {}},
        {"handler_result": ""},
    ],
    ids=[
        "completed",
        "published",
        "transport",
        "terminal",
        "false-terminal",
        "handler",
        "empty-handler",
    ],
)
def test_other_failure_classes_are_excluded(summary: dict[str, object]) -> None:
    assert failure.pre_publish_failure_from_receipt(_envelope(**summary)) is None


@pytest.mark.parametrize("workflow_result", ["failed", "error", "timeout", "", None])
def test_pre_publish_returns_the_original_summary(workflow_result: object) -> None:
    envelope = _envelope(workflow_result=workflow_result)
    assert failure.pre_publish_failure_from_receipt(envelope) is envelope["result"]


def test_missing_optional_summary_fields_still_classify() -> None:
    envelope = _envelope()
    envelope["result"] = {}
    assert failure.pre_publish_failure_from_receipt(envelope) is envelope["result"]


@pytest.mark.parametrize("runtime_error_type", [None, "", "ValidationError"])
def test_base_error_names_run_and_type_without_echoing_error(
    runtime_error_type: object,
) -> None:
    message = failure.pre_publish_failure_error(
        _envelope(runtime_error_type=runtime_error_type, error=_CONNECTION)
    )
    assert _RUN_ID in message
    assert "failed before publish" in message
    assert "no resolvable delegation terminal" not in message
    assert ("(ValidationError)" in message) is bool(runtime_error_type)
    assert _CONNECTION not in message


def test_base_error_accepts_a_non_summary_result() -> None:
    assert "failed before publish" in failure.pre_publish_failure_error(
        {"result": None}
    )


def test_pre_publish_error_keeps_the_resolver_catch_contract() -> None:
    error = failure.DelegatePrePublishFailureError("before publish")
    assert isinstance(error, DelegateTerminalUnresolvedError)
    assert str(error) == "before publish"


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ("input_model: fixture_models.Request", _MODEL_PATH),
        ("input_model: {module: fixture_models, class: Request}", _MODEL_PATH),
        ("input_model: {module: fixture_models, name: Request}", _MODEL_PATH),
        (
            "input_model: {module: fixture_models, class: Request, name: Other}",
            _MODEL_PATH,
        ),
        (
            "input_model: {module: fixture_models, class: '', name: Request}",
            _MODEL_PATH,
        ),
        ("[]", None),
        ("null", None),
        ("{}", None),
        ("input_model: Request", None),
        ("input_model: 42", None),
        ("input_model: {module: '', class: Request}", None),
        ("input_model: {module: fixture_models, class: ''}", None),
        ("input_model: {module: 42, class: Request}", None),
        ("input_model: {module: fixture_models, class: 42}", None),
        ("input_model: [", None),
    ],
)
def test_contract_model_forms(
    tmp_path: Path, document: str, expected: str | None
) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text(document, encoding="utf-8")
    assert failure._request_model_path(path) == expected


def test_missing_contract_has_no_model(tmp_path: Path) -> None:
    assert failure._request_model_path(tmp_path / "absent.yaml") is None


@pytest.mark.parametrize("distributions", [None, [], ["missing"]])
def test_distribution_not_found(
    monkeypatch: pytest.MonkeyPatch, distributions: list[str] | None
) -> None:
    monkeypatch.setattr(
        failure.metadata,
        "packages_distributions",
        lambda: {"fixture_models": distributions},
    )

    def missing_version(name: str) -> str:
        raise failure.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(failure.metadata, "version", missing_version)
    assert (
        failure._distribution_of(_MODEL_PATH)
        == "fixture_models (distribution not found)"
    )


def test_distribution_skips_missing_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        failure.metadata,
        "packages_distributions",
        lambda: {"fixture_models": ["missing", "fixture-models"]},
    )

    def version(name: str) -> str:
        if name == "missing":
            raise failure.metadata.PackageNotFoundError(name)
        return "1.2.3"

    monkeypatch.setattr(failure.metadata, "version", version)
    assert failure._distribution_of(_MODEL_PATH) == "fixture-models 1.2.3"


@pytest.mark.parametrize("kind", ["import", "attribute"])
def test_unimportable_model(
    monkeypatch: pytest.MonkeyPatch, payload: Path, kind: str
) -> None:
    def import_model(name: str) -> SimpleNamespace:
        assert name == "fixture_models"
        if kind == "import":
            raise ImportError("fixture unavailable")
        return SimpleNamespace()

    monkeypatch.setattr(failure.importlib, "import_module", import_model)
    refusals, reason = failure._field_refusals(_MODEL_PATH, payload)
    assert refusals == []
    assert "the request model could not be imported" in reason


@pytest.mark.parametrize("model", [object, 42])
def test_non_pydantic_model(
    monkeypatch: pytest.MonkeyPatch, payload: Path, model: object
) -> None:
    monkeypatch.setattr(
        failure.importlib, "import_module", lambda name: SimpleNamespace(Request=model)
    )
    assert failure._field_refusals(_MODEL_PATH, payload) == (
        [],
        "the request model is not a pydantic model",
    )


@pytest.mark.parametrize("document", [None, "{invalid json"])
def test_unreadable_payload(
    request_model: None, payload: Path, document: str | None
) -> None:
    if document is None:
        payload.unlink()
    else:
        payload.write_text(document, encoding="utf-8")
    refusals, reason = failure._field_refusals(_MODEL_PATH, payload)
    assert refusals == []
    assert "the payload file could not be read" in reason


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ("{}", "the model refused no field this CLI supplied"),
        ('{"count": 1}', "the payload validates against the model"),
    ],
)
def test_no_field_refusal(
    request_model: None, payload: Path, document: str, expected: str
) -> None:
    payload.write_text(document, encoding="utf-8")
    assert failure._field_refusals(_MODEL_PATH, payload) == ([], expected)


def test_field_refusals_filter_missing_and_omit_input(
    request_model: None, payload: Path
) -> None:
    payload.write_text(
        json.dumps({"extra": _CONNECTION, "another": True}), encoding="utf-8"
    )
    refusals, reason = failure._field_refusals(_MODEL_PATH, payload)
    assert reason == ""
    assert refusals == [
        "`extra` extra_forbidden: Extra inputs are not permitted",
        "`another` extra_forbidden: Extra inputs are not permitted",
    ]
    assert _CONNECTION not in str(refusals)


@pytest.mark.parametrize(
    ("model", "raw", "location"),
    [
        (ModelNestedRequest, {"request": {"count": "bad"}}, "request.count"),
        (RootModel[int], "bad", "<root>"),
    ],
)
def test_nested_and_root_refusal_locations(
    monkeypatch: pytest.MonkeyPatch,
    payload: Path,
    model: type[BaseModel],
    raw: object,
    location: str,
) -> None:
    monkeypatch.setattr(
        failure.importlib, "import_module", lambda name: SimpleNamespace(Request=model)
    )
    payload.write_text(json.dumps(raw), encoding="utf-8")
    refusals, reason = failure._field_refusals(_MODEL_PATH, payload)
    assert reason == ""
    assert len(refusals) == 1
    assert refusals[0].startswith(f"`{location}` int_parsing:")


@pytest.mark.parametrize("document", ['{"count": 1}', '{"count": 1, "extra": true}'])
def test_description_names_evidence_and_does_not_guess(
    request_model: None,
    monkeypatch: pytest.MonkeyPatch,
    contract: Path,
    payload: Path,
    document: str,
) -> None:
    monkeypatch.setattr(
        failure, "_distribution_of", lambda path: "fixture-models 1.2.3"
    )
    payload.write_text(document, encoding="utf-8")
    message = failure.describe_pre_publish_failure(
        envelope=_envelope(),
        contract_path=contract,
        payload_path=payload,
        capture_log_path=payload.parent / "capture.log",
    )
    assert _MODEL_PATH in message
    assert "fixture-models 1.2.3" in message
    assert f"Payload: {payload.resolve()}" in message
    assert f"Capture log: {payload.parent / 'capture.log'}" in message
    if "extra" in document:
        assert "`extra` extra_forbidden" in message
        assert "not a field-level refusal" not in message
    else:
        assert "not a field-level refusal" in message
        assert "the payload validates against the model" in message


def test_description_without_contract(payload: Path) -> None:
    message = failure.describe_pre_publish_failure(
        envelope=_envelope(),
        contract_path=payload.parent / "absent.yaml",
        payload_path=payload,
        capture_log_path=payload.parent / "capture.log",
    )
    assert "The request model could not be read" in message
    assert "Cause:" not in message
    assert "capture.log" in message


@pytest.mark.parametrize("source", ["validation", "import", "read"])
def test_diagnostic_errors_never_expose_credentials(
    monkeypatch: pytest.MonkeyPatch, contract: Path, payload: Path, source: str
) -> None:
    def import_model(name: str) -> SimpleNamespace:
        if source == "import":
            raise ImportError(f"unavailable at {_CONNECTION}")
        return SimpleNamespace(Request=ModelSensitiveRequest)

    monkeypatch.setattr(failure.importlib, "import_module", import_model)
    monkeypatch.setattr(
        failure, "_distribution_of", lambda path: "fixture-models 1.2.3"
    )
    payload.write_text(json.dumps({"connection": _CONNECTION}), encoding="utf-8")
    if source == "read":
        read_text = Path.read_text

        def read(path: Path, *args: object, **kwargs: object) -> str:
            if path == payload:
                raise OSError(f"unavailable at {_CONNECTION}")
            return read_text(path, encoding="utf-8")

        monkeypatch.setattr(Path, "read_text", read)
    message = failure.describe_pre_publish_failure(
        envelope=_envelope(),
        contract_path=contract,
        payload_path=payload,
        capture_log_path=payload.parent / "capture.log",
    )
    assert _CONNECTION not in message
    assert "fixture_password" not in message
    assert "[REDACTED" in message
    assert _MODEL_PATH in message
    assert "capture.log" in message


@pytest.mark.parametrize("correlation_id", [_CORRELATION_ID, UUID(_CORRELATION_ID)])
def test_typed_transport_refusal_and_written_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, correlation_id: str | UUID
) -> None:
    monkeypatch.setattr(cli_delegate, "_resolve_transport_bound", lambda: (3, 12.5))
    addressing = ModelDelegateRunAddressing(
        locus=EnumDelegateLocus.DEPLOYED_LANE, bus="kafka", lane="dev"
    )
    envelope = _envelope(
        runtime_error_is_transport=True,
        runtime_error_type="InfraConnectionError",
        error=f"connection failed: {_CONNECTION}",
    )
    envelope["correlation_id"] = correlation_id
    assert failure.pre_publish_failure_from_receipt(envelope) is None
    refusal = cli_delegate._transport_refusal_from_receipt(
        envelope=envelope,
        addressing=addressing,
        broker="db.example:5432",
        command_topic="fixture-topic",
    )
    assert isinstance(refusal, ModelDelegateTransportRefusal)
    assert refusal.model_dump(mode="json") == {
        "awaited": "broker_connection",
        "reason": "broker_unreachable",
        "correlation_id": _CORRELATION_ID,
        "bus": "kafka",
        "locus": "deployed-lane",
        "broker": "db.example:5432",
        "command_topic": "fixture-topic",
        "attempts_permitted": 3,
        "bound_seconds": 12.5,
        "elapsed_seconds": 1.25,
        "transport_error_type": "InfraConnectionError",
        "transport_error": "[REDACTED - potentially sensitive data]",
        "remediation": "",
    }
    cli_delegate._write_transport_refusal_run_files(
        refusal=refusal,
        run_id=_RUN_ID,
        state_root=tmp_path,
        prompt="fixture",
        task_type="document",
        task_type_resolution="explicit",
        addressing=addressing,
    )
    run_dir = tmp_path / "runs" / _RUN_ID
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert (
        receipt["receipt_id"]
        == receipt["correlation_id"]
        == run["correlation_id"]
        == _CORRELATION_ID
    )
    assert receipt["run_id"] == run["run_id"] == _RUN_ID
    assert receipt["status"] == "failed"
    assert receipt["terminal_class"] == "transport"
    assert receipt["terminal_failure_cause"] is None
    assert receipt["attempts"] == []
    assert receipt["route_attributed"] is False
    assert receipt["transport_refusal"] == refusal.model_dump(mode="json")
    assert (run_dir / "result.txt").read_text(encoding="utf-8") == ""
    for path in run_dir.iterdir():
        assert "fixture_password" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("correlation_id", [None, "not-a-uuid"])
def test_transport_refusal_rejects_invalid_correlation(
    monkeypatch: pytest.MonkeyPatch, correlation_id: object
) -> None:
    monkeypatch.setattr(cli_delegate, "_resolve_transport_bound", lambda: (3, 12.5))
    envelope = _envelope(
        runtime_error_is_transport=True, runtime_error_type="InfraConnectionError"
    )
    envelope["correlation_id"] = correlation_id
    with pytest.raises(ValueError):
        cli_delegate._transport_refusal_from_receipt(
            envelope=envelope,
            addressing=ModelDelegateRunAddressing(
                locus=EnumDelegateLocus.DEPLOYED_LANE, bus="kafka"
            ),
        )
