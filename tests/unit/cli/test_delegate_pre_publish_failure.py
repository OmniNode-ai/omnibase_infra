# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pre-publish classification and actionable, payload-safe diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.cli import delegate_pre_publish_failure as failure


class ModelRequiredRequest(BaseModel):
    """A runtime-supplied identity is required, and CLI extras are refused."""

    model_config = ConfigDict(extra="forbid")
    correlation_id: str


class ModelNestedRequest(BaseModel):
    """Exercise a refusal location containing both a field and an index."""

    values: list[int]


_MODEL_PATH = f"{__name__}.ModelRequiredRequest"
_PAYLOAD_SENTINEL = "synthetic-private-payload-value"
_RUNTIME_SENTINEL = "synthetic-private-runtime-detail"


def _receipt(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "workflow_result": "failed",
        "runtime_error_type": "ValidationError",
        "error": _RUNTIME_SENTINEL,
    }
    result.update(overrides)
    return {
        "run_id": "test-run",
        "result_model": "omnibase_infra.cli.ModelReceiptRuntimeSummary",
        "result": result,
    }


@pytest.fixture(autouse=True)
def isolated_distributions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the diagnostic independent of installed packages on the host."""
    monkeypatch.setattr(failure.metadata, "packages_distributions", dict)


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    contract = tmp_path / "contract.yaml"
    contract.write_text(f"input_model: {_MODEL_PATH}\n", encoding="utf-8")
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"correlation_id": _PAYLOAD_SENTINEL}), encoding="utf-8"
    )
    return contract, payload, tmp_path / "capture.log"


def _describe(paths: tuple[Path, Path, Path]) -> str:
    contract, payload, capture = paths
    envelope = _receipt()
    assert failure.pre_publish_failure_from_receipt(envelope) is envelope["result"]
    message = failure.describe_pre_publish_failure(
        envelope=envelope,
        contract_path=contract,
        payload_path=payload,
        capture_log_path=capture,
    )
    assert "delegate run test-run failed before publish (ValidationError):" in message
    assert "no command reached the broker" in message
    assert "The deployed lane is not implicated." in message
    assert f"Payload: {payload.resolve()}. Capture log: {capture}." in message
    assert _PAYLOAD_SENTINEL not in message
    assert _RUNTIME_SENTINEL not in message
    return message


@pytest.mark.parametrize("result", [None, [], "failed", 0])
def test_non_summary_results_are_not_pre_publish(result: object) -> None:
    envelope = _receipt()
    envelope["result"] = result
    assert failure.pre_publish_failure_from_receipt(envelope) is None


@pytest.mark.parametrize("result_model", [None, "", "ModelOtherResult"])
def test_other_result_models_are_not_pre_publish(result_model: object) -> None:
    envelope = _receipt()
    envelope["result_model"] = result_model
    assert failure.pre_publish_failure_from_receipt(envelope) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"workflow_result": "completed"},
        {"wire_correlation_id": "published-command"},
        {"runtime_error_is_transport": True},
        {"terminal_payload": {}},
        {"handler_result": {}},
        {"terminal_payload": False},
    ],
    ids=[
        "completed",
        "published",
        "transport",
        "terminal",
        "handler",
        "false-terminal",
    ],
)
def test_receipts_with_completion_or_publish_evidence_are_excluded(
    overrides: dict[str, object],
) -> None:
    assert failure.pre_publish_failure_from_receipt(_receipt(**overrides)) is None


@pytest.mark.parametrize("workflow_result", ["failed", "error", "timeout", "", None])
def test_incomplete_runs_without_publish_evidence_are_classified(
    workflow_result: object,
) -> None:
    envelope = _receipt(
        workflow_result=workflow_result, terminal_payload=None, handler_result=None
    )
    assert failure.pre_publish_failure_from_receipt(envelope) is envelope["result"]
    message = failure.pre_publish_failure_error(envelope)
    assert "failed before publish (ValidationError)" in message
    assert "there is no delegation terminal to look for" in message
    assert _RUNTIME_SENTINEL not in message


@pytest.mark.parametrize("result", [None, {}, {"runtime_error_type": None}])
def test_base_message_without_error_type_does_not_invent_a_cause(
    result: object,
) -> None:
    message = failure.pre_publish_failure_error(
        {"run_id": "test-run", "result": result}
    )
    assert message.startswith("delegate run test-run failed before publish:")
    assert "The deployed lane is not implicated." in message


@pytest.mark.parametrize(
    "contract_text",
    [
        "input_model: [",
        "- not-a-mapping\n",
        "",
        "input_model: UndottedModel\n",
        "input_model: 17\n",
        "input_model: {module: '', class: ModelRequiredRequest}\n",
        "input_model: {module: tests, class: ''}\n",
        "input_model: {module: 17, class: ModelRequiredRequest}\n",
        "input_model: {module: tests, class: 17}\n",
    ],
)
def test_unusable_contract_does_not_guess_a_refusing_field(
    paths: tuple[Path, Path, Path],
    contract_text: str,
) -> None:
    paths[0].write_text(contract_text, encoding="utf-8")
    message = _describe(paths)
    assert f"The request model could not be read from {paths[0]}" in message
    assert "the refusing field is not named here" in message
    assert "Cause:" not in message


def test_missing_contract_points_to_the_capture_log(
    paths: tuple[Path, Path, Path],
) -> None:
    paths[0].unlink()
    assert "The request model could not be read" in _describe(paths)


@pytest.mark.parametrize("model_key", ["class", "name"])
def test_mapping_contract_resolves_the_model_and_reports_valid_payload(
    paths: tuple[Path, Path, Path],
    model_key: str,
) -> None:
    paths[0].write_text(
        f"input_model:\n  module: {__name__}\n  {model_key}: ModelRequiredRequest\n",
        encoding="utf-8",
    )
    message = _describe(paths)
    assert f"not a field-level refusal by {_MODEL_PATH}" in message
    assert "the payload validates against the model" in message
    assert "The runtime's own error is in the capture log." in message


@pytest.mark.parametrize(
    ("model_path", "expected"),
    [
        ("missing_pre_publish_test_module.Request", "could not be imported"),
        (f"{__name__}.MissingRequest", "could not be imported"),
        ("builtins.str", "is not a pydantic model"),
        ("builtins.len", "is not a pydantic model"),
    ],
)
def test_unavailable_model_reports_why_the_field_is_unknown(
    paths: tuple[Path, Path, Path],
    model_path: str,
    expected: str,
) -> None:
    paths[0].write_text(f"input_model: {model_path}\n", encoding="utf-8")
    message = _describe(paths)
    assert f"not a field-level refusal by {model_path}" in message
    assert f"the request model {expected}" in message
    assert "The runtime's own error is in the capture log." in message


@pytest.mark.parametrize(
    "payload_text", [None, '{"private": "' + _PAYLOAD_SENTINEL + '"']
)
def test_unreadable_payload_reports_the_read_failure(
    paths: tuple[Path, Path, Path],
    payload_text: str | None,
) -> None:
    if payload_text is None:
        paths[1].unlink()
    else:
        paths[1].write_text(payload_text, encoding="utf-8")
    message = _describe(paths)
    assert "not a field-level refusal" in message
    assert "the payload file could not be read" in message
    assert "The runtime's own error is in the capture log." in message


def test_runtime_injected_missing_fields_are_not_reported_as_cli_refusals(
    paths: tuple[Path, Path, Path],
) -> None:
    paths[1].write_text("{}", encoding="utf-8")
    message = _describe(paths)
    assert "not a field-level refusal" in message
    assert "the model refused no field this CLI supplied" in message
    assert "`correlation_id` missing" not in message


def test_extra_field_is_named_without_its_private_value_or_missing_defaults(
    paths: tuple[Path, Path, Path],
) -> None:
    paths[1].write_text(
        json.dumps({"requested_timeout_seconds": _PAYLOAD_SENTINEL}), encoding="utf-8"
    )
    message = _describe(paths)
    assert f"Cause: the request payload was refused by {_MODEL_PATH}" in message
    assert (
        "`requested_timeout_seconds` extra_forbidden: Extra inputs are not permitted"
        in message
    )
    assert "`correlation_id` missing" not in message


def test_nested_refusal_location_names_the_index_without_private_input(
    paths: tuple[Path, Path, Path],
) -> None:
    model_path = f"{__name__}.ModelNestedRequest"
    paths[0].write_text(f"input_model: {model_path}\n", encoding="utf-8")
    paths[1].write_text(json.dumps({"values": [_PAYLOAD_SENTINEL]}), encoding="utf-8")
    message = _describe(paths)
    assert f"Cause: the request payload was refused by {model_path}" in message
    assert "`values.0` int_parsing: Input should be a valid integer" in message


def test_non_mapping_payload_reports_the_root_without_private_input(
    paths: tuple[Path, Path, Path],
) -> None:
    paths[1].write_text(json.dumps(_PAYLOAD_SENTINEL), encoding="utf-8")
    message = _describe(paths)
    assert "`<root>` model_type: Input should be a valid dictionary" in message


@pytest.mark.parametrize(
    ("distributions", "versions", "provider"),
    [
        ([], {}, "tests (distribution not found)"),
        (["absent-test-dist"], {}, "tests (distribution not found)"),
        (
            ["test-request-dist"],
            {"test-request-dist": "1.2.3"},
            "test-request-dist 1.2.3",
        ),
        (
            ["absent-test-dist", "test-request-dist"],
            {"test-request-dist": "1.2.3"},
            "test-request-dist 1.2.3",
        ),
    ],
    ids=["no-provider", "stale-provider", "installed-provider", "stale-then-installed"],
)
def test_diagnostic_names_the_available_distribution_or_an_explicit_fallback(
    paths: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    distributions: list[str],
    versions: dict[str, str],
    provider: str,
) -> None:
    monkeypatch.setattr(
        failure.metadata, "packages_distributions", lambda: {"tests": distributions}
    )

    def version(name: str) -> str:
        if name not in versions:
            raise failure.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(failure.metadata, "version", version)
    message = _describe(paths)
    assert f"{_MODEL_PATH} ({provider})" in message
