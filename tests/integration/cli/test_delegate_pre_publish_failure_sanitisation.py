# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A pre-publish refusal never echoes a credential to the operator (OMN-20425).

OMN-19131 AC4 makes the operator-visible error name the offending field and
the rejecting model's import path. The refusal message itself is the model's
own text, and a model validator is free to echo the value it rejected, which
for a prompt can be a credential the operator pasted. The message reaches two
places the operator and the caller's logs keep: stderr and the typed receipt
on stdout.

These tests force a pre-publish validation failure through the real
``delegate_command``, the real ``run_receipt_mode`` and the real
``RuntimeLocal``, against a stand-in request model whose validator puts a
credential-shaped value in its refusal message. They assert that:

* the credential is absent from stderr and from the receipt's ``error``, the
  field the CLI writes for the refusal on stdout;
* the offending field name and the rejecting model's import path are still
  named (OMN-19131 AC4), so redaction does not blind the reader;
* a refusal message with no credential shape keeps its text, so the redaction
  is not simply deleting every message (positive control).

``sanitize_error_string`` redacts by substring match on a fixed pattern list,
so the credential shapes below each carry a pattern the list holds: a bearer
header, a password assignment and an ``api_key`` assignment. The shapes the
artifact store's own secret detector refuses (``sk-`` followed by twenty
characters, ``ghp_`` and the like) are left out on purpose: that detector
aborts the command at the receipt write, a different layer from the one under
test.

The receipt carries a second field these tests do not assert on:
``result.capture_log`` holds the runtime's own raw log of the same refusal,
which this module's source change does not touch. The assertions are scoped to
``error`` for that reason, not because ``capture_log`` is clean.

Not driven here: the payload-read refusal path. The CLI writes the payload
file itself immediately before the run, so it cannot be made unreadable
through the command without replacing the code under test. The unit file
(``tests/unit/cli/test_delegate_pre_publish_failure.py``) covers the validation,
import and payload-read paths with a credential-bearing message each.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.delegate_caller import CALLER_LANE_ENV_VARS
from omnibase_infra.cli.model_receipt_runtime_summary import (
    ModelReceiptRuntimeSummary,
)
from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
    ModelDelegateSkillFixtureTerminal,
)

pytestmark = pytest.mark.integration

#: Fabricated filler, never a real credential. Long and distinctive so an
#: absence assertion cannot pass by coincidence.
_FILLER = "Zq7Lm2Xv9Rt4Kp8Wd3Hn6Bs1Yc5"

#: Each shape carries a substring ``sanitize_error_string`` redacts on.
_CREDENTIAL_DETAILS: dict[str, str] = {
    "bearer-header": f"upstream said Authorization: Bearer {_FILLER}",
    "password-assignment": f"upstream said password={_FILLER}",
    "api-key-assignment": f"upstream said api_key={_FILLER}",
}

#: Carries none of the sanitizer's patterns, so the message must survive.
_PLAIN_DETAIL = "upstream said reference 4521 is closed"

_PROMPT = "Reply with exactly the word READY"

#: What the stand-in validator appends to its refusal. A test sets it; the
#: prompt itself stays plain so the credential travels only in the message.
_INTAKE_DETAIL: dict[str, str] = {"text": ""}

_UNIMPORTABLE_MODULE = "omn20425_unimportable_request_models"


class ModelEchoingRequest(BaseModel):
    """Stand-in request model whose validator reports an upstream detail.

    Forbids extras and declares the same fields the command always sends
    (``metadata`` for the caller lane, ``tenant_id`` for the tenant stamp), so
    the one refusal is the prompt validator and not an unrelated extra field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: str = ""
    prompt: str = ""
    task_type: str = ""
    source: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    tenant_id: str | None = None

    @field_validator("prompt")
    @classmethod
    def _refuse_prompt(cls, value: str) -> str:
        raise ValueError(
            f"prompt was refused by the intake policy ({_INTAKE_DETAIL['text']})"
        )


class HandlerEchoingNoop:
    """The correlated no-op handler, fed from the echoing stand-in model."""

    def __init__(self) -> None:
        self._noop = HandlerCorrelatedNoop()

    def handle(self, request: ModelEchoingRequest) -> ModelDelegateSkillFixtureTerminal:
        return self._noop.handle(
            ModelCorrelatedNoopRequest(
                correlation_id=request.correlation_id,
                prompt=request.prompt,
                task_type=request.task_type,
            )
        )


_MODEL_IMPORT_PATH = f"{__name__}.ModelEchoingRequest"


def _contract(input_model: str) -> str:
    return (
        "---\n"
        "name: correlated_noop\n"
        "node_type: compute\n"
        "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
        f"input_model: {input_model}\n"
        "handler:\n"
        f"  module: {__name__}\n"
        "  class: HandlerEchoingNoop\n"
        f"  input_model: {input_model}\n"
        "handler_routing:\n"
        f"  default_handler: {__name__}:HandlerEchoingNoop\n"
    )


def _install_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_model: str
) -> Path:
    """Resolve a stand-in contract and keep the host's workspace out of it."""
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    # Named for the delegate node, as the packaged contract is: the receipt
    # writer scopes itself to delegate runs by the workflow path.
    contract_path = tmp_path / cli_delegate.DELEGATE_NODE_NAME / "contract.yaml"
    contract_path.parent.mkdir()
    contract_path.write_text(_contract(input_model), encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    return contract_path


@pytest.fixture
def echoing_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _install_contract(tmp_path, monkeypatch, _MODEL_IMPORT_PATH)


#: The caller attribution the command derives from its environment and working
#: directory is cleared, so a run started from a ticket worktree or a Claude
#: Code session is not refused by the stand-in model for a reason that is not
#: the one under test.
_CALLER_ENV_CLEARED: dict[str, str | None] = dict.fromkeys(
    (*CALLER_LANE_ENV_VARS, "CLAUDE_CODE_SESSION_ID", "ONEX_LANE_REGISTRY_ROOT")
)


def _invoke(tmp_path: Path) -> Result:
    runner = CliRunner(env=_CALLER_ENV_CLEARED)
    with runner.isolated_filesystem(temp_dir=tmp_path):
        return runner.invoke(
            delegate_command,
            [
                _PROMPT,
                "--json",
                "--task-type",
                "summarization",
                "--bus",
                "inmemory",
                "--locus",
                "in-process",
                "--state-root",
                str(tmp_path / "state"),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )


def _receipt_error(result: Result) -> str:
    receipt = ModelSkillResult[ModelReceiptRuntimeSummary].model_validate(
        json.loads(result.stdout.strip())
    )
    assert receipt.status is EnumSkillResultStatus.FAILED
    assert receipt.result.wire_correlation_id is None, "the run must not have published"
    return receipt.result.error


@pytest.mark.usefixtures("echoing_contract")
class TestValidationRefusalCarriesNoCredential:
    """The refusal message is redacted on both operator-visible surfaces."""

    @pytest.mark.parametrize("shape", sorted(_CREDENTIAL_DETAILS))
    def test_credential_is_absent_from_stderr_and_receipt_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
    ) -> None:
        monkeypatch.setitem(_INTAKE_DETAIL, "text", _CREDENTIAL_DETAILS[shape])

        result = _invoke(tmp_path)

        assert result.exit_code != 0, "a refused payload is not a successful run"
        error = _receipt_error(result)
        assert _FILLER not in result.stderr
        assert _FILLER not in error
        assert "before publish" in result.stderr
        assert "[REDACTED" in result.stderr
        assert "[REDACTED" in error

    @pytest.mark.parametrize("shape", sorted(_CREDENTIAL_DETAILS))
    def test_field_and_model_are_still_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
    ) -> None:
        """OMN-19131 AC4: redacting the message does not blind the reader."""
        monkeypatch.setitem(_INTAKE_DETAIL, "text", _CREDENTIAL_DETAILS[shape])

        result = _invoke(tmp_path)
        error = _receipt_error(result)

        for surface in (result.stderr, error):
            assert "`prompt` value_error" in surface
            assert _MODEL_IMPORT_PATH in surface

    def test_positive_control_plain_message_keeps_its_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without this the redaction could be deleting every message."""
        monkeypatch.setitem(_INTAKE_DETAIL, "text", _PLAIN_DETAIL)

        result = _invoke(tmp_path)

        assert result.exit_code != 0, "a refused payload is not a successful run"
        error = _receipt_error(result)
        expected = f"prompt was refused by the intake policy ({_PLAIN_DETAIL})"
        assert expected in result.stderr
        assert expected in error
        assert "[REDACTED" not in result.stderr
        assert "[REDACTED" not in error
        assert "`prompt` value_error" in error
        assert _MODEL_IMPORT_PATH in error


class TestImportRefusalCarriesNoCredential:
    """The request model itself fails to import with a credential in its error."""

    @pytest.fixture
    def unimportable_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        package = tmp_path / "models"
        package.mkdir()
        (package / f"{_UNIMPORTABLE_MODULE}.py").write_text(
            "raise ImportError("
            f'"cannot load the request model, retry with api_key={_FILLER}"'
            ")\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(package))
        monkeypatch.delitem(sys.modules, _UNIMPORTABLE_MODULE, raising=False)
        return _install_contract(
            tmp_path, monkeypatch, f"{_UNIMPORTABLE_MODULE}.ModelUnimportable"
        )

    def test_credential_is_absent_from_stderr_and_receipt_error(
        self, tmp_path: Path, unimportable_contract: Path
    ) -> None:
        result = _invoke(tmp_path)

        assert result.exit_code != 0
        error = _receipt_error(result)
        assert _FILLER not in result.stderr
        assert _FILLER not in error
        assert "could not be imported" in error
        assert "[REDACTED" in error
        assert f"{_UNIMPORTABLE_MODULE}.ModelUnimportable" in error
