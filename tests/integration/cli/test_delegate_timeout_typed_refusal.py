# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex delegate`` reports a delegation that never terminalized (OMN-17516).

These are end-to-end tests of the real command: the real click entry point, the
real payload write, the real ``run_receipt_mode``, the real ``RuntimeLocal``,
and the real in-memory bus. Only the packaged contract is redirected, to one
whose handler blocks — which is the condition under test and cannot be produced
any other way without replacing a layer whose behaviour is the subject.

They live under ``tests/integration`` rather than ``tests/unit`` for that
reason: nothing here is isolated from the dispatch stack, and the defect they
pin has now been closed twice on handler-isolated evidence (OMN-13601,
OMN-11318) and returned both times. A test that stubbed ``run_receipt_mode``
would have passed throughout (``feedback_real_dispatch_path_tests``).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from tests.helpers.cli_registry_stand_in import (
    use_stand_in_registry,
    wiring_authority,
)


@pytest.fixture(autouse=True)
def _no_ambient_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the run off the launching host's workspace and drift guard.

    ``OMNI_HOME`` binds the omnimarket co-install drift check, which is a real
    refusal about the machine rather than about the code under test; leaving it
    bound would make these tests report the host's venv state.
    """
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)


# A contract whose handler never returns inside any bound a test declares
# (OMN-17516). Delegate-shaped like ``_CORRELATED_NOOP_CONTRACT`` -- its request
# model round-trips ``correlation_id`` -- so the receipt the CLI has to produce
# on the timeout path is the real delegate receipt shape, not a stand-in.
_BLOCKING_NOOP_CONTRACT = (
    "---\n"
    "name: blocking_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.blocking-noop-completed.v1\n"
    "handler:\n"
    "  module: tests.fixtures.handler_blocking_noop\n"
    "  class: HandlerBlockingNoop\n"
    "  input_model: tests.fixtures.handler_blocking_noop"
    ".ModelBlockingNoopRequest\n"
    "handler_routing:\n"
    "  default_handler: tests.fixtures.handler_blocking_noop"
    ":HandlerBlockingNoop\n"
)


class TestDelegateTimeoutIsTypedOnStdout:
    """A terminal that never arrives surfaces a TYPED refusal (OMN-17516 AC4).

    ``onex delegate``'s documented contract is that stdout carries exactly ONE
    ``ModelSkillResult`` JSON. Every terminal outcome honours it -- a completed
    run, a failed rung, an unattributed run, a run whose artifact write was
    refused (the test directly above). The hard-timeout backstop did not: it
    printed prose to stderr, returned 1, and left stdout EMPTY, so a caller
    that parses stdout received nothing at all and could not tell a delegation
    that will never answer from one still working.

    That empty-stdout shape is the whole of the OMN-17516 report. The
    2026-08-26 observation -- "hung past 120 seconds and returned no result at
    all. Not a wrong result, not a typed timeout surfaced to the caller: no
    terminal" -- describes this path, not a lost event: ``--timeout`` bounds
    only ``RuntimeLocal``'s wait on the terminal event, which is reached only
    after the entry handler RETURNS, and on the in-process locus the delegate
    orchestrator does the entire delegation inside that handler call.

    These tests drive the REAL command against a real contract with a real
    blocking handler, over the real in-memory bus, so what is bounded is the
    same layer the customer's run bounds (``feedback_real_dispatch_path_tests``).
    """

    @pytest.fixture
    def blocking_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(_BLOCKING_NOOP_CONTRACT, encoding="utf-8")
        monkeypatch.setattr(
            cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
        )
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        use_stand_in_registry(
            monkeypatch, wiring_authority(terminal_delivery_margin_seconds=1)
        )
        # Shrink the grace window so the bound under test is seconds, not the
        # production ten. The fixture also declares a one-second delivery
        # margin, so the complete execution + margin + grace bound stays
        # short while preserving the production budget calculation.
        monkeypatch.setattr(cli_delegate, "_HARD_TIMEOUT_GRACE_SECONDS", 1)
        return contract_path

    def _invoke(self, tmp_path: Path) -> tuple[object, float]:
        started = time.monotonic()
        result = CliRunner().invoke(
            delegate_command,
            [
                "List the first five prime numbers",
                "--state-root",
                str(tmp_path / "state"),
                "--timeout",
                "1",
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
            ],
            catch_exceptions=False,
        )
        return result, time.monotonic() - started

    @pytest.mark.usefixtures("blocking_contract")
    def test_timeout_still_writes_one_typed_result_to_stdout(
        self, tmp_path: Path
    ) -> None:
        """The backstop trip is a receipt, not a bare exit code."""
        result, elapsed = self._invoke(tmp_path)

        assert result.exit_code == 1, result.stderr
        assert 3 <= elapsed < 10, (
            "the one-second execution window, one-second declared delivery margin, "
            f"and one-second hard-timeout grace did not bound the call: {elapsed}s"
        )
        assert result.stdout.strip(), (
            "timeout produced NO typed result on stdout -- a caller parsing the "
            "documented single-ModelSkillResult contract receives nothing, which "
            "is indistinguishable from a run that is still working"
        )
        receipt = ModelSkillResult.model_validate(json.loads(result.stdout.strip()))
        assert receipt.status is EnumSkillResultStatus.FAILED
        assert receipt.exit_code == 1

    @pytest.mark.usefixtures("blocking_contract")
    def test_timeout_receipt_names_what_was_waited_on_and_for_how_long(
        self, tmp_path: Path
    ) -> None:
        """AC4's second half: the refusal is legible, not merely typed.

        A receipt that says only "failed" is a typed shrug. The caller has to
        be able to read, off the receipt alone, that the thing not delivered
        was the delegation terminal, which transport and locus it was awaited
        on, and what bound was exceeded -- otherwise the next reader repeats
        the 2026-08-26 diagnosis from scratch.
        """
        result, _elapsed = self._invoke(tmp_path)

        payload = json.loads(result.stdout.strip())
        refusal = payload["result"]

        assert refusal["awaited"] == "delegation_terminal"
        # The receipt reports the wait it actually served: one requested
        # execution second plus this fixture's one-second delivery margin.
        assert refusal["declared_timeout_seconds"] == 2
        assert refusal["grace_seconds"] == 1
        assert refusal["elapsed_seconds"] >= 2
        assert refusal["bus"] == "inmemory"
        assert refusal["locus"] == "in-process"
        assert refusal["terminal_topic"] == (
            "onex.evt.proof.blocking-noop-completed.v1"
        )
        assert refusal["reason"] == "hard_timeout_backstop"

    @pytest.mark.usefixtures("blocking_contract")
    def test_timeout_receipt_carries_this_runs_own_correlation_id(
        self, tmp_path: Path
    ) -> None:
        """The refusal is attributable to the run that produced it.

        OMN-17295's whole finding was a receipt read as another run's. A
        timeout receipt with no correlation id, or with one the CLI did not
        mint for this invocation, is the same defect in the failure direction:
        two concurrent hung runs would be indistinguishable in a log.
        """
        result, _elapsed = self._invoke(tmp_path)

        receipt = json.loads(result.stdout.strip())
        minted = uuid.UUID(receipt["correlation_id"])

        # The id on the receipt is the one this invocation actually dispatched
        # under, proven against the request payload it wrote to disk rather
        # than against itself.
        scratch = sorted((tmp_path / "state" / "tmp").glob("delegate-input-*.json"))
        assert len(scratch) == 1, f"expected one request payload, got {scratch}"
        dispatched = json.loads(scratch[0].read_text(encoding="utf-8"))
        assert dispatched["correlation_id"] == str(minted)

        # ...and the refusal inside carries it too, so it stays attributable
        # when it is read apart from its envelope.
        assert receipt["result"]["correlation_id"] == str(minted)
        assert uuid.UUID(receipt["run_id"]) != minted

    @pytest.mark.usefixtures("blocking_contract")
    def test_timeout_still_says_it_on_stderr(self, tmp_path: Path) -> None:
        """The pre-existing human-facing message is kept, not traded away.

        The typed receipt is for the caller that parses; the stderr line is for
        the person watching. OMN-14397 added the latter and a test pins it;
        this asserts the new receipt did not replace it.
        """
        result, _elapsed = self._invoke(tmp_path)

        assert "exceeded hard timeout" in result.stderr
