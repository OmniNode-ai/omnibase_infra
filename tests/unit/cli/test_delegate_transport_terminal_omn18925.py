# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A transport-class dispatch failure writes a typed terminal (OMN-18925, C16).

Built from the 2026-09-21 incident rather than from an imagined one. The dev
Redpanda stalled its shard-0 reactor for 17.2 seconds under host CPU
starvation; two delegations landed inside that window; neither produced a run
directory, a ``receipt.json``, a ``run.json`` or a ``result.txt``. Every lane
in this workspace is instructed to read the terminal from
``.onex_state/runs/<run_id>/receipt.json`` rather than from an exit code, so on
that path the instruction named a file that did not exist.

The four acceptance criteria map onto the classes below:

* AC-1 -> :class:`TestTransportFailureWritesTypedTerminal`
* AC-2 -> :class:`TestConnectRetriesAcrossAStall` (and the event-bus/provider
  parity test in ``tests/unit/event_bus/test_kafka_connect_retry_omn18925.py``)
* AC-3 -> :class:`TestBothSurfacesResolveOneRetryDeclaration`
* AC-4 -> :class:`TestUnreachableBrokerFailsFastAndTyped`
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import _write_local_run_files
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
)
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.cli.model_delegate_transport_refusal import (
    ModelDelegateTransportRefusal,
)
from omnibase_infra.cli.model_receipt_runtime_summary import (
    ModelReceiptRuntimeSummary,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.runtime_identity import collect_runtime_identity

_ADDRESSING = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.DEPLOYED_LANE,
    bus="kafka",
    lane="dev",
    dispatch_target=(
        "onex.cmd.omnimarket.delegate-skill.v1 via omninode-pc.tail75df5e.ts.net:19092"
    ),
)

_CONTRACT = (
    "/site-packages/omnimarket/nodes/node_delegate_skill_orchestrator/contract.yaml"
)


def _receipt(
    *,
    is_transport: bool,
    error_type: str = "InfraConnectionError",
    error: str = "Failed to connect to Kafka: KafkaConnectionError",
    duration_ms: int = 17222,
) -> ModelSkillResult[ModelReceiptRuntimeSummary]:
    """A receipt shaped exactly as ``run_receipt_mode`` builds one on a raise."""
    return ModelSkillResult[ModelReceiptRuntimeSummary](
        skill_name=cli_delegate.DELEGATE_NODE_NAME,
        node_name=cli_delegate.DELEGATE_NODE_NAME,
        status=EnumSkillResultStatus.ERROR,
        correlation_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        exit_code=1,
        duration_ms=duration_ms,
        result=ModelReceiptRuntimeSummary(
            workflow_result="error",
            exit_code=1,
            workflow=_CONTRACT,
            terminal_payload=None,
            handler_result=None,
            error=error,
            runtime_error_type=error_type,
            runtime_error_is_transport=is_transport,
        ),
        result_model=(
            "omnibase_infra.cli.model_receipt_runtime_summary."
            "ModelReceiptRuntimeSummary"
        ),
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


def _write(receipt: object, state_root: Path) -> None:
    _write_local_run_files(
        receipt=receipt,
        state_root=state_root,
        addressing=_ADDRESSING,
        prompt="Reply with exactly the word READY",
        task_type="summarization",
        task_type_resolution="explicit",
        broker="omninode-pc.tail75df5e.ts.net:19092",
        command_topic="onex.cmd.omnimarket.delegate-skill.v1",
    )


class TestTransportFailureWritesTypedTerminal:
    """AC-1: the broker was unreachable, so there IS a receipt saying so.

    Falsifier named on the ticket: a connection error that leaves no receipt
    on disk. That is precisely what this asserts against.
    """

    def test_all_three_files_exist(self, tmp_path: Path) -> None:
        receipt = _receipt(is_transport=True)
        _write(receipt, tmp_path)

        run_dir = tmp_path / "runs" / str(receipt.run_id)
        for name in ("result.txt", "receipt.json", "run.json"):
            assert (run_dir / name).exists(), f"{name} was not written"

    def test_cause_names_the_transport_not_a_provider(self, tmp_path: Path) -> None:
        """The cause a reader acts on says 'broker', never 'provider'.

        The OMN-19004 invariant in its sharpest form: all three members of
        the delegation failure enum are provider-side, and a run that never
        reached a broker called no provider. Borrowing one would send the
        reader to an endpoint that was working.
        """
        receipt = _receipt(is_transport=True)
        _write(receipt, tmp_path)

        written = json.loads(
            (tmp_path / "runs" / str(receipt.run_id) / "receipt.json").read_text()
        )

        assert written["terminal_class"] == "transport"
        assert written["terminal_failure_cause"] is None
        refusal = written["transport_refusal"]
        assert refusal["awaited"] == "broker_connection"
        assert refusal["reason"] == "broker_unreachable"
        assert refusal["transport_error_type"] == "InfraConnectionError"
        assert refusal["broker"] == "omninode-pc.tail75df5e.ts.net:19092"
        assert refusal["command_topic"] == "onex.cmd.omnimarket.delegate-skill.v1"

    def test_distinguishable_from_still_running_and_from_quality_failure(
        self, tmp_path: Path
    ) -> None:
        """AC-1's three-way distinction, asserted as three-way.

        'Still running' writes nothing at all, so the file's existence
        settles that half. The quality half needs the record to carry no
        rung: a transport failure that reported attempts would be
        indistinguishable from a model that answered badly.
        """
        receipt = _receipt(is_transport=True)
        _write(receipt, tmp_path)

        written = json.loads(
            (tmp_path / "runs" / str(receipt.run_id) / "receipt.json").read_text()
        )

        assert written["status"] == EnumSkillResultStatus.FAILED.value
        assert written["attempts"] == []
        assert written["route_attributed"] is False
        # No rung ran, so no quality verdict exists to be stated -- absent,
        # not null-with-an-implied-verdict.
        for quality_key in (
            "quality_gate_passed",
            "quality_score",
            "quality_gates_failed",
        ):
            assert quality_key not in written

    def test_no_route_identity_is_synthesised(self, tmp_path: Path) -> None:
        """Nothing was selected, so nothing is attributed."""
        receipt = _receipt(is_transport=True)
        _write(receipt, tmp_path)

        written = json.loads(
            (tmp_path / "runs" / str(receipt.run_id) / "receipt.json").read_text()
        )
        for attributed_key in ("backend_id", "model", "endpoint", "routing_tier"):
            assert attributed_key not in written

    def test_elapsed_and_bound_are_both_recorded(self, tmp_path: Path) -> None:
        """AC-2's 'naming the deadline it exceeded', on the written artifact."""
        receipt = _receipt(is_transport=True, duration_ms=17222)
        _write(receipt, tmp_path)

        refusal = json.loads(
            (tmp_path / "runs" / str(receipt.run_id) / "receipt.json").read_text()
        )["transport_refusal"]

        assert refusal["elapsed_seconds"] == pytest.approx(17.222)
        assert refusal["bound_seconds"] > 0
        assert refusal["attempts_permitted"] >= 1

    def test_a_non_transport_unresolvable_terminal_still_raises(
        self, tmp_path: Path
    ) -> None:
        """POSITIVE CONTROL for the refusal that must NOT be softened.

        OMN-18569 made an unresolvable terminal raise rather than return
        quietly, and that is still right for every non-transport cause. If
        this test ever goes green by writing a file instead, the C16 fix has
        swallowed the OMN-18569 guarantee, which would be a worse defect
        than the one it closed.
        """
        receipt = _receipt(is_transport=False, error_type="ValidationError")

        with pytest.raises(DelegateTerminalUnresolvedError):
            _write(receipt, tmp_path)

        assert not (tmp_path / "runs").exists()

    def test_flag_without_an_error_type_does_not_fabricate_a_refusal(
        self, tmp_path: Path
    ) -> None:
        """A truncated envelope is reported, not papered over.

        The flag is only ever set beside a captured exception type. One
        without the other is a malformed receipt, and writing a refusal that
        cannot name what failed would be inventing the very field a reader
        needs.
        """
        receipt = _receipt(is_transport=True, error_type="")

        with pytest.raises(DelegateTerminalUnresolvedError):
            _write(receipt, tmp_path)

        assert not (tmp_path / "runs").exists()


class TestUnreachableBrokerFailsFastAndTyped:
    """AC-4: a permanently unreachable broker is typed, bounded, never a hang.

    Falsifier named on the ticket: an unreachable-broker test that hangs past
    its declared deadline.
    """

    def test_refusal_is_constructible_for_the_pre_dispatch_probe(self) -> None:
        """The locus-probe refusal is a distinct reason, not the same one.

        A probe finding no live consumer group may be reporting a lane that
        is not running rather than a sick broker. Collapsing the two would
        send half the readers to the wrong place.
        """
        refusal = ModelDelegateTransportRefusal(
            reason="locus_probe_refused",
            correlation_id=uuid.uuid4(),
            bus="kafka",
            locus=EnumDelegateLocus.DEPLOYED_LANE.value,
            broker="127.0.0.1:1",
            attempts_permitted=4,
            bound_seconds=127.0,
            elapsed_seconds=5.0,
            transport_error_type="DelegateLocusRefusedError",
            transport_error="no live consumer group on the command topic",
        )

        assert refusal.awaited == "broker_connection"
        assert refusal.reason == "locus_probe_refused"
        # Fast-failure is visible in the record: well under the bound.
        assert refusal.elapsed_seconds < refusal.bound_seconds

    def test_written_artifacts_survive_a_probe_refusal(self, tmp_path: Path) -> None:
        run_id = str(uuid.uuid4())
        cli_delegate._write_transport_refusal_run_files(
            refusal=ModelDelegateTransportRefusal(
                reason="locus_probe_refused",
                correlation_id=uuid.uuid4(),
                bus="kafka",
                locus=EnumDelegateLocus.DEPLOYED_LANE.value,
                broker="127.0.0.1:1",
                attempts_permitted=4,
                bound_seconds=127.0,
                elapsed_seconds=5.0,
                transport_error_type="DelegateLocusRefusedError",
                transport_error="no live consumer group",
            ),
            run_id=run_id,
            state_root=tmp_path,
            prompt="Reply with exactly the word READY",
            task_type="summarization",
            task_type_resolution="explicit",
            addressing=_ADDRESSING,
        )

        written = json.loads((tmp_path / "runs" / run_id / "receipt.json").read_text())
        assert written["transport_refusal"]["reason"] == "locus_probe_refused"
        assert written["run_id"] == run_id


class TestRefusalModelRefusesAnUntruthfulRecord:
    """The model itself will not hold a contradictory record (OMN-19004)."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("attempts_permitted", 0),
            ("bound_seconds", 0.0),
            ("elapsed_seconds", -1.0),
            ("transport_error_type", ""),
            ("bus", ""),
            ("locus", ""),
        ],
    )
    def test_impossible_values_are_refused(self, field: str, value: object) -> None:
        kwargs: dict[str, object] = {
            "reason": "broker_unreachable",
            "correlation_id": uuid.uuid4(),
            "bus": "kafka",
            "locus": "deployed_lane",
            "attempts_permitted": 4,
            "bound_seconds": 127.0,
            "elapsed_seconds": 17.2,
            "transport_error_type": "InfraConnectionError",
        }
        kwargs[field] = value
        with pytest.raises(ValueError):
            ModelDelegateTransportRefusal(**kwargs)  # type: ignore[arg-type]

    def test_an_unknown_reason_is_refused(self) -> None:
        """The reason set is closed: a third transport stage needs a decision."""
        with pytest.raises(ValueError):
            ModelDelegateTransportRefusal(
                reason="something_else",  # type: ignore[arg-type]
                correlation_id=uuid.uuid4(),
                bus="kafka",
                locus="deployed_lane",
                attempts_permitted=4,
                bound_seconds=127.0,
                elapsed_seconds=17.2,
                transport_error_type="InfraConnectionError",
            )
