# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structured reason+detail on the NOT-READY / reconciliation-exhausted logs
(OMN-15578).

``ModelTopicReadinessFailure`` carries a classified ``reason``
(``EnumTopicReadinessFailureReason``) and a human-readable ``detail`` per
topic, computed by the readiness confirm boundary. Two warning sites in
``handler_wiring.py`` discarded both fields before OMN-15578 and logged only
the bare topic/contract name:

- The per-contract "NOT-READY: topic metadata did not converge" warning
  (``_interleave_contract``), which previously did
  ``[f.topic for f in readiness.failures]``.
- The "NOT_READY reconciliation exhausted" warning
  (``run_not_ready_reconciliation_loop``), which previously did
  ``sorted(pending)`` (contract names only).

This directly blocked root-causing OMN-15577 AC1 (the cold-boot topic-storm
defect) — the classified reason (topic-absent / partition-mismatch / no-leader
/ config-mismatch, OMN-13237) and the detail string were computed and thrown
away at the logging boundary.

Reuses the real topic_match contract fixture + fake provisioner/bus doubles
from ``test_not_ready_reconciliation_omn15215.py`` (same class of test double
already accepted for this exact machinery — no live broker in a unit test).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_infra.event_bus.enum_topic_readiness_failure_reason import (
    EnumTopicReadinessFailureReason,
)
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_readiness_failure import (
    ModelTopicReadinessFailure,
)
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    run_not_ready_reconciliation_loop,
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.report import ModelAutoWiringReport
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

from .test_not_ready_reconciliation_omn15215 import (
    CONTRACT_NAME,
    TOPICS,
    HandlerFixtureNotReady,
    RecordingBus,
    _write_contract,
)

DELIBERATE_REASON = EnumTopicReadinessFailureReason.PARTITION_MISMATCH
DELIBERATE_DETAIL = (
    "expected 6 partitions, broker metadata reported 3 (omn15578-fixture)"
)


class DeliberateFailureProvisioner:
    """Always NOT_READY, with a fixed, distinctive reason+detail per topic.

    Distinct from ``FlakyProvisioner``/``AlwaysNotReadyProvisioner`` in the
    OMN-15215 fixture module: this one sets a non-default ``reason`` (those
    fixtures use the enum default-adjacent ``TOPIC_ABSENT`` with an EMPTY
    detail) and a deliberately identifiable ``detail`` string, so a test
    assertion cannot pass by accident against a zero-value field.
    """

    def __init__(self) -> None:
        self.confirm_calls = 0

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: object | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        return True

    async def confirm_topics_ready(
        self,
        topics: Sequence[str],
        *,
        expected_specs: Mapping[str, object] | None = None,
        config: ModelTopicReadinessConfig | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelTopicSetReadiness:
        self.confirm_calls += 1
        return ModelTopicSetReadiness(
            topics=tuple(topics),
            status=EnumTopicReadinessStatus.NOT_READY,
            ready_topics=(),
            failures=tuple(
                ModelTopicReadinessFailure(
                    topic=t,
                    reason=DELIBERATE_REASON,
                    detail=DELIBERATE_DETAIL,
                )
                for t in topics
            ),
            attempts=1,
        )


async def _wire_fixture_contract(
    contract_path: Path,
) -> tuple[ModelAutoWiringManifest, MessageDispatchEngine, ModelAutoWiringReport]:
    discovered = discover_contracts_from_paths([contract_path])
    contracts = getattr(discovered, "contracts", discovered)
    manifest = ModelAutoWiringManifest(contracts=tuple(contracts))
    engine = MessageDispatchEngine()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerFixtureNotReady,
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=None,
            environment="test",
            subscribe_immediately=False,
        )
    result = next(r for r in report.results if r.contract_name == CONTRACT_NAME)
    assert result.wirings
    return manifest, engine, report


class TestNotReadyWarningCarriesReasonAndDetail:
    """AC2 (site 1): the per-contract NOT-READY warning."""

    @pytest.mark.asyncio
    async def test_not_ready_log_record_carries_reason_and_detail(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        contract_path = _write_contract(tmp_path)
        manifest, engine, report = await _wire_fixture_contract(contract_path)

        provisioner = DeliberateFailureProvisioner()
        bus = RecordingBus()
        attach_out: list[ModelContractAttachResult] = []

        with caplog.at_level(
            "WARNING", logger="omnibase_infra.runtime.auto_wiring.handler_wiring"
        ):
            await subscribe_wired_contract_topics(
                manifest=manifest,
                report=report,
                dispatch_engine=engine,
                event_bus=bus,
                environment="test",
                provisioner=provisioner,
                readiness_config=ModelTopicReadinessConfig(),
                attach_results_out=attach_out,
            )

        not_ready_records = [r for r in caplog.records if "NOT-READY" in r.getMessage()]
        assert not_ready_records, (
            "expected a NOT-READY warning log record; none were emitted "
            f"(all records: {[r.getMessage() for r in caplog.records]})"
        )
        record = not_ready_records[0]

        # AC1/AC2: the record must carry reason+detail per topic, not just
        # the topic name. Structured fields (extra=) are asserted directly
        # off the LogRecord attributes so the data is grep/query-able, not
        # buried in a formatted string.
        assert hasattr(record, "readiness_failures"), (
            "NOT-READY log record has no structured 'readiness_failures' "
            "field — reason/detail were not attached via extra={}"
        )
        failures = record.readiness_failures
        assert failures, "readiness_failures must not be empty"
        for failure in failures:
            assert failure["reason"] == DELIBERATE_REASON.value, (
                f"expected reason={DELIBERATE_REASON.value!r}, got {failure!r}"
            )
            assert failure["detail"] == DELIBERATE_DETAIL, (
                f"expected detail={DELIBERATE_DETAIL!r}, got {failure!r}"
            )
            assert failure["topic"] in TOPICS


class TestReconciliationExhaustedWarningCarriesReasonAndDetail:
    """AC2 (site 2): the reconciliation-exhausted warning."""

    @pytest.mark.asyncio
    async def test_reconciliation_exhausted_log_record_carries_reason_and_detail(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        contract_path = _write_contract(tmp_path)
        manifest, engine, report = await _wire_fixture_contract(contract_path)

        provisioner = DeliberateFailureProvisioner()
        bus = RecordingBus()
        attach_out: list[ModelContractAttachResult] = []
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(),
            attach_results_out=attach_out,
        )

        async def _fake_sleep(seconds: float) -> None:
            return None

        with caplog.at_level(
            "WARNING", logger="omnibase_infra.runtime.auto_wiring.handler_wiring"
        ):
            await run_not_ready_reconciliation_loop(
                manifest,
                attach_out,
                engine,
                bus,
                "test",
                provisioner=provisioner,
                readiness_config=ModelTopicReadinessConfig(),
                max_attempts=2,
                sleep=_fake_sleep,
            )

        exhausted_records = [
            r for r in caplog.records if "reconciliation exhausted" in r.getMessage()
        ]
        assert exhausted_records, (
            "expected a 'reconciliation exhausted' warning log record; none "
            f"were emitted (all records: {[r.getMessage() for r in caplog.records]})"
        )
        record = exhausted_records[0]

        assert hasattr(record, "readiness_failures"), (
            "reconciliation-exhausted log record has no structured "
            "'readiness_failures' field — reason/detail were not attached "
            "via extra={}"
        )
        failures_by_contract = record.readiness_failures
        assert CONTRACT_NAME in failures_by_contract, (
            f"expected contract {CONTRACT_NAME!r} in {failures_by_contract!r}"
        )
        contract_failures = failures_by_contract[CONTRACT_NAME]
        assert contract_failures, "per-contract failure list must not be empty"
        for failure in contract_failures:
            assert failure["reason"] == DELIBERATE_REASON.value, (
                f"expected reason={DELIBERATE_REASON.value!r}, got {failure!r}"
            )
            assert failure["detail"] == DELIBERATE_DETAIL, (
                f"expected detail={DELIBERATE_DETAIL!r}, got {failure!r}"
            )
