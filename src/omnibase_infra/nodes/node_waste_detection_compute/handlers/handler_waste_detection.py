# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Session-window orchestrator for waste detection analyzers."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from uuid import UUID, uuid4

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RepositoryExecutionError
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_agent_loop import (
    analyze_agent_loop,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_high_output import (
    analyze_high_output,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_low_cache import (
    analyze_low_cache,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_model_overkill import (
    analyze_model_overkill,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_retry_waste import (
    analyze_retry_waste,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_tool_failure_waste import (
    analyze_tool_failure_waste,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteDetectionInput,
    ModelWasteFinding,
)
from omnibase_infra.utils import sanitize_error_message

Analyzer = Callable[
    [tuple[ModelWasteCall, ...], datetime], tuple[ModelWasteFinding, ...]
]

logger = logging.getLogger(__name__)

ANALYZERS: tuple[Analyzer, ...] = (
    analyze_tool_failure_waste,
    analyze_agent_loop,
    analyze_retry_waste,
    analyze_high_output,
    analyze_model_overkill,
    analyze_low_cache,
)

UPSERT_WASTE_FINDING_SQL = """
INSERT INTO waste_findings (
    session_id,
    rule_id,
    severity,
    waste_tokens,
    waste_cost_usd,
    evidence,
    evidence_hash,
    dedup_key,
    recommendation,
    repo_name,
    machine_id,
    detected_at
)
VALUES (
    $1,
    $2,
    $3::severity_type,
    $4,
    $5,
    $6::jsonb,
    $7,
    $8,
    $9,
    $10,
    $11,
    $12
)
ON CONFLICT (dedup_key) DO UPDATE SET
    severity = EXCLUDED.severity,
    waste_tokens = EXCLUDED.waste_tokens,
    waste_cost_usd = EXCLUDED.waste_cost_usd,
    evidence = EXCLUDED.evidence,
    recommendation = EXCLUDED.recommendation,
    repo_name = EXCLUDED.repo_name,
    machine_id = EXCLUDED.machine_id,
    detected_at = EXCLUDED.detected_at
"""


class HandlerWasteDetection:
    """Run pure waste analyzers over a session window."""

    @property
    def handler_id(self) -> str:
        return "handler-waste-detection"

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, detection_input: ModelWasteDetectionInput
    ) -> tuple[ModelWasteFinding, ...]:
        """Compute waste findings for a session window."""
        return self.detect(detection_input)

    def detect(
        self, detection_input: ModelWasteDetectionInput
    ) -> tuple[ModelWasteFinding, ...]:
        """Run all analyzers and return deterministic findings."""
        findings: list[ModelWasteFinding] = []
        for analyzer in ANALYZERS:
            findings.extend(
                analyzer(detection_input.calls, detection_input.detected_at)
            )
        return tuple(
            sorted(
                findings,
                key=lambda finding: (
                    finding.rule_id,
                    finding.evidence_hash,
                    finding.dedup_key,
                ),
            )
        )

    async def project_findings(
        self,
        connection: object,
        findings: Sequence[ModelWasteFinding],
        correlation_id: UUID | None = None,
    ) -> None:
        """Project findings into the waste_findings table with dedup upsert."""
        correlation_id = correlation_id or uuid4()
        execute_attr = "execute"
        try:
            run_statement = getattr(connection, execute_attr)
            for finding in findings:
                await run_statement(
                    UPSERT_WASTE_FINDING_SQL,
                    finding.session_id,
                    finding.rule_id,
                    finding.severity,
                    finding.waste_tokens,
                    finding.waste_cost_usd,
                    json.dumps(
                        finding.evidence,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    finding.evidence_hash,
                    finding.dedup_key,
                    finding.recommendation,
                    finding.repo_name,
                    finding.machine_id,
                    finding.detected_at,
                )
        except Exception as exc:
            sanitized_error = sanitize_error_message(exc)
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="project_findings",
                original_error_type=type(exc).__name__,
            )
            logger.warning(
                "Waste finding projection failed",
                extra={
                    "correlation_id": str(context.correlation_id),
                    "operation": "project_findings",
                    "transport_type": EnumInfraTransportType.DATABASE.value,
                    "execute_attr": execute_attr,
                    "sql_reference": "UPSERT_WASTE_FINDING_SQL",
                    "error_message": sanitized_error,
                },
            )
            raise RepositoryExecutionError(
                f"postgres project_findings failed: {sanitized_error}",
                op_name="project_findings",
                table="waste_findings",
                sql_fingerprint="UPSERT_WASTE_FINDING_SQL",
                context=context,
                execute_attr=execute_attr,
                sql_reference="UPSERT_WASTE_FINDING_SQL",
            ) from exc


__all__ = [
    "ANALYZERS",
    "HandlerWasteDetection",
    "UPSERT_WASTE_FINDING_SQL",
]
