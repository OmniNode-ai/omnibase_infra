# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The receipt one runtime canary run writes (OMN-26463)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.enums.enum_runtime_canary_verdict import EnumRuntimeCanaryVerdict
from omnibase_infra.models.model_runtime_canary_candidate import (
    ModelRuntimeCanaryCandidate,
)

__all__: list[str] = ["ModelRuntimeCanaryReceipt"]


class ModelRuntimeCanaryReceipt(BaseModel):
    """PASS or FAIL, the timings of each check and the first line that failed.

    A FAIL names its first failing line and a PASS names none, so a reader
    never has to open the container log to learn why a candidate was refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumRuntimeCanaryVerdict
    candidate: ModelRuntimeCanaryCandidate
    started_at: str
    finished_at: str
    ready_seconds: float | None = None
    wake_seconds: float | None = None
    heartbeat_seconds: float | None = None
    contracts_dormant: tuple[str, ...] = ()
    contracts_woken: tuple[str, ...] = ()
    contracts_unwoken: tuple[str, ...] = ()
    first_failing_line: str | None = Field(default=None)
    failed_check: str | None = None
    log_path: str | None = None

    @model_validator(mode="after")
    def _verdict_matches_evidence(self) -> ModelRuntimeCanaryReceipt:
        failing = (self.first_failing_line or "").strip()
        if self.verdict is EnumRuntimeCanaryVerdict.FAIL and not failing:
            raise ValueError("a FAIL receipt names its first failing line")
        if self.verdict is EnumRuntimeCanaryVerdict.PASS and (
            failing or self.contracts_unwoken
        ):
            raise ValueError(
                "a PASS receipt has no failing line and no unwoken contract"
            )
        return self
