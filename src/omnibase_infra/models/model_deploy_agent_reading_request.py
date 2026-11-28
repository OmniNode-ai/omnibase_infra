# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the three deploy agents' health answers and the merge's clock read (OMN-26724)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelDeployAgentReadingRequest(BaseModel):
    """Facts a reader took from the dev, stability and prod agents; the handler judges them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The dev agent's ``loaded_code_sha`` from its /health ("" when absent).
    dev_loaded_code_sha: str = ""
    #: The dev agent's ``uptime_seconds``; None when its health did not answer.
    dev_uptime_seconds: int | None = None
    #: The merge of the build-ref fix is an ancestor of the dev agent's loaded sha (read by git).
    dev_loaded_sha_has_merge: bool = False
    #: Seconds from that merge's commit time to the reading.
    merge_age_seconds: int = 0
    #: The dev agent's ``last_result`` status and runtime lane ("" when it has none).
    dev_last_status: str = ""
    dev_last_runtime_lane: str = ""
    #: How many phase results the dev agent's last result records.
    dev_last_phase_result_count: int = 0
    #: The stability and prod agents' ``uptime_seconds``; None when one did not answer.
    stability_uptime_seconds: int | None = None
    prod_uptime_seconds: int | None = None
