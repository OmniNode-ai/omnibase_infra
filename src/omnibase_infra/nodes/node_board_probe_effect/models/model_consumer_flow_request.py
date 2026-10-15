# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane and bounded sampling controls for the C28 board check."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelConsumerFlowRequest(BaseModel):
    """CLI-equivalent controls; subject_lane is the board result's subject."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    subject_lane: str = Field(min_length=1)
    base_url: str = Field(default="http://host.docker.internal:3002", min_length=1)
    docker_bin: str = Field(default="docker", min_length=1)
    samples: int = Field(default=4, ge=2)
    sample_interval: float = Field(default=30.0, ge=0, allow_inf_nan=False)
    settle_seconds: float = Field(default=300.0, ge=0, allow_inf_nan=False)
    # OMN-19812: extra wait per settle window while every unsettled container
    # reports Docker health "starting" (inside its declared start period). The
    # dev runtime turned healthy ~12 min after compose-up on 2026-10-14; two
    # windows of 300 + 300 bound a booting lane at 20 min, inside the C28 job's
    # 1700 s node timeout with the sampling, injection and negative tests.
    boot_grace_seconds: float = Field(default=300.0, ge=0, allow_inf_nan=False)
    injection_wait: float = Field(default=120.0, ge=0, allow_inf_nan=False)
    attempts: int = Field(default=2, ge=1)
    pytest_cmd: str = Field(default="uv run --frozen pytest", min_length=1)
    scratch: Path = Path()

    record: Path | None = None
