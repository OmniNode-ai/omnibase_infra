# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed inputs, policy and terminal result for runner TMPDIR preparation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelDeployRunnersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["runner.tmp.prepare"] = "runner.tmp.prepare"
    host: str
    host_dir: str
    compose_file: str
    prefix: str
    count: int = Field(ge=1)
    dry_run: bool = False
    secondary: bool = True


class ModelDeployRunnersResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    prepared: tuple[str, ...] = ()
    error_message: str | None = None


class ModelRunnerTmpPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    uid: int = Field(ge=0)
    mode: str
    timeout_seconds: int = Field(gt=0)


class ModelRunnerTmpVolume(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str
    source: str = ""
    target: str


class ModelRunnerTmpService(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    environment: dict[str, str | None] = {}
    volumes: tuple[ModelRunnerTmpVolume, ...] = ()


class ModelRunnerTmpCompose(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    services: dict[str, ModelRunnerTmpService]
