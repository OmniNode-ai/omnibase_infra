# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rendered Compose service entry."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_runner_tmp_volume import ModelRunnerTmpVolume


class ModelRunnerTmpContainer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    environment: dict[str, str | None] = {}
    volumes: tuple[ModelRunnerTmpVolume, ...] = ()
