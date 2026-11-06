# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rendered Compose document, TMPDIR fields only."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_runner_tmp_container import ModelRunnerTmpContainer


class ModelRunnerTmpCompose(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    services: dict[str, ModelRunnerTmpContainer]
