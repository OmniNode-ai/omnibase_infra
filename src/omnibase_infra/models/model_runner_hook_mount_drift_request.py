# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to compare every runner's executed hooks with the host copies."""

from pydantic import BaseModel, ConfigDict


class ModelRunnerHookMountDriftRequest(BaseModel):
    """Where the host copies live and which containers are runners."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host_hook_dir: str
    container_match: str
    scripts: tuple[str, ...] = (
        "runner-job-started.sh",
        "entrypoint.sh",
        "healthcheck.sh",
    )
