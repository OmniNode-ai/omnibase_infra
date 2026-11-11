# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to grade the C16 pinned-backend probe (OMN-23176)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelC16PinnedBackendRequest(BaseModel):
    """What to grade: a config of role -> backend id, and where observations come from.

    A ``replay_path`` grades recorded runs and rows. Without one the handler
    pins a delegation per role through ``delegate_argv`` and reads each row
    through ``psql_argv`` (empty grades SKIP, which is red).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_path: str = Field(description="YAML naming one backend id per role.")
    replay_path: str = ""
    delegate_argv: tuple[str, ...] = ()
    psql_argv: tuple[str, ...] = ()
    record_path: str = ""
    timeout_seconds: float = Field(
        description="Seconds each delegate or reader process may run; the workflow sets it."
    )
