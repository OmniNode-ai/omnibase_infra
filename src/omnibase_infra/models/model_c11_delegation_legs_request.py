# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to grade the C11 delegation legs (OMN-23177)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelC11DelegationLegsRequest(BaseModel):
    """What to grade: the legs config and the recorded observations file.

    ``plant_canary_in_answer`` puts the canary in the observed answer before
    grading, so the scan must fail; it proves the scan can fail.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_path: str = Field(description="YAML declaring the legs' expectations.")
    replay_path: str = Field(description="JSON file of recorded leg observations.")
    record_path: str = ""
    plant_canary_in_answer: bool = False
