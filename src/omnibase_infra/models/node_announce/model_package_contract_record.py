# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One contract of one installed package, as the metadata reader found it (OMN-21590)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPackageContractRecord(BaseModel):
    """What the metadata reader learned about one ``onex.nodes`` entry.

    A record with ``error_stage`` set is a failure the discovery manifest reports as
    an error; the stage says how far the reader got, so the manifest keeps the
    messages and ``contract_path`` it carried before the scan moved here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_point: str = Field(description="Entry point name in the package metadata")
    distribution: str = Field(description="Installed distribution name")
    version: str = Field(description="Exact installed distribution version")
    contract_path: Path | None = Field(
        default=None, description="Where the contract file was found, once resolved"
    )
    contract_name: str | None = Field(
        default=None, description="The contract's own name, or the entry point name"
    )
    content_hash: str | None = Field(
        default=None, description="Canonical content hash of the file's bytes"
    )
    declared_topics: tuple[str, ...] = Field(
        default=(), description="Every topic the contract declares"
    )
    runtime_profiles: tuple[str, ...] = Field(
        default=(), description="Runtime profiles the contract names"
    )
    error_stage: Literal["load", "resolve", "parse"] | None = Field(
        default=None, description="How far the reader got before it failed"
    )
    error: str | None = Field(default=None, description="The failure message")
