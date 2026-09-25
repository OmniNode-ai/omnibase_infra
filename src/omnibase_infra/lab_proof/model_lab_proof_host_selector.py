# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Where a profile variant may run.

Ticket: OMN-19565
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLabProofHostSelector(BaseModel):
    """Host requirements a scheduler (OMN-19569) and a hand-run prover read.

    ``max_load_ratio`` is load1 divided by the host's core count; the plan
    refuses to start a stack above it (a host at load 72-82 could not hold an
    isolated runtime healthy, RELEASE 2026-09-25T13:08:23Z lane=prove-200).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_hosts: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    min_free_mem_mib: int = Field(ge=0)
    max_load_ratio: float = Field(gt=0.0)


__all__ = ["ModelLabProofHostSelector"]
