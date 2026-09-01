# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scalar validators shared by the payload-free verified-grant models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

VerifiedGrantExpectedOutputTopic = Annotated[
    str,
    Field(
        min_length=3,
        max_length=255,
        pattern=r"^[a-z][a-z0-9-]*(?:[.][a-z][a-z0-9-]*)+$",
    ),
]
CanonicalModelEventClass = Annotated[
    str, Field(min_length=6, max_length=255, pattern=r"^Model[A-Za-z0-9]+$")
]
OpaqueIdentifier = Annotated[
    str, Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
FirstEffectRetryDisposition = Literal[
    "forbidden",
    "never-republish-after-ambiguous.v1",
]

__all__ = [
    "VerifiedGrantExpectedOutputTopic",
    "CanonicalModelEventClass",
    "FirstEffectRetryDisposition",
    "OpaqueIdentifier",
]
