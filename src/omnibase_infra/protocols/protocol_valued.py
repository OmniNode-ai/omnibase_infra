# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for an enum-like object exposing a string ``value`` (OMN-26718)."""

from __future__ import annotations

from typing import Protocol


class ProtocolValued(Protocol):
    @property
    def value(self) -> str: ...
