# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two runtime host listener pools the deploy agent holds (OMN-22379)."""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class EnumRuntimeListenerPool(StrEnum):
    """A pool name is also the systemd fd-store name of its listening socket."""

    MAIN = "runtime-main"
    EFFECTS = "runtime-effects"


__all__: list[str] = ["EnumRuntimeListenerPool"]
