# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CPU architectures the runner fleet is built for (OMN-17477)."""

from __future__ import annotations

from enum import Enum


class EnumRunnerHostArch(str, Enum):
    """Architectures a fleet host may declare.

    Spelled the way Docker's ``TARGETARCH`` build argument spells them, because
    that value is what the runner image's download URLs are resolved from --
    keeping one spelling means the inventory, the image and the runner label
    cannot drift into three different names for the same CPU.

    Closed on purpose. An unrecognised value here would produce a runner label
    no workflow targets and an image tag nothing builds, so it fails validation
    rather than becoming a host nobody can place work on.
    """

    AMD64 = "amd64"
    ARM64 = "arm64"


__all__ = ["EnumRunnerHostArch"]
