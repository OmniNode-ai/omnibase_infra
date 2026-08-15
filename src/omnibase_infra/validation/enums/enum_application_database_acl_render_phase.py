# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed stages of the additive application-database ACL rollout."""

from enum import StrEnum, unique


@unique
class EnumApplicationDatabaseAclRenderPhase(StrEnum):
    """Separate the additive scaffold from materialized-object hardening."""

    SCAFFOLD = "scaffold"
    FULL = "full"


__all__ = ["EnumApplicationDatabaseAclRenderPhase"]
