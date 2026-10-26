# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Load the declared lab lease kinds and their inventory rows (OMN-21427)."""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.lease import ModelLeaseKindDeclaration, ModelLeaseKindPolicy

LAB_LEASE_KINDS_PATH = Path(__file__).with_name("lab_lease_kinds.yaml")


class HandlerLeaseKinds:
    """Reads a lease kind declaration file."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def load_lease_kinds(self, path: Path) -> tuple[ModelLeaseKindDeclaration, ...]:
        """Validate each declaration and reject repeated kinds."""
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        declarations: list[ModelLeaseKindDeclaration] = []
        seen: set[str] = set()
        for entry in document["kinds"]:
            policy_data = dict(entry)
            inventory_row = policy_data.pop("inventory_row")
            replaces = policy_data.pop("replaces")
            policy = ModelLeaseKindPolicy.model_validate(policy_data)
            kind = policy.kind
            if kind in seen:
                raise ValueError(f"lease kind {kind!r} declared twice")
            seen.add(kind)
            declarations.append(
                ModelLeaseKindDeclaration(
                    policy=policy, inventory_row=inventory_row, replaces=replaces
                )
            )
        return tuple(declarations)


__all__ = ["LAB_LEASE_KINDS_PATH", "HandlerLeaseKinds"]
