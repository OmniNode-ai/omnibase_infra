# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Load the declared lab lease kinds and their inventory rows (OMN-21427)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import ValidationError

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
        if not isinstance(document, Mapping):
            raise ValueError("lease kinds document must be a mapping with a kinds list")
        entries = document.get("kinds")
        if not isinstance(entries, list) or not entries:
            raise ValueError("lease kinds must be a non-empty list")
        declarations: list[ModelLeaseKindDeclaration] = []
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            label = f"lease kinds[{index}]"
            if not isinstance(entry, Mapping):
                raise ValueError(f"{label} must be a mapping")
            kind = entry.get("kind")
            if isinstance(kind, str):
                label += f" (kind {kind!r})"
            for field in ("inventory_row", "replaces"):
                if field not in entry:
                    raise ValueError(f"{label} requires {field}")
            if isinstance(kind, str) and kind in seen:
                raise ValueError(f"{label} declared twice")
            policy_data = dict(entry)
            inventory_row = policy_data.pop("inventory_row")
            replaces = policy_data.pop("replaces")
            try:
                policy = ModelLeaseKindPolicy.model_validate(policy_data)
                declaration = ModelLeaseKindDeclaration(
                    policy=policy, inventory_row=inventory_row, replaces=replaces
                )
            except ValidationError as exc:
                fields = ", ".join(
                    str(error["loc"][0]) for error in exc.errors(include_input=False)
                )
                raise ValueError(f"{label} has invalid fields: {fields}") from None
            seen.add(policy.kind)
            declarations.append(declaration)
        return tuple(declarations)


__all__ = ["LAB_LEASE_KINDS_PATH", "HandlerLeaseKinds"]
