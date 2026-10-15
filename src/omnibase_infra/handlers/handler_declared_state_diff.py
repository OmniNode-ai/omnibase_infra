# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure diff of declared items against one round's observations."""

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.declared_state.enum_declared_state_operation_kind import (
    EnumDeclaredStateOperationKind,
)
from omnibase_infra.models.declared_state.model_declared_state_diff import (
    ModelDeclaredStateDiff,
)
from omnibase_infra.models.declared_state.model_declared_state_diff_request import (
    ModelDeclaredStateDiffRequest,
)
from omnibase_infra.models.declared_state.model_declared_state_operation import (
    ModelDeclaredStateOperation,
)


class HandlerDeclaredStateDiff:
    """Plan from what was observed, never from what a past round applied.

    A surface the round could not fully read plans nothing: a row that failed
    to read, or a declared item with no row at all, makes the whole surface
    unobservable, so silence never reads as "in sync".
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelDeclaredStateDiffRequest) -> ModelDeclaredStateDiff:
        """Plan creates and updates for declared items, then deletes of the rest."""
        rows = {row.item_key: row for row in request.observations}
        unreadable = sorted(key for key, row in rows.items() if not row.read_ok)
        unreached = sorted(
            item.item_key for item in request.declared if item.item_key not in rows
        )
        if unreadable or unreached:
            parts = []
            if unreadable:
                parts.append(f"unreadable items {unreadable}")
            if unreached:
                parts.append(f"declared items with no observation {unreached}")
            return self._diff(request, (), "; ".join(parts))

        operations: list[ModelDeclaredStateOperation] = []
        declared_keys = set()
        for item in sorted(request.declared, key=lambda i: i.item_key):
            declared_keys.add(item.item_key)
            observed = rows[item.item_key].observed_digest
            if observed == item.digest:
                continue
            kind = (
                EnumDeclaredStateOperationKind.CREATE
                if observed is None
                else EnumDeclaredStateOperationKind.UPDATE
            )
            operations.append(
                self._operation(request, kind, item.item_key, item.digest)
            )
        operations.extend(
            self._operation(request, EnumDeclaredStateOperationKind.DELETE, key, None)
            for key, row in sorted(rows.items())
            if key not in declared_keys and row.observed_digest is not None
        )
        return self._diff(request, tuple(operations), None)

    @staticmethod
    def _operation(
        request: ModelDeclaredStateDiffRequest,
        kind: EnumDeclaredStateOperationKind,
        item_key: str,
        digest: str | None,
    ) -> ModelDeclaredStateOperation:
        return ModelDeclaredStateOperation(
            kind=kind,
            surface_kind=request.surface_kind,
            surface_key=request.surface_key,
            item_key=item_key,
            digest=digest,
        )

    @staticmethod
    def _diff(
        request: ModelDeclaredStateDiffRequest,
        operations: tuple[ModelDeclaredStateOperation, ...],
        unobservable_reason: str | None,
    ) -> ModelDeclaredStateDiff:
        return ModelDeclaredStateDiff(
            surface_kind=request.surface_kind,
            surface_key=request.surface_key,
            generation=request.generation,
            operations=operations,
            unobservable_reason=unobservable_reason,
        )


__all__ = ["HandlerDeclaredStateDiff"]
