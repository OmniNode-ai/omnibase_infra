# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Global application-relation ownership validation report."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.validation.models.model_application_relation_declaration import (
    ModelApplicationRelationDeclaration,
)
from omnibase_infra.validation.models.model_application_relation_violation import (
    ModelApplicationRelationViolation,
)
from omnibase_infra.validation.models.model_live_application_relation import (
    RelationIdentity,
)


class ModelApplicationRelationOwnershipReport(BaseModel):
    """Complete normalized projection and its fail-closed violations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    declarations: tuple[ModelApplicationRelationDeclaration, ...]
    violations: tuple[ModelApplicationRelationViolation, ...]

    @property
    def is_valid(self) -> bool:
        return not self.violations

    def readers_for(self, identity: RelationIdentity) -> tuple[str, ...]:
        readers = {
            reader
            for declaration in self.declarations
            if declaration.identity == identity
            for reader in declaration.readers
        }
        return tuple(sorted(readers))


__all__ = ["ModelApplicationRelationOwnershipReport"]
