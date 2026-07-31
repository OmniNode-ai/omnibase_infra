# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL-catalog identity used by the exact application census gate."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.validation.enums.enum_application_inventory_object_kind import (
    EnumApplicationInventoryObjectKind,
)
from omnibase_infra.validation.types.type_application_database_function_signature import (
    ApplicationDatabaseFunctionSignature,
)


class ModelApplicationDatabaseCatalogIdentity(BaseModel):
    """One observed base relation, view, materialized view, or function."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema: str = Field(  # type: ignore[assignment]
        ..., pattern=r"^[a-z_][a-z0-9_]*$"
    )
    name: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    kind: EnumApplicationInventoryObjectKind
    function_signature: ApplicationDatabaseFunctionSignature | None = None

    @property
    def identity(
        self,
    ) -> tuple[
        str,
        str,
        EnumApplicationInventoryObjectKind,
        ApplicationDatabaseFunctionSignature | None,
    ]:
        """Return the exact application-catalog identity."""
        return (self.schema, self.name, self.kind, self.function_signature)


__all__ = ["ModelApplicationDatabaseCatalogIdentity"]
