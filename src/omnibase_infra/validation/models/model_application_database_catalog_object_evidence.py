# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Exact live-catalog identity and owner for one managed-schema object."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.validation.types.type_application_database_function_signature import (
    ApplicationDatabaseFunctionSignature,
)


class ModelApplicationDatabaseCatalogObjectEvidence(BaseModel):
    """Current object identity used by the transaction-start catalog guard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_ref: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    catalog_kind: Literal[
        "table",
        "view",
        "materialized_view",
        "foreign_table",
        "sequence",
        "function",
        "aggregate",
        "window_function",
        "procedure",
        "type",
        "base_type",
        "range_type",
        "multirange_type",
    ]
    object_ref: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    function_signature: ApplicationDatabaseFunctionSignature | None = None
    owner: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")

    @model_validator(mode="after")
    def validate_signature_shape(
        self,
    ) -> ModelApplicationDatabaseCatalogObjectEvidence:
        """Require exact signatures only for routines."""
        routine = self.catalog_kind in {
            "function",
            "aggregate",
            "window_function",
            "procedure",
        }
        if routine != (self.function_signature is not None):
            raise ValueError(
                "catalog routine evidence requires a signature and non-routines forbid it"
            )
        return self

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """Return the stable live-catalog identity."""
        return (
            self.catalog_kind,
            self.schema_ref,
            self.object_ref,
            self.function_signature or "",
        )


__all__ = ["ModelApplicationDatabaseCatalogObjectEvidence"]
