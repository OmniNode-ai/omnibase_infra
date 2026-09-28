# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Explicit database boundaries for an execution graph read."""

from __future__ import annotations

from typing import Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, SecretStr, model_validator


class ModelExecutionGraphReadDatabases(BaseModel):
    """Owner authority and event evidence live in separate databases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analytics_dsn: SecretStr
    ledger_dsn: SecretStr

    @model_validator(mode="after")
    def require_distinct_postgres_databases(self) -> Self:
        parsed = tuple(
            urlparse(value.get_secret_value())
            for value in (self.analytics_dsn, self.ledger_dsn)
        )
        for dsn in parsed:
            if (
                dsn.scheme not in {"postgres", "postgresql"}
                or not dsn.hostname
                or not dsn.path.lstrip("/")
            ):
                raise ValueError(
                    "graph read database DSNs must name PostgreSQL databases"
                )
        if parsed[0].path == parsed[1].path:
            raise ValueError("graph owner and ledger DSNs must name distinct databases")
        return self


__all__ = ["ModelExecutionGraphReadDatabases"]
