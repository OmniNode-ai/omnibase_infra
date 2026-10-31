# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""One entry of the contract-declared check catalog (OMN-21623)."""

from __future__ import annotations

import importlib
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumCheckSeverity, EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_infra.validation.checks.handler_check_executor import (
        HandlerCheckExecutor,
    )

_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "validation" / "checks" / "check_catalog.yaml"
)
_HANDLER_NAMESPACE = "omnibase_infra.validation.checks."


def _configuration_error(message: str, operation: str) -> ProtocolConfigurationError:
    context = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.FILESYSTEM,
        operation=operation,
    )
    return ProtocolConfigurationError(message, context=context)


class ModelCheckCatalogEntry(BaseModel):
    """A check in ``validation/checks/check_catalog.yaml``.

    Attributes:
        check_code: Unique check identifier (e.g., ``CHECK-PY-001``).
        handler: Dotted path of the executor class that runs the check.
        label: Human-readable name; set only for subprocess checks.
        severity: Severity; set only for subprocess checks.
        command: Command line; set only for subprocess checks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    check_code: str = Field(min_length=1)
    handler: str = Field(min_length=1)
    label: str = ""
    severity: EnumCheckSeverity | None = None
    command: str = ""

    @classmethod
    @cache
    def load_catalog(cls) -> tuple[ModelCheckCatalogEntry, ...]:
        """Read the declared catalog entries, in catalog order."""
        with _CATALOG_PATH.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        rows = document.get("checks") if isinstance(document, dict) else None
        if not isinstance(rows, list) or not rows:
            raise _configuration_error(
                f"{_CATALOG_PATH.name} declares no checks", "load_check_catalog"
            )
        entries = tuple(cls.model_validate(row) for row in rows)
        if len({entry.check_code for entry in entries}) != len(entries):
            raise _configuration_error(
                f"{_CATALOG_PATH.name} repeats a check_code", "load_check_catalog"
            )
        return entries

    def build_executor(self, **overrides: Path | None) -> HandlerCheckExecutor:
        """Build the executor this entry declares.

        Subprocess entries pass their label, severity and command to the
        class; other executors take only ``overrides``.

        Raises:
            ProtocolConfigurationError: If the handler is outside the checks
                namespace, is not an executor, or reports another check code.
        """
        # The checks package builds its registry at import time. Import the
        # executor base only after this catalog model has finished loading.
        from omnibase_infra.validation.checks.handler_check_executor import (
            HandlerCheckExecutor,
        )

        module_path, _, class_name = self.handler.rpartition(".")
        if not module_path.startswith(_HANDLER_NAMESPACE):
            raise _configuration_error(
                f"{self.check_code}: handler {self.handler!r} is outside "
                f"{_HANDLER_NAMESPACE}",
                "build_check_executor",
            )
        handler_class = getattr(importlib.import_module(module_path), class_name)
        if not (
            isinstance(handler_class, type)
            and issubclass(handler_class, HandlerCheckExecutor)
        ):
            raise _configuration_error(
                f"{self.check_code}: {self.handler!r} is not a HandlerCheckExecutor",
                "build_check_executor",
            )
        kwargs: dict[str, object] = dict(overrides)
        if self.command:
            kwargs.update(
                check_code=self.check_code,
                label=self.label,
                severity=self.severity,
                command=self.command,
            )
        executor: HandlerCheckExecutor = handler_class(**kwargs)
        if executor.check_code != self.check_code:
            raise _configuration_error(
                f"{self.check_code}: {self.handler!r} reports check_code "
                f"{executor.check_code!r}",
                "build_check_executor",
            )
        return executor


__all__: list[str] = ["ModelCheckCatalogEntry"]
