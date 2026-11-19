# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The container roots a dev-lane branch install may write under (OMN-25536).

The roots are data, in ``dev_lane_branch_install_roots.yaml`` beside this module.
The model has no defaults, so a file that omits a field fails to load.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.model_dev_lane_branch_install_root import (
    ModelDevLaneBranchInstallRoot,
)

_ROOTS_PATH = Path(__file__).resolve().parent / "dev_lane_branch_install_roots.yaml"


class ModelDevLaneBranchInstallRoots(BaseModel):
    """The declared roots, in file order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    roots: tuple[ModelDevLaneBranchInstallRoot, ...]

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"schema_version {value} is not 1")
        return value

    @field_validator("roots")
    @classmethod
    def _at_least_one_root(
        cls, value: tuple[ModelDevLaneBranchInstallRoot, ...]
    ) -> tuple[ModelDevLaneBranchInstallRoot, ...]:
        if not value:
            raise ValueError("at least one root is declared")
        return value

    def root_of(self, path: str) -> ModelDevLaneBranchInstallRoot | None:
        """The declared root a file path lies under, or None."""
        return next((r for r in self.roots if path.startswith(r.path)), None)

    @classmethod
    def from_file(cls, roots_path: Path) -> ModelDevLaneBranchInstallRoots:
        """Load the roots from a YAML file.

        Raises:
            ProtocolConfigurationError: If the file cannot supply valid roots.
        """
        try:
            with roots_path.open(encoding="utf-8") as handle:
                return cls.model_validate(yaml.safe_load(handle))
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_dev_lane_branch_install_roots",
            )
            raise ProtocolConfigurationError(
                f"Unable to load valid install roots from {roots_path.name}",
                context=context,
            ) from exc


@cache
def load_dev_lane_branch_install_roots() -> ModelDevLaneBranchInstallRoots:
    """Return the roots declared beside this module."""
    return ModelDevLaneBranchInstallRoots.from_file(_ROOTS_PATH)


__all__: list[str] = [
    "ModelDevLaneBranchInstallRoots",
    "load_dev_lane_branch_install_roots",
]
