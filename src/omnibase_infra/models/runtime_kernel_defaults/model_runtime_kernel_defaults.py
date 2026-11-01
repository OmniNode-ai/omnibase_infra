# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared defaults for the runtime kernel (OMN-21623).

The values live in the ``settings`` block of ``runtime_kernel_defaults.yaml``.
The model has no field defaults, so a block that omits a value fails to load
rather than falling back to a literal.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_SETTINGS_FILE = "runtime_kernel_defaults.yaml"
_ENDPOINT_ENV_PREFIX = "LLM_"
_ENDPOINT_ENV_SUFFIX = "_URL"


class ModelRuntimeKernelDefaults(BaseModel):
    """Locations and endpoint variables the runtime kernel uses by default.

    Attributes:
        default_contracts_dir: Contracts directory used when the environment names none.
        workspace_contracts_dir: Runtime contracts directory of a workspace,
            relative to its config root.
        llm_endpoint_env_keys: Environment variables that name the LLM endpoints
            the kernel health-probes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    default_contracts_dir: str = Field(min_length=1)
    workspace_contracts_dir: str = Field(min_length=1)
    llm_endpoint_env_keys: tuple[str, ...] = Field(min_length=1)

    @property
    def llm_endpoint_names(self) -> dict[str, str]:
        """Each endpoint variable mapped to its logical name.

        The name is the variable name between ``LLM_`` and ``_URL``, lower-cased
        and hyphenated.
        """
        return {
            key: key.removeprefix(_ENDPOINT_ENV_PREFIX)
            .removesuffix(_ENDPOINT_ENV_SUFFIX)
            .lower()
            .replace("_", "-")
            for key in self.llm_endpoint_env_keys
        }

    @classmethod
    def from_file(cls, settings_path: Traversable) -> ModelRuntimeKernelDefaults:
        """Load the settings from the ``settings`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``settings`` mapping.
        """
        with settings_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("settings") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_runtime_kernel_defaults",
            )
            raise ProtocolConfigurationError(
                f"{settings_path.name} has no settings mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_runtime_kernel_defaults() -> ModelRuntimeKernelDefaults:
    """Return the runtime kernel defaults declared beside this model.

    The file is read through the package resource API so a wheel imported from
    a zip archive loads it the same way an extracted one does.
    """
    return ModelRuntimeKernelDefaults.from_file(
        files("omnibase_infra.models.runtime_kernel_defaults").joinpath(_SETTINGS_FILE)
    )


__all__: list[str] = [
    "ModelRuntimeKernelDefaults",
    "load_runtime_kernel_defaults",
]
