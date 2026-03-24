# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract-driven node capability extractor for ONEX nodes.

Reads the ``node_capabilities`` block from contract YAML files and returns a
``ModelNodeCapabilities`` instance. This allows nodes to declare their
infrastructure capabilities (postgres, read, write, transactions, etc.)
declaratively in the contract — the single source of truth for node metadata.

The extractor follows the same pattern as ``ContractConfigExtractor``: it reads
raw YAML (not typed Pydantic models) because ``node_capabilities`` is an
infra-layer concern not modelled in ``omnibase_core``'s ``ModelContractBase``.

Thread Safety:
    This class is stateless and safe for concurrent use.

Example contract YAML::

    # contract.yaml
    node_capabilities:
      postgres: true
      read: true
      write: true
      transactions: true

Usage::

    extractor = ContractNodeCapabilityExtractor()
    caps = extractor.extract_from_yaml(Path("path/to/contract.yaml"))
    # caps is ModelNodeCapabilities(postgres=True, read=True, write=True, transactions=True)

.. versionadded:: 0.14.0
    Created as part of OMN-5054.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)

if TYPE_CHECKING:
    from omnibase_core.models.core.model_feature_flags import ModelFeatureFlags

logger = logging.getLogger(__name__)


class ContractNodeCapabilityExtractor:
    """Extracts ``ModelNodeCapabilities`` from contract YAML files.

    Reads the ``node_capabilities`` block from raw YAML and constructs a
    ``ModelNodeCapabilities`` instance. Unknown fields are accepted via
    ``ModelNodeCapabilities``'s ``extra="allow"`` config.

    If the contract has no ``node_capabilities`` block, returns a default
    (all-false) ``ModelNodeCapabilities``.

    Example::

        extractor = ContractNodeCapabilityExtractor()
        caps = extractor.extract_from_yaml(Path("contract.yaml"))
        assert caps.postgres is True
    """

    def __init__(self) -> None:
        self.last_validation_errors: list[str] = []

    def extract_from_yaml(self, contract_path: Path) -> ModelNodeCapabilities:
        """Extract node capabilities from a contract YAML file.

        Args:
            contract_path: Path to the contract YAML file.

        Returns:
            ``ModelNodeCapabilities`` populated from the ``node_capabilities``
            block, or a default instance if the block is absent.
        """
        try:
            raw = contract_path.read_text(encoding="utf-8")
            raw_data: object = yaml.safe_load(raw)
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "Failed to parse contract YAML for node capabilities: %s: %s",
                contract_path,
                exc,
            )
            return ModelNodeCapabilities()

        if not isinstance(raw_data, dict):
            return ModelNodeCapabilities()

        caps_block = raw_data.get("node_capabilities")
        if caps_block is None:
            return ModelNodeCapabilities()

        if not isinstance(caps_block, dict):
            logger.warning(
                "node_capabilities in %s is not a mapping (got %s), ignoring",
                contract_path,
                type(caps_block).__name__,
            )
            return ModelNodeCapabilities()

        try:
            return ModelNodeCapabilities(**caps_block)
        except (TypeError, ValidationError) as exc:
            logger.warning(
                "Failed to construct ModelNodeCapabilities from %s: %s",
                contract_path,
                exc,
            )
            return ModelNodeCapabilities()

    def extract_from_dict(self, data: dict[str, object]) -> ModelNodeCapabilities:
        """Extract node capabilities from an already-parsed YAML dict.

        Useful when the YAML has already been loaded (e.g., during runtime
        bootstrap) and you want to avoid re-reading the file.

        Args:
            data: Parsed YAML dict (top-level contract data).

        Returns:
            ``ModelNodeCapabilities`` populated from the ``node_capabilities``
            block, or a default instance if the block is absent.
        """
        caps_block = data.get("node_capabilities")
        if caps_block is None:
            return ModelNodeCapabilities()

        if not isinstance(caps_block, dict):
            return ModelNodeCapabilities()

        try:
            return ModelNodeCapabilities(**caps_block)
        except (TypeError, ValidationError) as exc:
            logger.warning(
                "Failed to construct ModelNodeCapabilities from dict: %s",
                exc,
            )
            return ModelNodeCapabilities()

    def extract_feature_flags_from_yaml(self, yaml_path: Path) -> ModelFeatureFlags:
        """Extract feature flags from a contract YAML file.

        Missing block -> empty ``ModelFeatureFlags()``.
        Invalid block -> empty ``ModelFeatureFlags()`` + structured warning +
        ``last_validation_errors`` populated.
        Valid block -> ``ModelFeatureFlags.from_contract_declarations(parsed_list)``.

        Args:
            yaml_path: Path to the contract YAML file.

        Returns:
            ``ModelFeatureFlags`` populated from the ``feature_flags`` block,
            or an empty instance if the block is absent or invalid.
        """
        from omnibase_core.models.core.model_feature_flags import ModelFeatureFlags

        self.last_validation_errors = []

        try:
            raw = yaml_path.read_text(encoding="utf-8")
            raw_data: object = yaml.safe_load(raw)
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "Failed to parse contract YAML for feature flags: %s: %s",
                yaml_path,
                exc,
            )
            return ModelFeatureFlags()

        if not isinstance(raw_data, dict):
            return ModelFeatureFlags()

        return self._parse_feature_flags_block(raw_data, str(yaml_path))

    def extract_feature_flags_from_dict(
        self, data: dict[str, object]
    ) -> ModelFeatureFlags:
        """Extract feature flags from an already-parsed contract dict.

        Args:
            data: Parsed YAML dict (top-level contract data).

        Returns:
            ``ModelFeatureFlags`` populated from the ``feature_flags`` block,
            or an empty instance if the block is absent or invalid.
        """
        self.last_validation_errors = []
        return self._parse_feature_flags_block(data, "<dict>")

    def _parse_feature_flags_block(
        self, data: dict[str, object], source: str
    ) -> ModelFeatureFlags:
        """Shared parsing logic for feature flags from a contract dict.

        Args:
            data: Top-level contract dict.
            source: Human-readable source identifier for log messages.

        Returns:
            Parsed ``ModelFeatureFlags`` or empty on error.
        """
        from omnibase_core.models.contracts.model_contract_feature_flag import (
            ModelContractFeatureFlag,
        )
        from omnibase_core.models.core.model_feature_flags import ModelFeatureFlags

        flags_block = data.get("feature_flags")
        if flags_block is None:
            return ModelFeatureFlags()

        if not isinstance(flags_block, list):
            msg = (
                f"feature_flags in {source} is not a list "
                f"(got {type(flags_block).__name__})"
            )
            logger.warning(msg)
            self.last_validation_errors.append(msg)
            return ModelFeatureFlags()

        declarations: list[ModelContractFeatureFlag] = []
        for idx, entry in enumerate(flags_block):
            if not isinstance(entry, dict):
                msg = (
                    f"feature_flags[{idx}] in {source} is not a mapping "
                    f"(got {type(entry).__name__})"
                )
                logger.warning(msg)
                self.last_validation_errors.append(msg)
                continue

            try:
                declarations.append(ModelContractFeatureFlag(**entry))
            except (TypeError, ValidationError) as exc:
                msg = f"feature_flags[{idx}] in {source} failed validation: {exc}"
                logger.warning(msg)
                self.last_validation_errors.append(msg)

        if self.last_validation_errors:
            return ModelFeatureFlags()

        return ModelFeatureFlags.from_contract_declarations(declarations)


__all__ = [
    "ContractNodeCapabilityExtractor",
]
