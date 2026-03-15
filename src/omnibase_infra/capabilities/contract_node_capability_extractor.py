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

import yaml

from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)

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
        except Exception as exc:
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
        except Exception as exc:
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
        except Exception as exc:
            logger.warning(
                "Failed to construct ModelNodeCapabilities from dict: %s",
                exc,
            )
            return ModelNodeCapabilities()


__all__ = [
    "ContractNodeCapabilityExtractor",
]
