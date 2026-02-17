# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""LLM model pricing table with YAML manifest loading.

Loads per-model token costs from a YAML pricing manifest and provides
cost estimation for LLM calls. The pricing manifest is stored in config
(not hardcoded) and updated via PR.

Design Decisions:
    - D1: The pricing manifest is a YAML file at a well-known path inside
      the ``configs/`` directory. It is loaded once and cached as an
      immutable ``ModelPricingTable`` instance.
    - D2: Unknown models return ``estimated_cost_usd=None`` (NOT ``0``).
      This makes it explicit that cost data is unavailable, rather than
      silently reporting zero cost.
    - D3: Local models return ``estimated_cost_usd=0.0`` explicitly.
      They are listed in the manifest with ``input_cost_per_1k: 0.0``
      and ``output_cost_per_1k: 0.0``.
    - D4: Cost formula: ``(prompt_tokens / 1000 * input_cost_per_1k) +
      (completion_tokens / 1000 * output_cost_per_1k)``.
    - D5: The table is frozen (immutable) after loading. To update
      pricing, create a new table instance.

Related Tickets:
    - OMN-2239: E1-T3 Model pricing table and cost estimation

.. versionadded:: 0.10.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.models.pricing.model_cost_estimate import ModelCostEstimate
from omnibase_infra.models.pricing.model_pricing_entry import ModelPricingEntry

logger = logging.getLogger(__name__)

# Default manifest path relative to this package.
_DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "pricing_manifest.yaml"
)


class ModelPricingTable(BaseModel):
    """Immutable lookup table for LLM model pricing.

    Loads per-model token costs from a YAML manifest. Provides cost
    estimation via :meth:`estimate_cost`.

    Attributes:
        schema_version: Version of the pricing manifest schema.
        models: Mapping from model identifier to pricing entry.

    Example:
        >>> table = ModelPricingTable.from_yaml()
        >>> estimate = table.estimate_cost("claude-opus-4-6", 1000, 500)
        >>> estimate.estimated_cost_usd is not None
        True
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    schema_version: str = Field(
        ...,
        min_length=1,
        description="Version of the pricing manifest schema.",
    )
    models: dict[str, ModelPricingEntry] = Field(
        default_factory=dict,
        description="Mapping from model identifier to pricing entry.",
    )

    @model_validator(mode="after")
    def validate_non_empty_models(self) -> ModelPricingTable:
        """Warn when the pricing table has no model entries.

        An empty table is valid (may be intentional in test environments)
        but likely indicates a misconfiguration in production.

        Returns:
            Self, unchanged.
        """
        if not self.models:
            logger.warning(
                "Pricing table loaded with zero model entries. "
                "All cost estimates will return None."
            )
        return self

    def estimate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> ModelCostEstimate:
        """Estimate the USD cost for an LLM call.

        Cost formula:
            ``(prompt_tokens / 1000 * input_cost_per_1k) +
            (completion_tokens / 1000 * output_cost_per_1k)``

        Args:
            model_id: Identifier of the LLM model (e.g.
                ``"claude-opus-4-6"``, ``"qwen2.5-coder-14b"``).
            prompt_tokens: Number of input (prompt) tokens.
            completion_tokens: Number of output (completion) tokens.

        Returns:
            A :class:`ModelCostEstimate` with the computed cost, or
            ``estimated_cost_usd=None`` if the model is not in the
            pricing manifest.
        """
        entry = self.models.get(model_id)
        if entry is None:
            logger.debug(
                "Model %r not found in pricing table; returning null cost.",
                model_id,
            )
            return ModelCostEstimate(
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=None,
            )

        cost = (prompt_tokens / 1000.0) * entry.input_cost_per_1k + (
            completion_tokens / 1000.0
        ) * entry.output_cost_per_1k

        return ModelCostEstimate(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=round(cost, 10),
        )

    def has_model(self, model_id: str) -> bool:
        """Check whether a model exists in the pricing table.

        Args:
            model_id: The model identifier to look up.

        Returns:
            ``True`` if the model has a pricing entry, ``False`` otherwise.
        """
        return model_id in self.models

    def get_entry(self, model_id: str) -> ModelPricingEntry | None:
        """Look up the pricing entry for a model.

        Args:
            model_id: The model identifier to look up.

        Returns:
            The :class:`ModelPricingEntry` if found, or ``None``.
        """
        return self.models.get(model_id)

    @staticmethod
    def from_yaml(path: Path | str | None = None) -> ModelPricingTable:
        """Load a pricing table from a YAML manifest file.

        Args:
            path: Path to the YAML manifest. Defaults to the bundled
                ``configs/pricing_manifest.yaml``.

        Returns:
            A frozen :class:`ModelPricingTable` instance.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
            ValueError: If the manifest YAML is malformed or fails
                schema validation.
        """
        manifest_path = Path(path) if path is not None else _DEFAULT_MANIFEST_PATH

        if not manifest_path.exists():
            raise FileNotFoundError(f"Pricing manifest not found: {manifest_path}")

        raw_text = manifest_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)

        if not isinstance(data, dict):
            raise ValueError(
                f"Pricing manifest must be a YAML mapping, got: {type(data).__name__}"
            )

        return ModelPricingTable._from_raw_dict(data)

    @staticmethod
    # ONEX_EXCLUDE: any_type - dict[str, Any] required for raw YAML manifest data
    def from_dict(data: dict[str, Any]) -> ModelPricingTable:
        """Construct a pricing table from a raw dictionary.

        Useful for programmatic construction and testing.

        Args:
            data: Dictionary matching the pricing manifest schema.

        Returns:
            A frozen :class:`ModelPricingTable` instance.
        """
        return ModelPricingTable._from_raw_dict(data)

    @staticmethod
    # ONEX_EXCLUDE: any_type - dict[str, Any] required for raw YAML manifest data
    def _from_raw_dict(data: dict[str, Any]) -> ModelPricingTable:
        """Internal constructor from a raw manifest dictionary.

        Parses the ``models`` section into :class:`ModelPricingEntry`
        instances and validates the schema version.

        Args:
            data: Raw manifest dictionary.

        Returns:
            A validated :class:`ModelPricingTable`.

        Raises:
            ValueError: If required fields are missing or malformed.
        """
        schema_version = data.get("schema_version")
        if not schema_version:
            raise ValueError(
                "Pricing manifest missing required field: 'schema_version'"
            )

        raw_models = data.get("models", {})
        if not isinstance(raw_models, dict):
            raise ValueError(
                f"Pricing manifest 'models' must be a mapping, "
                f"got: {type(raw_models).__name__}"
            )

        entries: dict[str, ModelPricingEntry] = {}
        for model_id, entry_data in raw_models.items():
            if not isinstance(entry_data, dict):
                raise ValueError(
                    f"Pricing entry for model {model_id!r} must be a mapping, "
                    f"got: {type(entry_data).__name__}"
                )
            entries[model_id] = ModelPricingEntry(**entry_data)

        return ModelPricingTable(
            schema_version=str(schema_version),
            models=entries,
        )


__all__: list[str] = ["ModelPricingTable"]
