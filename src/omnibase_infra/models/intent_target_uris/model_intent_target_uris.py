# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared base URIs of reducer and projection intent targets (OMN-21792).

The values live in the ``intent_targets`` block of
``models/intent_target_uris/intent_target_uris.yaml``. The model has no field
defaults, so a block that omits a value fails to load rather than falling back
to a literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_INTENT_TARGETS_PATH = Path(__file__).resolve().parent / "intent_target_uris.yaml"


class ModelIntentTargetUris(BaseModel):
    """Base URIs of the intent targets that reducers and projections emit.

    Attributes:
        build_loop_runs: Base of the build loop run intent targets.
        contracts: Base of the contract registry intent targets.
        topics: Base of the topic registry intent targets.
        gateway_link_health: Base of the gateway link health intent targets.
        event_ledger: Base of the event ledger intent targets.
        node_registrations: Base of the node registration intent targets.
        validation_event_ledger: Base of the validation event ledger intent targets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    build_loop_runs: str = Field(min_length=1)
    contracts: str = Field(min_length=1)
    topics: str = Field(min_length=1)
    gateway_link_health: str = Field(min_length=1)
    event_ledger: str = Field(min_length=1)
    node_registrations: str = Field(min_length=1)
    validation_event_ledger: str = Field(min_length=1)

    @staticmethod
    def target(base: str, *segments: object) -> str:
        """Return ``base`` followed by each segment, joined with ``/``."""
        return "/".join((base, *(str(segment) for segment in segments)))

    @classmethod
    def from_file(cls, uris_path: Path) -> ModelIntentTargetUris:
        """Load the base URIs from the ``intent_targets`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``intent_targets`` mapping.
        """
        with uris_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("intent_targets") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_intent_target_uris",
            )
            raise ProtocolConfigurationError(
                f"{uris_path.name} has no intent_targets mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_intent_target_uris() -> ModelIntentTargetUris:
    """Return the base URIs declared in ``intent_target_uris.yaml``."""
    return ModelIntentTargetUris.from_file(_INTENT_TARGETS_PATH)


__all__: list[str] = ["ModelIntentTargetUris", "load_intent_target_uris"]
