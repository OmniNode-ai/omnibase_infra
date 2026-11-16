# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract hot-load lane overlay (OMN-21318).

Per-lane contract hot-load configuration is read from an overlay root, not
from the frozen runtime config file: the overlay names a lane and its
``ModelContractHotloadConfig`` row, so it moves with the overlays repository
instead of the workspace's runtime config. The first root holding
``<root>/runtime_contract_hotload/overlay.yaml`` wins; a root holding no such
file is skipped, so no overlay anywhere means no lane row and the caller
falls back to its default. An overlay that exists but does not validate is a
loud refusal, never a quiet skip.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.model_contract_hotload_config import (
    ModelContractHotloadConfig,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

__all__: list[str] = [
    "ModelContractHotloadLaneOverlay",
    "lane_defers_topics_to_first_request",
]

OVERLAY_ID = "runtime_contract_hotload"


class ModelContractHotloadLaneOverlay(BaseModel):
    """One lane overlay file, validated whole."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    lanes: dict[str, ModelContractHotloadConfig]

    @classmethod
    def for_lane(
        cls, environ: Mapping[str, str], lane: str
    ) -> ModelContractHotloadConfig | None:
        """The lane's config from the first overlay root holding one, or None.

        The roots, in order: the ``ONEX_SKILL_OVERLAY_ROOTS`` entries
        (``os.pathsep`` separated). The first root holding
        ``<root>/runtime_contract_hotload/overlay.yaml`` is read and validated
        whole; a root holding no such file is skipped. A file that exists but
        fails to parse or validate raises ``ProtocolConfigurationError``.
        """
        lane = lane.strip()
        if not lane:
            return None
        roots = [
            Path(entry)
            for entry in environ.get("ONEX_SKILL_OVERLAY_ROOTS", "").split(os.pathsep)
            if entry
        ]
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="contract_hotload.lane_overlay",
        )
        for root in roots:
            path = root / OVERLAY_ID / "overlay.yaml"
            if not path.is_file():
                continue
            try:
                overlay = cls.model_validate(
                    yaml.safe_load(path.read_text(encoding="utf-8"))
                )
            except (yaml.YAMLError, ValidationError) as exc:
                raise ProtocolConfigurationError(
                    f"invalid contract hot-load lane overlay: {sanitize_error_message(exc)}",
                    context=context,
                ) from exc
            return overlay.lanes.get(lane)
        return None

    @classmethod
    def row_of_this_lane(
        cls, environ: Mapping[str, str] | None = None
    ) -> ModelContractHotloadConfig | None:
        """The row of the lane this runtime declares, or None.

        The lane is ``ONEX_RUNTIME_LANE`` as the runtime lane identity resolves it
        (a registered lane, else none). A runtime that declares no registered
        lane has no row and reads the shared runtime config as before.
        """
        from omnibase_infra.runtime.health.runtime_lane_identity import (
            resolve_declared_runtime_lane,
        )

        source: Mapping[str, str] = os.environ if environ is None else environ
        return cls.for_lane(source, resolve_declared_runtime_lane(source) or "")


def lane_defers_topics_to_first_request(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """True when this lane's overlay row declares ``on_demand`` (OMN-21318).

    Start-up reads this to decide whether a demand-deferred contract's topics are
    created at boot (any other lane, as OMN-23951) or when its first request
    starts it (OMN-23953). The shared runtime config's block carries no lane, so
    only the lane's own row answers here.
    """
    row = ModelContractHotloadLaneOverlay.row_of_this_lane(environ)
    return row is not None and row.defers_topics_to_first_request
