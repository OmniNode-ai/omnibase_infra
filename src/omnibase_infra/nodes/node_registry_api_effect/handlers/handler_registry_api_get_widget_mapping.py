# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: get_widget_mapping.

Loads and returns the widget mapping configuration from YAML.

Ticket: OMN-4482
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiResponse,
)

logger = logging.getLogger(__name__)

__all__ = ["HandlerRegistryApiGetWidgetMapping"]

# Default path: src/omnibase_infra/configs/widget_mapping.yaml
_DEFAULT_WIDGET_MAPPING_PATH = (
    Path(__file__).parent.parent.parent.parent / "configs" / "widget_mapping.yaml"
)


class HandlerRegistryApiGetWidgetMapping:
    """Handler for operation: get_widget_mapping.

    Returns capability-to-widget mapping configuration parsed from YAML.

    Attributes:
        _mapping_path: Path to the widget_mapping.yaml file.
    """

    def __init__(self, mapping_path: Path | None = None) -> None:
        """Initialise the handler with an optional custom mapping path.

        Args:
            mapping_path: Path to the widget_mapping.yaml file. Defaults to
                the canonical src/omnibase_infra/configs/widget_mapping.yaml.
        """
        self._mapping_path = mapping_path or _DEFAULT_WIDGET_MAPPING_PATH

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (external I/O)."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: object, correlation_id: UUID) -> object:
        """Handle get_widget_mapping operation.

        Reads and parses the widget_mapping.yaml file. Returns an error
        response if the file is missing or malformed rather than raising.

        Args:
            request: Ignored for this operation.
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with parsed widget mapping in ``data``.
        """
        try:
            with self._mapping_path.open() as fh:
                mapping = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning(
                "HandlerRegistryApiGetWidgetMapping: mapping file not found at %s",
                self._mapping_path,
            )
            return ModelRegistryApiResponse(
                operation="get_widget_mapping",
                correlation_id=correlation_id,
                success=False,
                error=f"Widget mapping file not found: {self._mapping_path}",
            )
        except yaml.YAMLError as exc:
            logger.warning(
                "HandlerRegistryApiGetWidgetMapping: YAML parse error — %s", exc
            )
            return ModelRegistryApiResponse(
                operation="get_widget_mapping",
                correlation_id=correlation_id,
                success=False,
                error=f"Widget mapping YAML parse error: {exc}",
            )

        logger.debug(
            "HandlerRegistryApiGetWidgetMapping: loaded %d top-level keys",
            len(mapping) if isinstance(mapping, dict) else 0,
        )
        return ModelRegistryApiResponse(
            operation="get_widget_mapping",
            correlation_id=correlation_id,
            success=True,
            data={"widget_mapping": mapping},
        )
