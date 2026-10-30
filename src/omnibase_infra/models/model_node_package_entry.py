# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One installed ``onex.node_package`` entry point and its package directory.

Moved out of ``scripts/create_kafka_topics.py`` (OMN-21529): that script is a
baselined file that only shrinks (OMN-20304), and the discovery it held is a
property of what an entry point names, so it lives with the model that names it.
"""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

logger = logging.getLogger(__name__)


class ModelNodePackageEntry(BaseModel):
    """An entry point name and the directory of the package it loads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Entry point name.")
    path: Path = Field(..., description="Package directory.")

    @classmethod
    def discover(
        cls, *, filter_names: list[str] | None = None, lenient: bool = False
    ) -> list[ModelNodePackageEntry]:
        """Return entries for all ``onex.node_package`` entry points, sorted by name.

        Args:
            filter_names: If set, only return entry points whose ``ep.name`` is in
                this list.  Matches are case-sensitive.
            lenient: When ``True``, unloadable entry points are logged as a
                warning and skipped.  When ``False`` (default), any load failure
                raises, because a broken entry point in the canonical path
                indicates a broken environment.

        Raises:
            ProtocolConfigurationError: In strict mode, when an entry point cannot
                be loaded or its package directory does not exist.
        """
        eps = importlib.metadata.entry_points(group="onex.node_package")
        result: list[ModelNodePackageEntry] = []
        for ep in eps:
            if filter_names is not None and ep.name not in filter_names:
                continue
            try:
                pkg = ep.load()
            except (ImportError, AttributeError, OSError, SyntaxError) as exc:
                msg = f"Could not load entry point {ep.name!r}: {exc}"
                if lenient:
                    logger.warning("%s - skipping", msg)
                    continue
                raise cls._refusal(msg) from exc
            pkg_path = Path(pkg.__path__[0])
            if not pkg_path.exists():
                msg = f"Entry point {ep.name!r} path does not exist: {pkg_path}"
                if lenient:
                    logger.warning("%s - skipping", msg)
                    continue
                raise cls._refusal(msg)
            result.append(cls(name=ep.name, path=pkg_path))
        return sorted(result, key=lambda e: e.name)

    @staticmethod
    def _refusal(message: str) -> ProtocolConfigurationError:
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="discover_node_packages",
        )
        return ProtocolConfigurationError(message, context=context)
