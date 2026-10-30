# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One installed ``onex.node_package`` entry point and its package directory.

Moved out of ``scripts/create_kafka_topics.py`` (OMN-21529): that script is a
baselined file that only shrinks (OMN-20304), and the discovery it held is a
property of what an entry point names, so it lives with the model that names it.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelNodePackageEntry(BaseModel):
    """An entry point name and the directory of the package it loads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Entry point name.")
    path: Path = Field(..., description="Package directory.")

    @staticmethod
    def discover_pairs(
        *, filter_names: list[str] | None = None, lenient: bool = False
    ) -> list[tuple[str, Path]]:
        """Return [(name, nodes_dir_path)] for all ``onex.node_package`` entry points.

        Args:
            filter_names: If set, only return entry points whose ``ep.name`` is in
                this list.  Matches are case-sensitive.
            lenient: When ``True``, unloadable entry points emit a warning and are
                skipped.  When ``False`` (default), any load failure causes a hard
                exit (non-zero) because a broken entry point in the canonical path
                indicates a broken environment.

        Returns:
            Sorted list of ``(ep.name, Path)`` tuples.
        """
        eps = importlib.metadata.entry_points(group="onex.node_package")
        result: list[tuple[str, Path]] = []
        for ep in eps:
            if filter_names is not None and ep.name not in filter_names:
                continue
            try:
                pkg = ep.load()
            except (ImportError, AttributeError, OSError, SyntaxError) as exc:
                msg = f"Could not load entry point {ep.name!r}: {exc}"
                if lenient:
                    sys.stderr.write(f"WARNING: {msg} — skipping" + "\n")
                    continue
                sys.stderr.write(f"ERROR: {msg}" + "\n")
                sys.exit(1)
            pkg_path = Path(pkg.__path__[0])
            if not pkg_path.exists():
                msg = f"Entry point {ep.name!r} path does not exist: {pkg_path}"
                if lenient:
                    sys.stderr.write(f"WARNING: {msg} — skipping" + "\n")
                    continue
                sys.stderr.write(f"ERROR: {msg}" + "\n")
                sys.exit(1)
            result.append((ep.name, pkg_path))
        return sorted(result, key=lambda t: t[0])
