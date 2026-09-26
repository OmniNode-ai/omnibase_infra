# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The local-home config overlay source (OMN-19747, model-config plan B4).

A local install, and a runtime whose deployment mounts a local-home directory,
reads its overlay documents as JSON files keyed by scope::

    <root>/<environment>/<lane>/<key>.json

``<root>`` is ``~/.omninode/config``. The scope segments are part of the path
because one machine may address several lanes (runtime lane overlays plan
section 3.5); a flat ``<key>.json`` could hold only one lane's document.

A missing file is not an error here: :meth:`get_document` returns ``None`` and
the caller, which knows what the key is for, refuses and names the path. A
file that exists but cannot be trusted or parsed raises. There is no default
document on any branch.
"""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from omnibase_core.enums.enum_config_overlay_key import EnumConfigOverlayKey
from omnibase_core.enums.enum_config_overlay_source import EnumConfigOverlaySource
from omnibase_core.models.config_overlay import (
    ModelConfigOverlayDocument,
    ModelConfigOverlayScope,
)
from omnibase_infra.errors import ProtocolConfigurationError

__all__ = ["SourceConfigOverlayLocalHome", "default_local_home_root"]


def default_local_home_root(home: Path | None = None) -> Path:
    """Return ``~/.omninode/config`` for ``home`` (the process home by default)."""
    return (home if home is not None else Path.home()) / ".omninode" / "config"


class SourceConfigOverlayLocalHome:
    """Reads overlay documents from scope-keyed JSON files under one root."""

    source: EnumConfigOverlaySource = EnumConfigOverlaySource.LOCAL_HOME

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path_for(
        self, key: EnumConfigOverlayKey, scope: ModelConfigOverlayScope
    ) -> Path:
        """The file a document for ``key`` at ``scope`` is read from."""
        return self._root / scope.environment / scope.lane / f"{key.value}.json"

    def describe(
        self, key: EnumConfigOverlayKey, scope: ModelConfigOverlayScope
    ) -> str:
        """Name the source and the exact place a document is looked for."""
        return f"{self.source.value} {self.path_for(key, scope)}"

    def get_document(
        self, key: EnumConfigOverlayKey, scope: ModelConfigOverlayScope
    ) -> ModelConfigOverlayDocument | None:
        """Return the document for ``key`` at ``scope``, or ``None`` when absent.

        Raises:
            ProtocolConfigurationError: the file is readable by group or other
                (it must be 0600, as every ``~/.onex`` file is), is not UTF-8
                JSON, is not a JSON object, or carries no ``schema_version``.
        """
        path = self.path_for(key, scope)
        if not path.is_file():
            return None
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ProtocolConfigurationError(
                f"overlay document {path} is mode {mode:04o}; it must be 0600 "
                f"(owner-only). Refusing to read {key.value} from it. Fix with: "
                f"chmod 600 {path}"
            )
        raw = path.read_bytes()
        try:
            content = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolConfigurationError(
                f"overlay document {path} for {key.value} is not UTF-8 JSON: {exc}"
            ) from exc
        if not isinstance(content, dict):
            raise ProtocolConfigurationError(
                f"overlay document {path} for {key.value} must be a JSON object, "
                f"found {type(content).__name__}"
            )
        schema_version = content.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version:
            raise ProtocolConfigurationError(
                f"overlay document {path} for {key.value} carries no "
                "schema_version string"
            )
        return ModelConfigOverlayDocument(
            key=key,
            schema_ref=key.schema_ref,
            schema_version=schema_version,
            source=self.source,
            sha256=hashlib.sha256(raw).hexdigest(),
            content=content,
        )
