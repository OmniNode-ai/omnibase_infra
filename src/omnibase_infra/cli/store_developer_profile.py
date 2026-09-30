# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Per-developer profile block in ``~/.onex/config.yaml``.

This is the per-developer tier of the same config authority that
``onex delegate`` resolves its transport from — not a separate CLI ladder.
It never holds a secret; the identity for the bound lane stays in the
``lanes:`` block written by ``onex auth lane-login``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.store_onex_home_files import StoreOnexHomeFiles

__all__ = ["StoreDeveloperProfile"]

_DEVELOPER_KEY: Final[str] = "developer"
_LANE_BINDING_KEY: Final[str] = "lane_binding"


class StoreDeveloperProfile:
    """Read/write ``developer.lane_binding`` in the shared onex home config."""

    def __init__(self, *, onex_home: Path) -> None:
        self._files = StoreOnexHomeFiles(onex_home)

    @property
    def config_path(self) -> Path:
        return self._files.config_path

    def lane_binding(self) -> str | None:
        """Return the bound lane, or None if unset."""
        developer = self._developer_block(self._files.load_config(must_exist=False))
        value = developer.get(_LANE_BINDING_KEY)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ModelOnexError(
                f"'{_DEVELOPER_KEY}.{_LANE_BINDING_KEY}' in {self.config_path} "
                f"must be a non-empty string; fix it with "
                f"'onex profile bind-lane <lane>' or 'onex profile unbind-lane'.",
                error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
            )
        return value.strip()

    def bind_lane(self, lane: str) -> None:
        """Set ``developer.lane_binding``, preserving all other keys."""
        lane = lane.strip()
        if not lane:
            raise ModelOnexError(
                "Cannot bind an empty lane; use 'onex profile bind-lane <lane>'.",
                error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
            )
        document = self._files.load_config(must_exist=False)
        developer = self._developer_block(document)
        developer[_LANE_BINDING_KEY] = lane
        document[_DEVELOPER_KEY] = developer
        self._files.write_config(document)

    def unbind_lane(self) -> bool:
        """Remove ``developer.lane_binding``; return True if anything was removed."""
        document = self._files.load_config(must_exist=False)
        developer = self._developer_block(document)
        if _LANE_BINDING_KEY not in developer:
            return False
        del developer[_LANE_BINDING_KEY]
        if developer:
            document[_DEVELOPER_KEY] = developer
        else:
            del document[_DEVELOPER_KEY]
        self._files.write_config(document)
        return True

    def _developer_block(self, document: dict[str, object]) -> dict[str, object]:
        """The ``developer`` mapping (a copy), or an empty one when absent.

        A non-mapping value is refused rather than replaced: binding a lane
        must never be a way to lose whatever another writer put there.
        """
        developer = document.get(_DEVELOPER_KEY)
        if developer is None:
            return {}
        if not isinstance(developer, dict):
            raise ModelOnexError(
                f"'{_DEVELOPER_KEY}' in {self.config_path} must be a mapping; "
                f"fix it by hand, then use 'onex profile bind-lane <lane>' or "
                f"'onex profile unbind-lane'.",
                error_code=EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR,
            )
        return {str(key): value for key, value in developer.items()}
