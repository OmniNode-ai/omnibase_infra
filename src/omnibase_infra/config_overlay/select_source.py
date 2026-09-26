# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Choose the one config overlay source a deployment reads (OMN-19747).

Model-config plan section 3.1, operator-reviewed 2026-09-23: a deployment reads
exactly one source, chosen from bootstrap state that already exists, and the two
sources are never layered. No new environment variable is read.

- ``store``: the process carries the config-store bootstrap identity
  (``INFISICAL_ADDR`` set and non-blank).
- ``local-home``: the ``~/.onex/config.yaml`` bootstrap file says
  ``config_source: local-home`` (written by ``onex local init``, or mounted by a
  deployment that supplies its overlay as files).

Both present, or neither, refuses and names both. The store half of B4 is not
in this build: a deployment that selects it is refused with that fact named,
never read through a fallback.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from omnibase_core.enums.enum_config_overlay_source import EnumConfigOverlaySource
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.store_onex_home_files import StoreOnexHomeFiles
from omnibase_infra.config_overlay.source_local_home import (
    SourceConfigOverlayLocalHome,
    default_local_home_root,
)
from omnibase_infra.errors import ProtocolConfigurationError

__all__ = [
    "CONFIG_SOURCE_FIELD",
    "ENV_STORE_IDENTITY",
    "select_config_overlay_source",
]

#: The bootstrap variable that marks a process as holding the store identity.
ENV_STORE_IDENTITY = "INFISICAL_ADDR"

#: The ``~/.onex/config.yaml`` field that selects the local-home source.
CONFIG_SOURCE_FIELD = "config_source"


def select_config_overlay_source(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> SourceConfigOverlayLocalHome:
    """Return the deployment's one overlay source, or refuse naming both.

    Args:
        environ: Process environment override, for tests.
        home: Home directory override, for tests. ``~/.onex`` and
            ``~/.omninode/config`` are both read under it.

    Raises:
        ProtocolConfigurationError: neither source is configured, both are,
            the bootstrap file is unreadable, or the store is selected.
    """
    env: Mapping[str, str] = os.environ if environ is None else environ
    home_dir = home if home is not None else Path.home()
    onex_files = StoreOnexHomeFiles(home_dir / ".onex")
    try:
        bootstrap = onex_files.load_config(must_exist=False)
    except ModelOnexError as exc:
        raise ProtocolConfigurationError(
            f"cannot read the overlay source selection from {onex_files.config_path}: "
            f"{exc.message}"
        ) from exc

    store_selected = bool((env.get(ENV_STORE_IDENTITY) or "").strip())
    local_selected = (
        bootstrap.get(CONFIG_SOURCE_FIELD) == EnumConfigOverlaySource.LOCAL_HOME.value
    )
    both = (
        f"the {EnumConfigOverlaySource.STORE.value} source (selected by a "
        f"non-blank {ENV_STORE_IDENTITY}) and the "
        f"{EnumConfigOverlaySource.LOCAL_HOME.value} source (selected by "
        f"'{CONFIG_SOURCE_FIELD}: {EnumConfigOverlaySource.LOCAL_HOME.value}' in "
        f"{onex_files.config_path}, documents under "
        f"{default_local_home_root(home_dir)})"
    )
    if store_selected and local_selected:
        raise ProtocolConfigurationError(
            f"both config overlay sources are configured: {both}. A deployment "
            "reads exactly one; the two are never layered. Remove one."
        )
    if not store_selected and not local_selected:
        raise ProtocolConfigurationError(
            f"no config overlay source is configured. Configure exactly one of {both}."
        )
    if store_selected:
        raise ProtocolConfigurationError(
            f"this deployment selects the {EnumConfigOverlaySource.STORE.value} "
            f"config overlay source ({ENV_STORE_IDENTITY} is set), and this build "
            "reads overlay documents only from "
            f"{EnumConfigOverlaySource.LOCAL_HOME.value}: the store source is the "
            "other half of model-config plan task B4 and is not built. Supply "
            "the documents through local-home instead, or run a build that has "
            "the store source."
        )
    return SourceConfigOverlayLocalHome(default_local_home_root(home_dir))
