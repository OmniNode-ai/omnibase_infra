# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read and validate the deployment runtime.lane overlay before discovery."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from omnibase_core.enums.enum_config_overlay_key import EnumConfigOverlayKey
from omnibase_core.enums.enum_config_overlay_source import EnumConfigOverlaySource
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.config_overlay import (
    RUNTIME_LANE_ENV_VAR,
    ModelConfigOverlayDocument,
    ModelConfigOverlayScope,
    ModelRuntimeLaneDeclaration,
)
from omnibase_infra.cli.store_onex_home_files import StoreOnexHomeFiles
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.model_runtime_lane_resolution import (
    ModelRuntimeLaneResolution,
)
from omnibase_infra.models.model_runtime_lane_resolution_request import (
    ModelRuntimeLaneResolutionRequest,
)

ENV_STORE_IDENTITY = "INFISICAL_ADDR"
CONFIG_SOURCE_FIELD = "config_source"
ENV_RUNTIME_LANE = RUNTIME_LANE_ENV_VAR
ENV_RUNTIME_ENVIRONMENT = "ONEX_ENVIRONMENT"


class HandlerRuntimeLaneResolution:
    """Startup-only EFFECT handler; runs before the runtime container is wired.

    The store source remains owned by model-config B4. Selecting it refuses
    explicitly until that source is available; sources are never layered.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelRuntimeLaneResolutionRequest
    ) -> ModelRuntimeLaneResolution:
        """Resolve one startup request without publishing or discovering contracts."""
        return self.resolve(environ=request.environ, home=request.home)

    def resolve(
        self,
        *,
        environ: Mapping[str, str],
        home: Path | None = None,
    ) -> ModelRuntimeLaneResolution:
        """Resolve this runtime's lane from its overlay, or refuse naming why.

        Args:
            environ: The process environment, injected by the caller.
            home: Home directory override, for tests.

        Raises:
            ProtocolConfigurationError: the environment or lane is unset or not a
                scope segment, no overlay source (or both) is configured, the
                ``runtime.lane`` document is absent at the scope, invalid, or
                declares a different lane.
        """
        key = EnumConfigOverlayKey.RUNTIME_LANE
        environment = (environ.get(ENV_RUNTIME_ENVIRONMENT) or "").strip()
        declared_lane = environ.get(ENV_RUNTIME_LANE)
        lane = (declared_lane or "").strip()
        if not environment:
            raise ProtocolConfigurationError(
                f"{ENV_RUNTIME_ENVIRONMENT} is not set, so this runtime cannot name "
                f"the overlay scope its {key.value} document lives at and refuses to "
                "start. Set it in the deployment that runs this process."
            )
        if not self._is_slug(environment):
            raise ProtocolConfigurationError(
                f"{ENV_RUNTIME_ENVIRONMENT}={environment!r} is not an overlay scope "
                "segment (lowercase letters, digits and hyphens, at most 64); this "
                "runtime refuses to start."
            )
        if not self._is_slug(lane):
            # resolve() names an unset lane and a malformed one, each precisely,
            # before it looks at any document.
            try:
                ModelRuntimeLaneDeclaration.resolve(
                    declared_lane_id=declared_lane,
                    document=None,
                    where=f"the {key.value} overlay at environment {environment!r}",
                )
            except ModelOnexError as exc:
                raise ProtocolConfigurationError(exc.message) from exc
        scope = ModelConfigOverlayScope(environment=environment, lane=lane)
        root = self._select_local_home(environ=environ, home=home)
        path = root / scope.environment / scope.lane / f"{key.value}.json"
        document = self._read_document(path, key)
        where = f"local-home {path}"
        try:
            declaration = ModelRuntimeLaneDeclaration.resolve(
                declared_lane_id=declared_lane, document=document, where=where
            )
        except ModelOnexError as exc:
            raise ProtocolConfigurationError(exc.message) from exc
        if document is None:  # resolve() refuses a missing document; typing only
            raise ProtocolConfigurationError(f"no {key.value} document at {where}")
        return ModelRuntimeLaneResolution(
            declaration=declaration,
            scope=scope,
            source=EnumConfigOverlaySource.LOCAL_HOME,
            location=str(path),
            sha256=document.sha256,
        )

    def _is_slug(self, value: str) -> bool:
        """Whether ``value`` is one overlay scope segment."""
        try:
            ModelConfigOverlayScope(environment=value, lane=value)
        except ValidationError:
            return False
        return True

    def _select_local_home(
        self,
        *,
        environ: Mapping[str, str],
        home: Path | None = None,
    ) -> Path:
        """Return the deployment's one overlay source, or refuse naming both.

        Args:
            environ: The process environment, injected by the caller.
            home: Home directory override, for tests. ``~/.onex`` and
                ``~/.omninode/config`` are both read under it.

        Raises:
            ProtocolConfigurationError: neither source is configured, both are,
                the bootstrap file is unreadable, or the store is selected.
        """
        home_dir = home if home is not None else Path.home()
        onex_files = StoreOnexHomeFiles(home_dir / ".onex")
        try:
            bootstrap = onex_files.load_config(must_exist=False)
        except ModelOnexError as exc:
            raise ProtocolConfigurationError(
                f"cannot read the overlay source selection from {onex_files.config_path}: "
                f"{exc.message}"
            ) from exc

        store_selected = bool((environ.get(ENV_STORE_IDENTITY) or "").strip())
        local_selected = (
            bootstrap.get(CONFIG_SOURCE_FIELD)
            == EnumConfigOverlaySource.LOCAL_HOME.value
        )
        both = (
            f"the {EnumConfigOverlaySource.STORE.value} source (selected by a "
            f"non-blank {ENV_STORE_IDENTITY}) and the "
            f"{EnumConfigOverlaySource.LOCAL_HOME.value} source (selected by "
            f"'{CONFIG_SOURCE_FIELD}: {EnumConfigOverlaySource.LOCAL_HOME.value}' in "
            f"{onex_files.config_path}, documents under "
            f"{home_dir / '.omninode' / 'config'})"
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
        return home_dir / ".omninode" / "config"

    def _read_document(
        self, path: Path, key: EnumConfigOverlayKey
    ) -> ModelConfigOverlayDocument | None:
        """Return the document for ``key`` at ``scope``, or ``None`` when absent.

        Raises:
            ProtocolConfigurationError: the file is readable by group or other
                (it must be 0600, as every ``~/.onex`` file is), is not UTF-8
                JSON, is not a JSON object, or carries no ``schema_version``.
        """
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
            if not path.is_file():
                raise ProtocolConfigurationError(
                    f"overlay document {path} for {key.value} is not a regular file"
                )
            if mode & 0o077:
                raise ProtocolConfigurationError(
                    f"overlay document {path} is mode {mode:04o}; it must be 0600 "
                    f"(owner-only). Refusing to read {key.value} from it. Fix with: "
                    f"chmod 600 {path}"
                )
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ProtocolConfigurationError(
                f"cannot read overlay document {path} for {key.value}: {exc.strerror}"
            ) from exc
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
            source=EnumConfigOverlaySource.LOCAL_HOME,
            sha256=hashlib.sha256(raw).hexdigest(),
            content=content,
        )
