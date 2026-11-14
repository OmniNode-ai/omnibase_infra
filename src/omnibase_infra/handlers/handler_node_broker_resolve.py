# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve the broker a node is reached on from its contract overlay (OMN-22362).

Operator ruling 2026-11-03: "The default broker should be in the contract overlay." ``onex
run-node`` (omnibase_core) cannot read the lab lane declaration, so it asks the resolver this
package registers under ``onex.node_broker_resolvers`` and reads no environment variable for the
broker. The node's ``event_bus`` is read from
``<workspace config root>/config/onex/overlays/<node>/overlay.yaml``, the deployment overlay the
node's other private facts already live in:

* ``type: kafka`` with a ``lane`` resolves the lane to its declared broker and client transport
  with the same readers ``onex delegate`` uses (the lane declaration, then the machine's stored
  lane identity, then the SASL environment a lab host or CI runner holds);
* ``type: inmemory`` hosts the node in the caller's process and carries no transport.

The overlay is read from the roots ``ONEX_SKILL_OVERLAY_ROOTS`` declares first (the one overlay root
resolver renders that list: store root, canonical overlays clone, declared roots), then from the
workspace config root. When the node's own overlay is missing or has no ``event_bus`` key, the
resolver reads one shared overlay named ``run_node_default`` (the same lookup order: the overlay
roots first, then the workspace config root) and binds the node on the lane that overlay declares;
the binding's overlay path is the default file that answered. A node's own ``event_bus`` always
wins, and a node overlay whose ``event_bus`` is malformed is refused as before with its own path
and never falls back. When the default overlay is missing or has no ``event_bus`` either, the
original refusal for the node stands, naming its own overlay path. A malformed block and a lane the
declaration does not bind are refused with the overlay path that named them. No environment
variable or default broker answers for a lane; it comes from an overlay file alone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import yaml
from pydantic import SecretStr, ValidationError

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_core.models.event_bus.model_node_broker_binding import (
    ModelNodeBrokerBinding,
)
from omnibase_core.models.event_bus.model_node_broker_resolution_error import (
    ModelNodeBrokerResolutionError,
)
from omnibase_infra.cli.delegate_lane import (
    LANE_DECLARATION_RELATIVE_PATH,
    DelegateLaneSelectionError,
    resolve_lane_declaration_path,
)
from omnibase_infra.cli.delegate_lane_credentials import (
    ENV_SASL_PASSWORD,
    ENV_SASL_USERNAME,
    DelegateLaneCredentialError,
    resolve_lane_binding,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.workspace import (
    ModelNodeBrokerResolveRequest,
    ModelNodeEventBusOverlay,
)
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs

_EVENT_BUS_KEY: Final[str] = "event_bus"
_KIND_NAMES: Final[tuple[str, str]] = ("kafka", "inmemory")
_OVERLAY_ROOTS_ENV: Final[str] = "ONEX_SKILL_OVERLAY_ROOTS"
_DEFAULT_OVERLAY_KEY: Final[str] = "run_node_default"


class _MissingOrNoKeyOverlayError(ModelNodeBrokerResolutionError):
    """The overlay file does not exist, or it holds no ``event_bus`` key.

    ``_read_event_bus`` raises this for those two cases only; a node whose own overlay
    did not answer may be reached through the ``run_node_default`` overlay, while every
    other refusal (malformed ``event_bus``, an unreadable or non-mapping file) propagates
    untouched.
    """


def _workspace_root_from_environment() -> Path | None:
    """``$OMNIBASE_PATH``, else ``$OMNI_HOME``: where the workspace is, never which broker."""
    for name in ("OMNIBASE_PATH", "OMNI_HOME"):
        value = expand_contract_env_refs(f"${{env.{name}}}").strip()
        if value:
            return Path(value)
    return None


def _declared_overlay_roots() -> tuple[str, ...]:
    """The roots ``ONEX_SKILL_OVERLAY_ROOTS`` declares, in order: the selector every overlay reader takes.

    The one overlay root resolver (omnimarket) is what renders that variable for a unit or a lane
    (the overlay store root, then the canonical overlays clone, then any declared root), so this
    handler reads the selector and builds no list of its own.
    """
    raw = expand_contract_env_refs(f"${{env.{_OVERLAY_ROOTS_ENV}}}")
    return tuple(
        str(Path(entry).expanduser())
        for entry in raw.split(os.pathsep)
        if entry.strip()
    )


class HandlerNodeBrokerResolve:
    """Answers ``ProtocolNodeBrokerResolver`` for ``onex run-node``."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Classify this infrastructure handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Reads the overlay, the lane declaration and the credential store."""
        return EnumHandlerTypeCategory.EFFECT

    def resolve_node_broker(self, node_id: str) -> ModelNodeBrokerBinding:
        """The protocol entry: the workspace root comes from the operator's workspace variables."""
        return self.handle(
            ModelNodeBrokerResolveRequest(
                overlay_key=node_id, workspace_root=_workspace_root_from_environment()
            )
        )

    def handle(self, request: ModelNodeBrokerResolveRequest) -> ModelNodeBrokerBinding:
        """Read the node's overlay and resolve it to a binding, or raise with the overlay path.

        When the node's own overlay is missing or has no ``event_bus`` key, one shared
        overlay (``run_node_default``) is read with the same lookup order and the node
        is bound on the lane it declares; the binding's overlay path is that file.
        """
        return self._handle_overlay_key(request, request.overlay_key)

    def _handle_overlay_key(
        self, request: ModelNodeBrokerResolveRequest, overlay_key: str
    ) -> ModelNodeBrokerBinding:
        overlay_path = self._locate_overlay(overlay_key, request.workspace_root)
        try:
            declared = self._read_event_bus(overlay_key, overlay_path)
        except _MissingOrNoKeyOverlayError as node_refusal:
            # The node's own overlay did not answer; one shared overlay may. If the
            # default overlay does not answer either, the node's original refusal stands.
            default_path = self._locate_overlay(
                _DEFAULT_OVERLAY_KEY, request.workspace_root
            )
            try:
                declared = self._read_event_bus(_DEFAULT_OVERLAY_KEY, default_path)
                overlay_path = default_path
            except _MissingOrNoKeyOverlayError:
                raise node_refusal
        if declared.type == EnumEventBusType.INMEMORY:
            return ModelNodeBrokerBinding(
                bus_type=EnumEventBusType.INMEMORY, overlay_path=overlay_path
            )
        return self._kafka_binding(
            str(declared.lane), request.workspace_root, overlay_path
        )

    def _locate_overlay(self, overlay_key: str, workspace_root: Path | None) -> Path:
        """The overlay for ``overlay_key``: the overlay roots first, else the workspace root."""
        for root in _declared_overlay_roots():
            candidate = Path(root) / overlay_key / "overlay.yaml"
            if candidate.is_file():
                return candidate
        return self._workspace_overlay_path(overlay_key, workspace_root)

    def _workspace_overlay_path(
        self, overlay_key: str, workspace_root: Path | None
    ) -> Path:
        relative = Path("config", "onex", "overlays", overlay_key, "overlay.yaml")
        if workspace_root is None:
            raise ModelNodeBrokerResolutionError(
                f"{overlay_key} names no broker: no workspace root is set, so the node's "
                f"overlay (<workspace config root>/{relative}) cannot be located. Set "
                "$OMNIBASE_PATH to the workspace root.",
                overlay_path=f"<workspace config root>/{relative}",
            )
        try:
            declaration = resolve_lane_declaration_path(workspace_root)
        except DelegateLaneSelectionError as exc:
            raise ModelNodeBrokerResolutionError(
                f"{overlay_key} names no broker: the workspace config root that holds its "
                f"overlay (<workspace config root>/{relative}) cannot be located. {exc}",
                overlay_path=f"<workspace config root>/{relative}",
            ) from exc
        # The overlay and the lane declaration live under the one workspace config root.
        config_root = declaration.parents[len(LANE_DECLARATION_RELATIVE_PATH.parts) - 1]
        return config_root / relative

    def _read_event_bus(
        self, node_id: str, overlay_path: Path
    ) -> ModelNodeEventBusOverlay:
        """The overlay's ``event_bus``; ``_MissingOrNoKeyOverlayError`` for a
        missing file or key only."""
        refusal = (
            f"{node_id} declares no event_bus in its contract overlay {overlay_path}"
        )
        declaration = (
            "Declare `event_bus: {type: kafka, lane: <lane id>}` (or `{type: inmemory}`) there; "
            "the broker address stays in the lane declaration."
        )
        if not overlay_path.is_file():
            raise _MissingOrNoKeyOverlayError(
                f"{refusal}: the file does not exist. {declaration}",
                overlay_path=str(overlay_path),
            )
        try:
            raw: object = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ModelNodeBrokerResolutionError(
                f"node overlay {overlay_path} is unreadable ({type(exc).__name__})",
                overlay_path=str(overlay_path),
            ) from None
        if not isinstance(raw, dict):
            raise ModelNodeBrokerResolutionError(
                f"node overlay {overlay_path} must be a mapping",
                overlay_path=str(overlay_path),
            )
        if _EVENT_BUS_KEY not in raw:
            raise _MissingOrNoKeyOverlayError(
                f"{refusal}: the file has no event_bus key. {declaration}",
                overlay_path=str(overlay_path),
            )
        try:
            return ModelNodeEventBusOverlay.model_validate(raw[_EVENT_BUS_KEY])
        except ValidationError as exc:
            fields = sorted(
                ".".join(str(part) for part in err["loc"]) or _EVENT_BUS_KEY
                for err in exc.errors(include_input=False, include_context=False)
            )
            raise ModelNodeBrokerResolutionError(
                f"node overlay {overlay_path} has an invalid {_EVENT_BUS_KEY}: fields {fields}; "
                f"type is one of {', '.join(_KIND_NAMES)}",
                overlay_path=str(overlay_path),
            ) from None

    def _kafka_binding(
        self, lane: str, workspace_root: Path | None, overlay_path: Path
    ) -> ModelNodeBrokerBinding:
        try:
            selection, transport = resolve_lane_binding(
                lane,
                environ={
                    "OMNIBASE_PATH": str(workspace_root or ""),
                    ENV_SASL_USERNAME: expand_contract_env_refs(
                        f"${{env.{ENV_SASL_USERNAME}}}"
                    ),
                    ENV_SASL_PASSWORD: expand_contract_env_refs(
                        f"${{env.{ENV_SASL_PASSWORD}}}"
                    ),
                },
            )
        except (DelegateLaneSelectionError, DelegateLaneCredentialError) as exc:
            raise ModelNodeBrokerResolutionError(
                f"the lane {lane!r} named by {overlay_path} is not usable: {exc}",
                overlay_path=str(overlay_path),
            ) from exc
        if transport is not None:
            return ModelNodeBrokerBinding(
                bus_type=EnumEventBusType.KAFKA,
                overlay_path=overlay_path,
                lane=lane,
                bootstrap_servers=transport.bootstrap_servers,
                security_protocol=transport.security_protocol,
                sasl_mechanism=transport.sasl_mechanism,
                sasl_username=transport.sasl_username,
                sasl_password=transport.sasl_password,
            )
        # The lab host and CI runner hold the SASL environment, not a client store: the
        # credential resolver stepped aside and that environment answers (OMN-18432).
        return ModelNodeBrokerBinding(
            bus_type=EnumEventBusType.KAFKA,
            overlay_path=overlay_path,
            lane=lane,
            bootstrap_servers=selection.bootstrap_servers,
            security_protocol=selection.security_protocol,
            sasl_mechanism=selection.sasl_mechanism,
            sasl_username=expand_contract_env_refs(f"${{env.{ENV_SASL_USERNAME}}}"),
            sasl_password=SecretStr(
                expand_contract_env_refs(f"${{env.{ENV_SASL_PASSWORD}}}")
            ),
        )


__all__: list[str] = [
    "HandlerNodeBrokerResolve",
]
