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

A missing overlay, an overlay with no ``event_bus``, a malformed block and a lane the declaration
does not bind are refused with the overlay path. Nothing falls back to a default broker.
"""

from __future__ import annotations

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


def _workspace_root_from_environment() -> Path | None:
    """``$OMNIBASE_PATH``, else ``$OMNI_HOME``: where the workspace is, never which broker."""
    for name in ("OMNIBASE_PATH", "OMNI_HOME"):
        value = expand_contract_env_refs(f"${{env.{name}}}").strip()
        if value:
            return Path(value)
    return None


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
        """Read the node's overlay and resolve it to a binding, or raise with the overlay path."""
        overlay_path = self._overlay_path(request)
        declared = self._read_event_bus(request.overlay_key, overlay_path)
        if declared.type == EnumEventBusType.INMEMORY:
            return ModelNodeBrokerBinding(
                bus_type=EnumEventBusType.INMEMORY, overlay_path=overlay_path
            )
        return self._kafka_binding(
            str(declared.lane), request.workspace_root, overlay_path
        )

    def _overlay_path(self, request: ModelNodeBrokerResolveRequest) -> Path:
        relative = Path(
            "config", "onex", "overlays", request.overlay_key, "overlay.yaml"
        )
        if request.workspace_root is None:
            raise ModelNodeBrokerResolutionError(
                f"{request.overlay_key} names no broker: no workspace root is set, so the node's "
                f"overlay (<workspace config root>/{relative}) cannot be located. Set "
                "$OMNIBASE_PATH to the workspace root.",
                overlay_path=f"<workspace config root>/{relative}",
            )
        try:
            declaration = resolve_lane_declaration_path(request.workspace_root)
        except DelegateLaneSelectionError as exc:
            raise ModelNodeBrokerResolutionError(
                f"{request.overlay_key} names no broker: the workspace config root that holds its "
                f"overlay (<workspace config root>/{relative}) cannot be located. {exc}",
                overlay_path=f"<workspace config root>/{relative}",
            ) from exc
        # The overlay and the lane declaration live under the one workspace config root.
        config_root = declaration.parents[len(LANE_DECLARATION_RELATIVE_PATH.parts) - 1]
        return config_root / relative

    def _read_event_bus(
        self, node_id: str, overlay_path: Path
    ) -> ModelNodeEventBusOverlay:
        refusal = (
            f"{node_id} declares no event_bus in its contract overlay {overlay_path}"
        )
        declaration = (
            "Declare `event_bus: {type: kafka, lane: <lane id>}` (or `{type: inmemory}`) there; "
            "the broker address stays in the lane declaration."
        )
        if not overlay_path.is_file():
            raise ModelNodeBrokerResolutionError(
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
            raise ModelNodeBrokerResolutionError(
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
