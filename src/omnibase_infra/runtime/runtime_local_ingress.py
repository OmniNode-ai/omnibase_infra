# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unix-socket transport for runtime-owned local command ingress."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import stat
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from omnibase_core.types import JsonType
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.models.model_local_runtime_ingress_error import (
    ModelLocalRuntimeIngressError,
)
from omnibase_infra.runtime.models.model_local_runtime_ingress_request import (
    ModelLocalRuntimeIngressRequest,
)
from omnibase_infra.runtime.models.model_local_runtime_ingress_response import (
    ModelLocalRuntimeIngressResponse,
)

logger = logging.getLogger(__name__)


def _preferred_request_name(raw: object) -> str:
    if not isinstance(raw, dict):
        return "unknown"
    for key in ("command_name", "node_alias"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


class ModelRuntimeLocalIngressRoute(BaseModel):
    """Resolved route for a node exposed through the local runtime ingress."""

    model_config = ConfigDict(frozen=True)

    node_name: str  # pattern-ok: structural route identifier, not an entity name
    contract_name: str  # pattern-ok: contract file identifier, not an entity name
    command_topic: str
    event_type: str | None
    terminal_event: str | None
    contract_path: str
    package_name: str  # pattern-ok: Python package identifier, not an entity name
    terminal_events: tuple[str, ...] = ()
    input_model_module: str | None = None
    input_model_name: str | None = None


def parse_active_runtime_packages(
    configured_packages: Sequence[str],
    *,
    env: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Resolve active runtime packages, honoring ONEX_ACTIVE_RUNTIME_PACKAGES."""

    env_map = os.environ if env is None else env
    raw = env_map.get("ONEX_ACTIVE_RUNTIME_PACKAGES", "")
    if raw.strip():
        resolved = tuple(part.strip() for part in raw.split(",") if part.strip())
        if resolved:
            return resolved

    normalized = tuple(part.strip() for part in configured_packages if part.strip())
    if not normalized:
        raise ValueError("No runtime packages configured for local ingress")
    return normalized


def discover_runtime_local_ingress_routes(
    package_names: Sequence[str],
) -> dict[str, ModelRuntimeLocalIngressRoute]:
    """Discover local-ingress routes from installed package node contracts."""

    routes: dict[str, ModelRuntimeLocalIngressRoute] = {}
    alias_sources: dict[str, str] = {}
    ambiguous_public_aliases: set[str] = set()

    for package_name in package_names:
        package_root = _resolve_package_root(package_name)
        nodes_dir = package_root / "nodes"
        if not nodes_dir.is_dir():
            continue

        for contract_path in sorted(nodes_dir.glob("*/contract.yaml")):
            try:
                raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                logger.warning(
                    "Skipping malformed local ingress contract",
                    extra={"contract_path": str(contract_path), "error": str(exc)},
                )
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping unreadable local ingress contract",
                    extra={"contract_path": str(contract_path), "error": str(exc)},
                    exc_info=True,
                )
                continue
            if not isinstance(raw, dict):
                continue

            contract_name = str(raw.get("name", "")).strip()
            if not contract_name:
                continue

            event_bus_section = raw.get("event_bus")
            if not isinstance(event_bus_section, dict):
                continue
            subscribe_topics = event_bus_section.get("subscribe_topics")
            if not isinstance(subscribe_topics, list):
                continue

            command_topic = _select_command_topic(subscribe_topics)
            if command_topic is None:
                continue

            node_dir_name = contract_path.parent.name
            input_model_module, input_model_name = _extract_input_model_ref(raw)
            route = ModelRuntimeLocalIngressRoute(
                node_name=node_dir_name,
                contract_name=contract_name,
                command_topic=command_topic,
                event_type=_derive_route_event_type(raw, command_topic),
                terminal_event=_safe_optional_string(raw.get("terminal_event")),
                terminal_events=_extract_terminal_events(raw),
                contract_path=str(contract_path),
                package_name=package_name,
                input_model_module=input_model_module,
                input_model_name=input_model_name,
            )

            for alias in _package_scoped_route_aliases(route):
                _register_local_ingress_route_alias(
                    routes,
                    alias_sources,
                    alias,
                    route,
                    source="package_scoped",
                    ambiguous_public_aliases=ambiguous_public_aliases,
                )

            for alias in tuple(dict.fromkeys((contract_name, node_dir_name))):
                was_ambiguous = _register_local_ingress_route_alias(
                    routes,
                    alias_sources,
                    alias,
                    route,
                    source="base",
                    ambiguous_public_aliases=ambiguous_public_aliases,
                    allow_ambiguous_public_alias=True,
                )
                if was_ambiguous:
                    logger.warning(
                        "Omitting ambiguous public local ingress route alias",
                        extra={
                            "alias": alias,
                            "contract_path": route.contract_path,
                        },
                    )

            for operation_alias, operation_route in _extract_handler_operation_routes(
                raw, route
            ):
                for package_alias in _package_scoped_operation_aliases(
                    operation_route, operation_alias
                ):
                    _register_local_ingress_route_alias(
                        routes,
                        alias_sources,
                        package_alias,
                        operation_route,
                        source="package_scoped_operation",
                        ambiguous_public_aliases=ambiguous_public_aliases,
                    )

                for qualified_alias in _qualified_operation_aliases(
                    operation_route, operation_alias
                ):
                    was_ambiguous = _register_local_ingress_route_alias(
                        routes,
                        alias_sources,
                        qualified_alias,
                        operation_route,
                        source="operation_qualified",
                        ambiguous_public_aliases=ambiguous_public_aliases,
                        allow_ambiguous_public_alias=True,
                    )
                    if was_ambiguous:
                        logger.warning(
                            "Omitting ambiguous public local ingress operation alias",
                            extra={
                                "alias": qualified_alias,
                                "contract_path": route.contract_path,
                            },
                        )

                if (
                    "." in operation_alias
                    or operation_alias in ambiguous_public_aliases
                ):
                    continue

                was_ambiguous = _register_local_ingress_route_alias(
                    routes,
                    alias_sources,
                    operation_alias,
                    operation_route,
                    source="operation_raw",
                    ambiguous_public_aliases=ambiguous_public_aliases,
                    allow_ambiguous_public_alias=True,
                )
                if was_ambiguous:
                    logger.warning(
                        "Omitting ambiguous unqualified local ingress operation alias",
                        extra={
                            "alias": operation_alias,
                            "contract_path": route.contract_path,
                        },
                    )

    return routes


def _register_local_ingress_route_alias(
    routes: dict[str, ModelRuntimeLocalIngressRoute],
    alias_sources: dict[str, str],
    alias: str,
    route: ModelRuntimeLocalIngressRoute,
    *,
    source: str,
    ambiguous_public_aliases: set[str] | None = None,
    allow_ambiguous_public_alias: bool = False,
) -> bool:
    """Register a local ingress alias, returning True when it is ambiguous."""

    if ambiguous_public_aliases is not None and alias in ambiguous_public_aliases:
        return True

    existing = routes.get(alias)
    if existing is None:
        routes[alias] = route
        alias_sources[alias] = source
        return False

    if _local_ingress_routes_equivalent(existing, route):
        logger.info(
            "Ignoring duplicate local ingress route alias with matching interface",
            extra={
                "alias": alias,
                "kept_contract_path": existing.contract_path,
                "ignored_contract_path": route.contract_path,
            },
        )
        return False

    existing_source = alias_sources.get(alias)
    if (
        allow_ambiguous_public_alias
        and _is_public_alias_source(source)
        and _is_public_alias_source(existing_source)
        and ambiguous_public_aliases is not None
    ):
        ambiguous_public_aliases.add(alias)
        routes.pop(alias, None)
        alias_sources.pop(alias, None)
        logger.warning(
            "Removed ambiguous public local ingress alias",
            extra={
                "alias": alias,
                "first_source": existing_source,
                "second_source": source,
                "first_contract_path": existing.contract_path,
                "second_contract_path": route.contract_path,
            },
        )
        return True

    raise ValueError(
        f"Duplicate local ingress route alias '{alias}' for "
        f"{existing.contract_path} and {route.contract_path}"
    )


def _is_public_alias_source(source: str | None) -> bool:
    return source in {"base", "operation_qualified", "operation_raw"}


def _local_ingress_routes_equivalent(
    left: ModelRuntimeLocalIngressRoute,
    right: ModelRuntimeLocalIngressRoute,
) -> bool:
    """Return whether two routes expose the same local-ingress interface."""
    return (
        left.node_name == right.node_name
        and left.contract_name == right.contract_name
        and left.command_topic == right.command_topic
        and left.event_type == right.event_type
        and left.terminal_event == right.terminal_event
        and left.terminal_events == right.terminal_events
        and left.input_model_module == right.input_model_module
        and left.input_model_name == right.input_model_name
    )


def _extract_terminal_events(raw: dict[object, object]) -> tuple[str, ...]:
    """Return all contract-declared terminal topics for local ingress waits."""

    terminal_events: list[str] = []
    terminal_event = _safe_optional_string(raw.get("terminal_event"))
    if terminal_event is not None:
        terminal_events.append(terminal_event)

    raw_terminal_events = raw.get("terminal_events")
    if isinstance(raw_terminal_events, dict):
        values: Iterable[object] = raw_terminal_events.values()
    elif isinstance(raw_terminal_events, list | tuple):
        values = raw_terminal_events
    else:
        values = ()

    for value in values:
        topic = _safe_optional_string(value)
        if topic is not None:
            terminal_events.append(topic)

    return tuple(dict.fromkeys(terminal_events))


def _package_scoped_route_aliases(
    route: ModelRuntimeLocalIngressRoute,
) -> tuple[str, ...]:
    """Return package-scoped aliases that stay deterministic across repos."""

    aliases = (
        f"{route.package_name}.{route.contract_name}",
        f"{route.package_name}.{route.node_name}",
    )
    return tuple(dict.fromkeys(aliases))


def _package_scoped_operation_aliases(
    route: ModelRuntimeLocalIngressRoute,
    operation_alias: str,
) -> tuple[str, ...]:
    """Return package-scoped operation aliases for colliding public names."""

    aliases = (
        f"{route.package_name}.{route.contract_name}.{operation_alias}",
        f"{route.package_name}.{route.node_name}.{operation_alias}",
    )
    return tuple(dict.fromkeys(aliases))


def _qualified_operation_aliases(
    route: ModelRuntimeLocalIngressRoute,
    operation_alias: str,
) -> tuple[str, ...]:
    """Return deterministic qualified aliases for a handler operation."""

    if "." in operation_alias:
        return (operation_alias,)

    aliases = (
        f"{route.contract_name}.{operation_alias}",
        f"{route.node_name}.{operation_alias}",
    )
    return tuple(dict.fromkeys(aliases))


def _extract_handler_operation_aliases(raw: dict[object, object]) -> tuple[str, ...]:
    """Return handler operation names that can act as local ingress aliases."""
    return tuple(
        dict.fromkeys(
            alias for alias, _route in _extract_handler_operation_routes(raw, None)
        )
    )


def _extract_handler_operation_routes(
    raw: dict[object, object],
    base_route: ModelRuntimeLocalIngressRoute | None,
) -> tuple[tuple[str, ModelRuntimeLocalIngressRoute], ...]:
    """Return handler operation aliases with handler-specific route metadata."""
    handler_routing = raw.get("handler_routing")
    if not isinstance(handler_routing, dict):
        return ()

    handlers = handler_routing.get("handlers")
    if not isinstance(handlers, list):
        return ()

    aliases: list[tuple[str, ModelRuntimeLocalIngressRoute]] = []
    for handler in handlers:
        if not isinstance(handler, dict):
            continue
        operation = handler.get("operation")
        if not isinstance(operation, str):
            continue
        normalized = operation.strip()
        if not normalized:
            continue
        route = base_route
        if route is not None:
            event_type = _handler_event_type(handler, route.event_type)
            input_model_module, input_model_name = _extract_input_model_ref(handler)
            route = route.model_copy(
                update={
                    "event_type": event_type,
                    "input_model_module": input_model_module
                    or route.input_model_module,
                    "input_model_name": input_model_name or route.input_model_name,
                }
            )
        aliases.append(
            (
                normalized,
                route
                or ModelRuntimeLocalIngressRoute(
                    node_name="",
                    contract_name="",
                    command_topic="",
                    event_type=None,
                    terminal_event=None,
                    contract_path="",
                    package_name="",
                ),
            )
        )
    return tuple(aliases)


def _handler_event_type(
    handler_entry: dict[object, object],
    fallback: str | None,
) -> str | None:
    raw_event_type = handler_entry.get("event_type")
    if isinstance(raw_event_type, str) and raw_event_type.strip():
        return raw_event_type.strip()
    return fallback


def validate_runtime_local_ingress_payload(
    route: ModelRuntimeLocalIngressRoute,
    payload: dict[str, JsonType],
) -> dict[str, JsonType]:
    """Validate and JSON-normalize an ingress payload against its route contract."""

    model_cls = _load_route_input_model(route)
    if model_cls is None:
        return payload

    model = model_cls.model_validate(payload)
    return cast("dict[str, JsonType]", model.model_dump(mode="json", exclude_none=True))


def _load_route_input_model(
    route: ModelRuntimeLocalIngressRoute,
) -> type[BaseModel] | None:
    if route.input_model_module is None and route.input_model_name is None:
        return None
    if route.input_model_module is None or route.input_model_name is None:
        raise ValueError(
            "Local ingress route declares an input_model without both module and name"
        )

    module = importlib.import_module(route.input_model_module)
    model_cls = getattr(module, route.input_model_name, None)
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
        raise TypeError(
            "Local ingress route input_model is not a pydantic BaseModel: "
            f"{route.input_model_module}.{route.input_model_name}"
        )
    return model_cls


def _extract_input_model_ref(
    raw: dict[object, object],
) -> tuple[str | None, str | None]:
    input_model = raw.get("input_model")
    if isinstance(input_model, dict):
        module = _safe_optional_string(
            input_model.get("module")
        ) or _safe_optional_string(raw.get("handler_module"))
        name = _safe_optional_string(input_model.get("name"))
        return (
            (module, name) if module is not None and name is not None else (None, None)
        )
    if isinstance(input_model, str):
        normalized = input_model.strip()
        if not normalized:
            return None, None
        module, separator, name = normalized.rpartition(".")
        if separator:
            return module, name
        handler_module = _safe_optional_string(raw.get("handler_module"))
        if handler_module is not None:
            return handler_module, normalized
    return None, None


class RuntimeLocalIngressServer:
    """Async Unix-socket server for local runtime dispatch requests."""

    def __init__(
        self,
        socket_path: str,
        request_handler: Callable[
            [ModelLocalRuntimeIngressRequest],
            Awaitable[ModelLocalRuntimeIngressResponse],
        ],
        *,
        socket_timeout_seconds: float = 5.0,
        socket_permissions: int = 0o660,
        max_payload_bytes: int = 1_048_576,
    ) -> None:
        self._socket_path = socket_path
        self._request_handler = request_handler
        self._socket_timeout_seconds = socket_timeout_seconds
        self._socket_permissions = socket_permissions
        self._max_payload_bytes = max_payload_bytes
        self._server: asyncio.Server | None = None
        self._shutdown_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._server.is_serving()

    @property
    def socket_path(self) -> str:
        return self._socket_path

    async def start(self) -> None:
        socket_path = Path(self._socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        _unlink_existing_socket(socket_path, raise_on_refusal=True)

        stream_limit = self._max_payload_bytes + 4096
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=self._socket_path,
            limit=stream_limit,
        )
        socket_path.chmod(self._socket_permissions)
        self._shutdown_event.clear()
        logger.info("RuntimeLocalIngressServer listening on %s", self._socket_path)

    async def stop(self) -> None:
        self._shutdown_event.set()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        socket_path = Path(self._socket_path)
        _unlink_existing_socket(socket_path, raise_on_refusal=False)

        logger.info("RuntimeLocalIngressServer stopped")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while not self._shutdown_event.is_set():
                try:
                    line = await asyncio.wait_for(
                        reader.readline(),
                        timeout=self._socket_timeout_seconds,
                    )
                except TimeoutError:
                    break

                if not line:
                    break

                response = await self._process_request_line(line)
                writer.write(response.model_dump_json().encode("utf-8") + b"\n")
                await writer.drain()
        except ConnectionResetError:
            logger.debug("Local ingress client reset the Unix-socket connection")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_request_line(
        self,
        line: bytes,
    ) -> ModelLocalRuntimeIngressResponse:
        if len(line) > self._max_payload_bytes:
            return ModelLocalRuntimeIngressResponse(
                ok=False,
                command_name="unknown",
                error=ModelLocalRuntimeIngressError(
                    code="validation_error",
                    message="Request exceeds max_payload_bytes",
                ),
            )

        try:
            raw = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return ModelLocalRuntimeIngressResponse(
                ok=False,
                command_name="unknown",
                error=ModelLocalRuntimeIngressError(
                    code="validation_error",
                    message=f"Invalid JSON request: {exc}",
                ),
            )

        try:
            request = ModelLocalRuntimeIngressRequest.model_validate(raw, strict=False)
        except ValidationError as exc:
            return ModelLocalRuntimeIngressResponse(
                ok=False,
                command_name=_preferred_request_name(raw),
                error=ModelLocalRuntimeIngressError(
                    code="validation_error",
                    message="Invalid local runtime ingress request",
                    details={"errors": json.loads(exc.json(include_url=False))},
                ),
            )

        return await self._request_handler(request)


def _resolve_package_root(package_name: str) -> Path:
    module = importlib.import_module(package_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise ValueError(f"Cannot resolve package root for '{package_name}'")
    return Path(module_file).resolve().parent


def _unlink_existing_socket(
    socket_path: Path,
    *,
    raise_on_refusal: bool,
) -> None:
    if not socket_path.exists() and not socket_path.is_symlink():
        return

    existing_stat = socket_path.lstat()
    allowed_group_ids = set(os.getgroups()) | {os.getgid(), os.getegid()}
    parent_group_id = socket_path.parent.stat().st_gid
    is_owned_socket = (
        stat.S_ISSOCK(existing_stat.st_mode)
        and existing_stat.st_uid == os.getuid()
        and existing_stat.st_gid in allowed_group_ids | {parent_group_id}
    )
    if is_owned_socket:
        socket_path.unlink()
        return

    message = (
        f"Refusing to unlink local ingress path {socket_path}: existing path is "
        "not an owned Unix socket"
    )
    if raise_on_refusal:
        raise FileExistsError(message)
    logger.warning(message)


def _safe_optional_string(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _select_command_topic(subscribe_topics: list[object]) -> str | None:
    normalized_topics = [
        str(topic).strip()
        for topic in subscribe_topics
        if isinstance(topic, str) and topic.strip()
    ]
    for topic in normalized_topics:
        if ".cmd." in topic:
            return topic
    return normalized_topics[0] if normalized_topics else None


def _derive_route_event_type(
    contract: dict[str, object],
    command_topic: str,
) -> str | None:
    handler_routing = contract.get("handler_routing")
    if isinstance(handler_routing, dict):
        handlers = handler_routing.get("handlers")
        if isinstance(handlers, list):
            for handler_entry in handlers:
                if isinstance(handler_entry, dict):
                    raw_event_type = handler_entry.get("event_type")
                    if isinstance(raw_event_type, str) and raw_event_type.strip():
                        return raw_event_type.strip()

    return EventBusSubcontractWiring._derive_event_type_from_topic(command_topic)


__all__ = [
    "ModelRuntimeLocalIngressRoute",
    "RuntimeLocalIngressServer",
    "discover_runtime_local_ingress_routes",
    "parse_active_runtime_packages",
]
