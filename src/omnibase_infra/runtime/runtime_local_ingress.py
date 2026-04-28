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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

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


@dataclass(frozen=True, slots=True)
class RuntimeLocalIngressRoute:
    """Resolved route for a node exposed through the local runtime ingress."""

    node_name: str
    contract_name: str
    command_topic: str
    event_type: str | None
    terminal_event: str | None
    contract_path: str
    package_name: str


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
) -> dict[str, RuntimeLocalIngressRoute]:
    """Discover local-ingress routes from installed package node contracts."""

    routes: dict[str, RuntimeLocalIngressRoute] = {}

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
            route = RuntimeLocalIngressRoute(
                node_name=node_dir_name,
                contract_name=contract_name,
                command_topic=command_topic,
                event_type=_derive_route_event_type(raw, command_topic),
                terminal_event=_safe_optional_string(raw.get("terminal_event")),
                contract_path=str(contract_path),
                package_name=package_name,
            )

            for alias in (contract_name, node_dir_name):
                existing = routes.get(alias)
                if existing is not None and existing != route:
                    raise ValueError(
                        f"Duplicate local ingress route alias '{alias}' for "
                        f"{existing.contract_path} and {route.contract_path}"
                    )
                routes[alias] = route

    return routes


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
    "RuntimeLocalIngressRoute",
    "RuntimeLocalIngressServer",
    "discover_runtime_local_ingress_routes",
    "parse_active_runtime_packages",
]
