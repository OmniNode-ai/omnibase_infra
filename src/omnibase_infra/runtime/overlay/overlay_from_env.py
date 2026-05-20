# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Classify environment key-value pairs into overlay files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

from omnibase_core.enums.enum_overlay_scope import EnumOverlayScope
from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.transport_config_map import (
    TransportConfigMap,
)
from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter

_DEFAULT_OVERLAY_OUTPUT = Path.home() / ".omnibase" / "overlay.yaml"

_PREFIX_TO_TRANSPORT: dict[str, EnumInfraTransportType] = {
    "FS_": EnumInfraTransportType.FILESYSTEM,
    "GRAPH_": EnumInfraTransportType.GRAPH,
    "GRPC_": EnumInfraTransportType.GRPC,
    "HTTP_": EnumInfraTransportType.HTTP,
    "INFISICAL_": EnumInfraTransportType.INFISICAL,
    "KAFKA_": EnumInfraTransportType.KAFKA,
    "LLM_": EnumInfraTransportType.LLM,
    "MCP_": EnumInfraTransportType.MCP,
    "POSTGRES_": EnumInfraTransportType.DATABASE,
    "QDRANT_": EnumInfraTransportType.QDRANT,
    "REDIS_": EnumInfraTransportType.VALKEY,
    "VALKEY_": EnumInfraTransportType.VALKEY,
}


def _build_reverse_key_map() -> dict[str, EnumInfraTransportType]:
    result: dict[str, EnumInfraTransportType] = {}
    for transport in EnumInfraTransportType:
        for key in TransportConfigMap.keys_for_transport(transport):
            result[key] = transport
    return result


_EXACT_KEY_MAP: dict[str, EnumInfraTransportType] = _build_reverse_key_map()


def _classify_key(key: str) -> EnumInfraTransportType | None:
    if key in _EXACT_KEY_MAP:
        return _EXACT_KEY_MAP[key]
    for prefix, transport in _PREFIX_TO_TRANSPORT.items():
        if key.startswith(prefix):
            return transport
    return None


@overload
def overlay_from_env_dict(
    env_dict: dict[str, str],
    *,
    output_path: Path,
    environment: str = ...,
    scope: object = ...,
    allow_unclassified: bool = ...,
    return_warnings: bool = ...,
) -> Path: ...


@overload
def overlay_from_env_dict(
    env_dict: dict[str, str],
    *,
    output_path: None = ...,
    environment: str = ...,
    scope: object = ...,
    allow_unclassified: bool = ...,
    return_warnings: Literal[False] = ...,
) -> ModelOverlayFile:
    pass


@overload
def overlay_from_env_dict(
    env_dict: dict[str, str],
    *,
    output_path: None = ...,
    environment: str = ...,
    scope: object = ...,
    allow_unclassified: bool = ...,
    return_warnings: Literal[True],
) -> tuple[ModelOverlayFile, list[str]]:
    pass


def overlay_from_env_dict(
    env_dict: dict[str, str],
    *,
    output_path: Path | None = None,
    environment: str = "local",
    scope: object = EnumOverlayScope.ENV,
    allow_unclassified: bool = False,
    return_warnings: bool = False,
) -> object:
    """Classify env keys into a ModelOverlayFile, optionally writing it to disk."""
    transports: dict[str, dict[str, str]] = {}
    services: dict[str, dict[str, str]] = {}
    warnings: list[str] = []

    for key, value in env_dict.items():
        transport = _classify_key(key)
        if transport is not None:
            section = transport.value
            if section not in transports:
                transports[section] = {}
            transports[section][key] = value
        elif allow_unclassified:
            if "unclassified" not in services:
                services["unclassified"] = {}
            services["unclassified"][key] = value
        else:
            warnings.append(
                f"Key '{key}' could not be classified into any transport section. "
                "It will be omitted from the overlay. Pass allow_unclassified=True "
                "to place it in services.unclassified."
            )

    scope_value = scope.value if isinstance(scope, EnumOverlayScope) else str(scope)
    overlay = ModelOverlayFile.model_validate(
        {
            "environment": environment,
            "overlay_version": "1.0.0",
            "scope": scope_value,
            "services": services,
            "transports": transports,
        }
    )

    if output_path is not None:
        OverlayWriter().write(overlay, output_path)
        return output_path
    if return_warnings:
        return overlay, warnings
    return overlay


__all__ = ["overlay_from_env_dict"]
