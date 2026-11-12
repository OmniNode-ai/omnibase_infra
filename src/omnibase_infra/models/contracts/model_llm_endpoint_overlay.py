# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Overlay model for LLM endpoint addresses (OMN-24017).

The embedding endpoint address is no longer a hardcoded value in this repo.
It is read at run time from the private overlays repository:
``overlays/llm_endpoints/overlay.yaml``. The first overlay root holding
``<root>/llm_endpoints/overlay.yaml`` wins; a root holding no such file is
skipped, so no overlay anywhere means the flagged slot cannot be resolved.
An overlay file that exists but does not validate is a loud refusal, never a
quiet skip.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

OVERLAY_ID = "llm_endpoints"

__all__ = [
    "OVERLAY_ID",
    "ModelLlmEndpointOverlay",
    "resolve_overlay_endpoints",
]


def _validate_endpoint_url(url: str) -> str:
    """One overlay URL: http(s) host:port, no path/query/fragment."""
    url = url.rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"URL {url!r} must be http(s)")
    if not parsed.hostname:
        raise ValueError(f"URL {url!r} must carry a hostname")
    if parsed.port is None:
        raise ValueError(f"URL {url!r} must carry an explicit port")
    if parsed.path not in {"", "/"}:
        raise ValueError(f"URL {url!r} must not carry a path")
    if parsed.query or parsed.fragment:
        raise ValueError(f"URL {url!r} must not carry a query/fragment")
    return url


class ModelLlmEndpointOverlay(BaseModel):
    """One llm_endpoints overlay file, validated whole."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    #: slot id -> base URL (http(s) host:port, no path/query/fragment)
    endpoint_urls: dict[str, str]

    @field_validator("endpoint_urls", mode="before")
    @classmethod
    def _validate_endpoint_urls(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict) or not value:
            raise ValueError(
                "endpoint_urls must be a non-empty mapping of slot id to base URL"
            )
        result: dict[str, str] = {}
        for slot_id, raw in value.items():
            result[str(slot_id)] = _validate_endpoint_url(str(raw))
        return result

    @classmethod
    def roots(cls, environ: Mapping[str, str]) -> list[Path]:
        """Overlay roots in precedence order: named roots, then clones.

        The ``ONEX_SKILL_OVERLAY_ROOTS`` entries (os.pathsep split), then the
        overlays clone beside the deploy-agent home, then the overlays clone
        beside the workspace config root.
        """
        roots: list[Path] = [
            Path(entry)
            for entry in environ.get("ONEX_SKILL_OVERLAY_ROOTS", "").split(os.pathsep)
            if entry
        ]
        clone = environ.get("DEPLOY_AGENT_OVERLAY_HOME_CLONE", "")
        if clone:
            roots.append(Path(clone) / "overlays")
        workspace = environ.get("ONEX_WORKSPACE_CONFIG_ROOT", "")
        if workspace:
            roots.append(Path(workspace).parent / "overlays" / "overlays")
        return roots

    @classmethod
    def read(cls, environ: Mapping[str, str]) -> ModelLlmEndpointOverlay | None:
        """The first overlay file any root holds, or None when none does.

        A root holding no ``<root>/llm_endpoints/overlay.yaml`` is skipped. A
        file that exists but does not validate is a loud refusal (ValueError),
        never a quiet skip.
        """
        for root in cls.roots(environ):
            path = root / OVERLAY_ID / "overlay.yaml"
            if not path.is_file():
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(
                    f"invalid llm_endpoints overlay file {path}: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"invalid llm_endpoints overlay file {path}: top level must be a mapping"
                )
            try:
                return cls.model_validate(cast("dict[str, Any]", data))
            except Exception as exc:
                raise ValueError(
                    f"invalid llm_endpoints overlay file {path}: {exc}"
                ) from exc
        return None


def resolve_overlay_endpoints(
    endpoints: list[dict[str, Any]],
    environ: Mapping[str, str],
    store: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Fill host/port/endpoint_url for slots flagged ``endpoint_from_overlay``.

    Returns copies of the endpoint dicts. Every dict with
    ``endpoint_from_overlay`` true gets host, port and endpoint_url filled from
    the overlay's ``endpoint_urls[slot_id]``; dicts without the flag are
    returned unchanged. Raises ``ValueError`` naming the slot id (and the
    roots searched) when no overlay file exists or the slot is not in it.

    ``store`` is the settings store being checked: a flagged slot whose
    ``url_env_var`` that store leaves empty or unset is returned unresolved (no
    value is checked against it, so no address is needed). With no ``store``
    every flagged slot is resolved.
    """
    roots = ModelLlmEndpointOverlay.roots(environ)
    overlay = ModelLlmEndpointOverlay.read(environ)

    result: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if not endpoint.get("endpoint_from_overlay"):
            result.append(dict(endpoint))
            continue
        slot_id = str(endpoint.get("slot_id", ""))
        env_var = endpoint.get("url_env_var")
        if store is not None and env_var and not store.get(str(env_var), "").strip():
            result.append(dict(endpoint))
            continue
        if overlay is None:
            raise ValueError(
                f"overlay slot {slot_id!r}: no llm_endpoints overlay file found; "
                f"searched roots: {', '.join(str(root) for root in roots) or '<none>'}"
            )
        if slot_id not in overlay.endpoint_urls:
            raise ValueError(
                f"overlay slot {slot_id!r} is not in the llm_endpoints overlay "
                f"endpoint_urls; searched roots: "
                f"{', '.join(str(root) for root in roots) or '<none>'}"
            )
        url = overlay.endpoint_urls[slot_id]
        parsed = urlparse(url)
        new_endpoint = dict(endpoint)
        new_endpoint["host"] = parsed.hostname
        new_endpoint["port"] = parsed.port
        new_endpoint["endpoint_url"] = url
        result.append(new_endpoint)
    return result
