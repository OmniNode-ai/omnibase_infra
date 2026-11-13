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

The overlay home's clone is read at a git ref (``OVERLAY_HOME_REF``), never
its working tree, the same way the runtime lane env overlay reader does.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

OVERLAY_ID = "llm_endpoints"
OVERLAY_HOME_RELPATH = f"overlays/{OVERLAY_ID}/overlay.yaml"

#: The ref and the git read bound of the overlay home's clone are the node contract's own
#: declaration, the one the runtime_lane_env overlay reader takes, so both readers share them.
_NODE_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "nodes/node_setup_local_provision_effect/contract.yaml"
)
_OVERLAY_HOME = yaml.safe_load(_NODE_CONTRACT.read_text(encoding="utf-8"))["config"][
    "overlay_home_lane_env"
]
OVERLAY_HOME_REF = str(_OVERLAY_HOME["ref"])
OVERLAY_HOME_GIT_TIMEOUT_SECONDS = int(_OVERLAY_HOME["read_timeout_seconds"])

__all__ = [
    "OVERLAY_ID",
    "OVERLAY_HOME_GIT_TIMEOUT_SECONDS",
    "OVERLAY_HOME_RELPATH",
    "OVERLAY_HOME_REF",
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
    def _validate_endpoint_urls(cls, value: object) -> dict[str, str]:
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
        """The named overlay roots, in precedence order.

        The ``ONEX_SKILL_OVERLAY_ROOTS`` entries (os.pathsep split).
        """
        return [
            Path(entry)
            for entry in environ.get("ONEX_SKILL_OVERLAY_ROOTS", "").split(os.pathsep)
            if entry
        ]

    @classmethod
    def clones(cls, environ: Mapping[str, str]) -> list[Path]:
        """The overlay home's clones, in precedence order.

        The deploy agent's overlay home clone when named, then the overlays
        clone beside the workspace config root when named.
        """
        clones: list[Path] = []
        clone = environ.get("DEPLOY_AGENT_OVERLAY_HOME_CLONE", "")
        if clone:
            clones.append(Path(clone))
        workspace = environ.get("ONEX_WORKSPACE_CONFIG_ROOT", "")
        if workspace:
            clones.append(Path(workspace).parent / "overlays")
        return clones

    @classmethod
    def read_clone(
        cls, clone: Path, ref: str = OVERLAY_HOME_REF
    ) -> ModelLlmEndpointOverlay | None:
        """The overlay at ``ref`` in ``clone``, or None when the ref holds none.

        A clone that is missing, is not a git clone, or holds no ``ref``
        raises ValueError naming the clone and ref, because a clone that
        cannot be read is not an overlay that holds nothing.
        """
        if not clone.is_dir():
            raise ValueError(
                f"{clone} is not a directory; cannot read the llm_endpoints "
                f"overlay at {ref}"
            )

        def git(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", "-C", str(clone), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=OVERLAY_HOME_GIT_TIMEOUT_SECONDS,
                env=scrub_git_location_env(),
            )

        if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
            raise ValueError(
                f"{clone} holds no {ref}; the llm_endpoints overlay cannot be "
                "read, and a clone that cannot be read is not the same as an "
                "overlay that holds nothing"
            )
        if git("cat-file", "-e", f"{ref}:{OVERLAY_HOME_RELPATH}").returncode != 0:
            return None
        shown = git("show", f"{ref}:{OVERLAY_HOME_RELPATH}")
        if shown.returncode != 0:
            raise ValueError(
                f"git show {ref}:{OVERLAY_HOME_RELPATH} in {clone} exited "
                f"{shown.returncode}: {shown.stderr.strip()[:240]}"
            )
        return _parse_overlay(shown.stdout, f"{clone}@{ref}:{OVERLAY_HOME_RELPATH}")

    @classmethod
    def read(cls, environ: Mapping[str, str]) -> ModelLlmEndpointOverlay | None:
        """The first overlay any root or clone holds, or None when none does.

        Named roots are tried first; a root holding no overlay file is
        skipped. Then every clone is tried with ``read_clone``; a clone that
        is named but unreadable raises ValueError, it is not skipped. A file
        that exists but does not validate is a loud refusal, never a quiet
        skip.
        """
        for root in cls.roots(environ):
            path = root / OVERLAY_ID / "overlay.yaml"
            if not path.is_file():
                continue
            return _parse_overlay(path.read_text(encoding="utf-8"), str(path))
        for clone in cls.clones(environ):
            overlay = cls.read_clone(clone)
            if overlay is not None:
                return overlay
        return None


def _parse_overlay(text: str, source: str) -> ModelLlmEndpointOverlay:
    """One overlay document, validated whole; any problem is a ValueError."""
    try:
        data = yaml.safe_load(text)
    except Exception as exc:
        raise ValueError(f"invalid llm_endpoints overlay file {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"invalid llm_endpoints overlay file {source}: top level must be a mapping"
        )
    try:
        return ModelLlmEndpointOverlay.model_validate(cast("dict[str, Any]", data))
    except Exception as exc:
        raise ValueError(f"invalid llm_endpoints overlay file {source}: {exc}") from exc


def _searched_locations(environ: Mapping[str, str]) -> list[str]:
    """The roots and clones, as strings, for a refusal message."""
    locations = [str(root) for root in ModelLlmEndpointOverlay.roots(environ)]
    for clone in ModelLlmEndpointOverlay.clones(environ):
        locations.append(f"{clone}@{OVERLAY_HOME_REF}")
    return locations


def resolve_overlay_endpoints(
    endpoints: list[dict[str, object]],
    environ: Mapping[str, str],
    store: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """Fill host/port/endpoint_url for slots flagged ``endpoint_from_overlay``.

    Returns copies of the endpoint dicts. Every dict with
    ``endpoint_from_overlay`` true gets host, port and endpoint_url filled
    from the overlay's ``endpoint_urls[slot_id]``; dicts without the flag are
    returned unchanged. Raises ``ValueError`` naming the slot id (and the
    roots and clones searched) when no overlay file exists or the slot is not
    in it.

    ``store`` is the settings store being checked: a flagged slot whose
    ``url_env_var`` that store leaves empty or unset is returned unresolved
    (no value is checked against it, so no address is needed). With no
    ``store`` every flagged slot is resolved.
    """
    overlay = ModelLlmEndpointOverlay.read(environ)
    locations = _searched_locations(environ)
    searched = ", ".join(locations) or "<none>"

    result: list[dict[str, object]] = []
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
                f"overlay slot {slot_id!r}: no llm_endpoints overlay file "
                f"found; searched roots: {searched}"
            )
        if slot_id not in overlay.endpoint_urls:
            raise ValueError(
                f"overlay slot {slot_id!r} is not in the llm_endpoints overlay "
                f"endpoint_urls; searched roots: {searched}"
            )
        url = overlay.endpoint_urls[slot_id]
        parsed = urlparse(url)
        new_endpoint = dict(endpoint)
        new_endpoint["host"] = parsed.hostname
        new_endpoint["port"] = parsed.port
        new_endpoint["endpoint_url"] = url
        result.append(new_endpoint)
    return result
