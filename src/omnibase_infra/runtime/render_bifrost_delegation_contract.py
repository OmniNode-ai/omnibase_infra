# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render the deployed Bifrost delegation contract for runtime boot.

The source ``bifrost_delegation.yaml`` intentionally keeps ``endpoint_url``
empty. Runtime deployments provide endpoint locations through the provider
backend named by each backend's ``base_url_env`` field. This module materializes
the deployed contract once at container startup so delegation code reads a
contract artifact, not scattered environment variables.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import yaml
from omnimarket.adapters.llm.bifrost.config_loader_bifrost_delegation import (
    load_bifrost_delegation_config,
)

from omnibase_infra.errors import ProtocolConfigurationError

_DEFAULT_SOURCE_PATH = Path("/app/src/omnibase_infra/configs/bifrost_delegation.yaml")
_DEFAULT_TARGET_PATH = Path("/app/data/delegation/bifrost_delegation.yaml")
_DEFAULT_ENDPOINT_PROBE_TIMEOUT_SECONDS = 3.0

EndpointProbe = Callable[[str, str, float], str | None]


def _load_yaml(path: Path) -> dict[str, object]:
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProtocolConfigurationError(
            f"Bifrost delegation contract must have a mapping root: {path}"
        )
    return {str(key): value for key, value in raw.items()}


def _has_populated_endpoint(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = _load_yaml(path)
    except ProtocolConfigurationError:
        return False
    backends = data.get("backends", [])
    if not isinstance(backends, list):
        return False
    return any(
        isinstance(backend, dict)
        and isinstance(backend.get("endpoint_url"), str)
        and bool(backend["endpoint_url"].strip())
        for backend in backends
    )


def _env_flag(value: str | None, *, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _probe_openai_model_endpoint(
    base_url: str,
    model_name: str,
    timeout_seconds: float,
) -> str | None:
    """Return None when ``base_url`` proves it serves ``model_name``."""
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        return f"unsupported endpoint URL scheme for {base_url!r}"
    endpoint = base_url.rstrip("/") + "/v1/models"
    request = Request(endpoint, headers={"accept": "application/json"})  # noqa: S310
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            body = response.read()
    except HTTPError as exc:
        return f"{endpoint} returned HTTP {exc.code}"
    except OSError as exc:
        return f"{endpoint} is not reachable: {exc}"

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"{endpoint} did not return JSON: {exc}"

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return f"{endpoint} response missing data[]"

    model_ids = {
        item.get("id")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if model_name not in model_ids:
        return f"{endpoint} does not list required model {model_name!r}"
    return None


def _populate_backend_endpoint(
    backend: dict[object, object],
    *,
    env: Mapping[str, str],
    verify_endpoints: bool,
    endpoint_probe: EndpointProbe,
    unresolved_required: list[str],
) -> bool:
    existing_url = backend.get("endpoint_url", "")
    endpoint_url = existing_url.strip() if isinstance(existing_url, str) else ""

    env_key = backend.get("base_url_env")
    if not endpoint_url and isinstance(env_key, str) and env_key.strip():
        endpoint_url = env.get(env_key, "").strip()

    required = backend.get("required", False) is True
    backend_id = backend.get("backend_id", "<unknown>")
    model_name = backend.get("model_name", "")

    if not endpoint_url:
        if required and isinstance(env_key, str) and env_key.strip():
            unresolved_required.append(env_key)
        return False

    if verify_endpoints and isinstance(model_name, str) and model_name.strip():
        failure = endpoint_probe(
            endpoint_url,
            model_name,
            _DEFAULT_ENDPOINT_PROBE_TIMEOUT_SECONDS,
        )
        if failure is not None:
            if required:
                unresolved_required.append(f"{backend_id}: {failure}")
            backend["endpoint_url"] = ""
            return False

    backend["endpoint_url"] = endpoint_url
    return True


_RENDER_HINT_FIELDS = frozenset({"required", "base_url_env"})


def _strip_render_hint_fields(path: Path) -> None:
    """Remove render-only hint fields from a contract file in place."""
    if not path.exists():
        return
    try:
        data = _load_yaml(path)
    except ProtocolConfigurationError:
        return
    backends = data.get("backends")
    if not isinstance(backends, list):
        return
    changed = False
    for backend in backends:
        if isinstance(backend, dict):
            for field in _RENDER_HINT_FIELDS:
                if field in backend:
                    del backend[field]  # type: ignore[attr-defined]
                    changed = True
    if changed:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _load_render_source(
    *,
    should_verify: bool,
    target: Path,
    source: Path,
) -> dict[str, object]:
    if should_verify and _has_populated_endpoint(target):
        return _load_yaml(target)
    if source.exists():
        return _load_yaml(source)
    raise ProtocolConfigurationError(f"Bifrost source contract not found at {source}")


def _populate_backend_endpoints(
    backends: list[object],
    *,
    env: Mapping[str, str],
    should_verify: bool,
    endpoint_probe: EndpointProbe,
) -> int:
    populated = 0
    unresolved_required: list[str] = []
    for backend in backends:
        if not isinstance(backend, dict):
            continue
        if _populate_backend_endpoint(
            backend,
            env=env,
            verify_endpoints=should_verify,
            endpoint_probe=endpoint_probe,
            unresolved_required=unresolved_required,
        ):
            populated += 1

    if unresolved_required:
        missing = "; ".join(sorted(set(unresolved_required)))
        raise ProtocolConfigurationError(
            f"Required Bifrost endpoint providers are unresolved: {missing}"
        )
    return populated


def render_bifrost_delegation_contract(
    *,
    source_path: Path | None = None,
    target_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    verify_endpoints: bool | None = None,
    endpoint_probe: EndpointProbe | None = None,
) -> Path:
    """Render the deployed Bifrost delegation contract and return its path."""
    env = environ if environ is not None else os.environ
    should_verify = (
        _env_flag(env.get("BIFROST_VERIFY_ENDPOINTS"), default=False)
        if verify_endpoints is None
        else verify_endpoints
    )
    probe = endpoint_probe or _probe_openai_model_endpoint
    source = source_path or Path(
        env.get("BIFROST_SOURCE_CONTRACT_PATH", str(_DEFAULT_SOURCE_PATH))
    )
    target = target_path or Path(
        env.get("BIFROST_CONTRACT_PATH", str(_DEFAULT_TARGET_PATH))
    )

    if not should_verify and _has_populated_endpoint(target):
        _strip_render_hint_fields(target)
        load_bifrost_delegation_config(target)
        return target

    data = _load_render_source(
        should_verify=should_verify, target=target, source=source
    )
    backends = data.get("backends")
    if not isinstance(backends, list):
        raise ProtocolConfigurationError(
            f"Bifrost source contract must declare a backends list: {source}"
        )

    populated = _populate_backend_endpoints(
        backends,
        env=env,
        should_verify=should_verify,
        endpoint_probe=probe,
    )
    if populated == 0:
        raise ProtocolConfigurationError(
            "Bifrost delegation contract rendered with no populated endpoint_url "
            "values. Add base_url_env fields to the source contract or provide a "
            "pre-rendered contract via BIFROST_CONTRACT_PATH."
        )

    # Strip render-only hint fields before writing the deployed contract —
    # `required` is used internally by _populate_backend_endpoint but is not
    # part of the deployed ModelDelegationBackendConfig schema.
    for backend in backends:
        if isinstance(backend, dict):
            backend.pop("required", None)
            backend.pop("base_url_env", None)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    load_bifrost_delegation_config(target)
    return target


def main() -> int:
    rendered = render_bifrost_delegation_contract()
    sys.stdout.write(f"[entrypoint] Bifrost delegation contract ready: {rendered}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
