# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render the deployed Bifrost delegation contract for runtime boot.

The source ``bifrost_delegation.yaml`` intentionally keeps ``endpoint_url``
empty. Runtime deployments provide endpoint locations through the provider
backend named by each backend's ``base_url_env`` field. This module materializes
the deployed contract once at container startup so delegation code reads a
contract artifact, not scattered environment variables.

OMN-12864 — committed overlay authority
    BIFROST_LOCAL_*_ENDPOINT_URL values are now committed deployment bindings
    in docker/lane-overlays/dev.bifrost.yaml (and per-lane siblings). They are
    no longer ambient shell exports. A clean-shell compose up picks them up via
    the lane overlay env_file.

OMN-12814 — fail-loud loader
    ``render_bifrost_delegation_contract`` raises ``ProtocolConfigurationError``
    on any error that previously produced a silent empty result:
    - FileNotFoundError on missing source contract
    - yaml.YAMLError on malformed source YAML
    - ValidationError on schema violations
    - zero populated endpoints after rendering
    There is no lru_cache on the render result. Every restart re-renders from
    the packaged source (OMN-12945), so a stale cache can never pin a broken
    result across a deploy.

OMN-12945 — re-seed from packaged source on deploy
    The ``force_reseed`` parameter (and the ``BIFROST_FORCE_RESEED`` env flag)
    bypasses the stale-volume early-return path and always rebuilds the volume
    copy from the packaged bifrost_delegation.yaml merged with the committed
    lane overlay env-var endpoints. The entrypoint sets ``BIFROST_FORCE_RESEED``
    so every container restart re-seeds, eliminating the named-volume drift
    defect (OMN-12945).
"""

from __future__ import annotations

import importlib.resources
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.config_provenance import (
    build_config_provenance,
    write_provenance_sidecar,
)

_LEGACY_SOURCE_PATH = Path("/app/src/omnibase_infra/configs/bifrost_delegation.yaml")
_DEFAULT_TARGET_PATH = Path("/app/data/delegation/bifrost_delegation.yaml")


def _resolve_default_source_path() -> Path:
    """Resolve the bundled bifrost_delegation.yaml source path.

    Prefers the omnimarket package (canonical home since OMN-10865). Falls back
    to the legacy infra path so existing explicit BIFROST_SOURCE_CONTRACT_PATH
    overrides continue to work without change.
    """
    try:
        ref = importlib.resources.files("omnimarket").joinpath(
            "configs/bifrost_delegation.yaml"
        )
        candidate = Path(str(ref))
        if candidate.exists():
            return candidate
    except (ModuleNotFoundError, TypeError, AttributeError):
        pass
    return _LEGACY_SOURCE_PATH


_DEFAULT_SOURCE_PATH = _resolve_default_source_path()
_DEFAULT_ENDPOINT_PROBE_TIMEOUT_SECONDS = 3.0
_CHAT_COMPLETIONS_PATH_SUFFIX = "/chat/completions"

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
    endpoint_url: str,
    model_name: str,
    timeout_seconds: float,
) -> str | None:
    """Return None when ``endpoint_url`` proves it serves ``model_name``."""
    parsed = urlsplit(endpoint_url)
    if parsed.scheme not in {"http", "https"}:
        return f"unsupported endpoint URL scheme for {endpoint_url!r}"

    path = parsed.path.rstrip("/")
    if path.endswith(_CHAT_COMPLETIONS_PATH_SUFFIX):
        path = path[: -len(_CHAT_COMPLETIONS_PATH_SUFFIX)]
        models_path = f"{path}/models"
    else:
        models_path = f"{path}/v1/models"
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, models_path, "", ""))
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


def _endpoint_url_completeness_errors(
    data: dict[str, object],
    *,
    path: Path,
) -> list[str]:
    backends = data.get("backends")
    if not isinstance(backends, list):
        return []

    errors: list[str] = []
    for backend in backends:
        if not isinstance(backend, dict):
            continue
        endpoint_url = backend.get("endpoint_url")
        if not isinstance(endpoint_url, str) or not endpoint_url.strip():
            continue

        parsed = urlsplit(endpoint_url.strip())
        if parsed.scheme not in {"http", "https"}:
            continue

        backend_id = backend.get("backend_id", "<unknown>")
        if parsed.query or parsed.fragment:
            errors.append(
                f"{backend_id}: endpoint_url must not include query or fragment: "
                f"{endpoint_url!r}"
            )
            continue
        if not parsed.path.rstrip("/").endswith(_CHAT_COMPLETIONS_PATH_SUFFIX):
            errors.append(
                f"{backend_id}: endpoint_url must be a complete chat completion "
                f"URL ending in {_CHAT_COMPLETIONS_PATH_SUFFIX!r}: "
                f"{endpoint_url!r}"
            )

    return errors


def _validate_bifrost_delegation_endpoint_urls(
    data: dict[str, object],
    *,
    path: Path,
) -> None:
    errors = _endpoint_url_completeness_errors(data, path=path)
    if errors:
        joined = "; ".join(errors)
        raise ProtocolConfigurationError(
            "Bifrost delegation HTTP endpoint_url values must be complete chat "
            f"completion URLs: {joined}"
        )


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
                    # Why: Optional dependency or runtime adapter exposes this attribute dynamically.
                    del backend[field]  # type: ignore[attr-defined]
                    changed = True
    if changed:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _validate_bifrost_delegation_data(
    data: dict[str, object],
    *,
    path: Path,
    validate_endpoint_urls: bool = True,
) -> None:
    backends = data.get("backends")
    routing_rules = data.get("routing_rules")
    default_backends = data.get("default_backends", [])
    if not isinstance(backends, list):
        raise ProtocolConfigurationError(
            f"Bifrost delegation config must declare backends: {path}"
        )
    if not isinstance(routing_rules, list):
        raise ProtocolConfigurationError(
            f"Bifrost delegation config must declare routing_rules: {path}"
        )
    if not isinstance(default_backends, list):
        raise ProtocolConfigurationError(
            f"Bifrost delegation config default_backends must be a list: {path}"
        )

    backend_ids = {
        backend.get("backend_id")
        for backend in backends
        if isinstance(backend, dict) and isinstance(backend.get("backend_id"), str)
    }
    if len(backend_ids) != len(backends):
        raise ProtocolConfigurationError(
            f"Bifrost delegation config has undeclared backend_id entries: {path}"
        )

    validated_defaults: list[str] = []
    for backend_id in default_backends:
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ProtocolConfigurationError(
                "Bifrost delegation config default_backends entries must be "
                f"non-empty strings: {path}"
            )
        validated_defaults.append(backend_id)

    unknown_defaults = {
        backend_id for backend_id in validated_defaults if backend_id not in backend_ids
    }
    if unknown_defaults:
        raise ProtocolConfigurationError(
            "Bifrost delegation config default_backends references undeclared "
            f"backend(s): {sorted(unknown_defaults)}"
        )

    rule_ids: list[str] = []
    for rule in routing_rules:
        if not isinstance(rule, dict):
            raise ProtocolConfigurationError(
                f"Bifrost delegation config routing rule must be a mapping: {path}"
            )
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ProtocolConfigurationError(
                "Bifrost delegation config routing rule_id must be a non-empty "
                f"string: {path}"
            )
        rule_ids.append(rule_id)

        rule_backend_ids = rule.get("backend_ids", [])
        if not isinstance(rule_backend_ids, list):
            raise ProtocolConfigurationError(
                "Bifrost delegation config routing rule backend_ids must be a list"
            )
        validated_rule_backend_ids: list[str] = []
        for backend_id in rule_backend_ids:
            if not isinstance(backend_id, str) or not backend_id.strip():
                raise ProtocolConfigurationError(
                    "Bifrost delegation config routing rule backend_ids entries "
                    f"must be non-empty strings: {path}"
                )
            validated_rule_backend_ids.append(backend_id)

        unknown_rule_backends = {
            backend_id
            for backend_id in validated_rule_backend_ids
            if backend_id not in backend_ids
        }
        if unknown_rule_backends:
            raise ProtocolConfigurationError(
                "Bifrost delegation config routing rule references undeclared "
                f"backend(s): {sorted(unknown_rule_backends)}"
            )

    if len(rule_ids) != len(set(rule_ids)):
        raise ProtocolConfigurationError(
            "Bifrost delegation config contains duplicate routing rule_id values"
        )

    if validate_endpoint_urls:
        _validate_bifrost_delegation_endpoint_urls(data, path=path)


def _validate_bifrost_delegation_config(path: Path) -> None:
    data = _load_yaml(path)
    _validate_bifrost_delegation_data(data, path=path)


def _has_complete_endpoint_urls(path: Path) -> bool:
    try:
        data = _load_yaml(path)
    except ProtocolConfigurationError:
        return False
    return not _endpoint_url_completeness_errors(data, path=path)


def _backend_index_by_id(data: dict[str, object]) -> dict[str, dict[object, object]]:
    backends = data.get("backends", [])
    if not isinstance(backends, list):
        return {}
    indexed: dict[str, dict[object, object]] = {}
    for backend in backends:
        if not isinstance(backend, dict):
            continue
        backend_id = backend.get("backend_id")
        if isinstance(backend_id, str) and backend_id.strip():
            indexed[backend_id] = backend
    return indexed


def _target_satisfies_declared_env_backends(
    *,
    target_data: dict[str, object],
    source_data: dict[str, object],
    env: Mapping[str, str],
) -> bool:
    source_backends = source_data.get("backends", [])
    if not isinstance(source_backends, list):
        return True

    target_backends = _backend_index_by_id(target_data)
    for source_backend in source_backends:
        if not isinstance(source_backend, dict):
            continue

        backend_id = source_backend.get("backend_id")
        env_key = source_backend.get("base_url_env")
        if not (
            isinstance(backend_id, str)
            and backend_id.strip()
            and isinstance(env_key, str)
            and env_key.strip()
        ):
            continue

        env_url = env.get(env_key, "").strip()
        if not env_url:
            continue

        target_backend = target_backends.get(backend_id)
        if target_backend is None:
            return False

        target_url = target_backend.get("endpoint_url")
        if not isinstance(target_url, str) or target_url.strip() != env_url:
            return False

        source_model = source_backend.get("model_name")
        if isinstance(source_model, str) and source_model.strip():
            target_model = target_backend.get("model_name")
            if not isinstance(target_model, str) or target_model != source_model:
                return False

    return True


def _load_render_source(
    *,
    should_verify: bool,
    target: Path,
    source: Path,
) -> dict[str, object]:
    if should_verify and _has_populated_endpoint(target):
        data = _load_yaml(target)
        _validate_bifrost_delegation_data(
            data,
            path=target,
            validate_endpoint_urls=False,
        )
        if _has_complete_endpoint_urls(target):
            return data
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


def _resolve_target_path(
    *,
    target_path: Path | None,
    env: Mapping[str, str],
) -> Path | None:
    if target_path is not None:
        return target_path
    configured_path = env.get("BIFROST_CONTRACT_PATH")
    if configured_path is None:
        return _DEFAULT_TARGET_PATH
    stripped_path = configured_path.strip()
    if not stripped_path:
        return None
    return Path(stripped_path)


def render_bifrost_delegation_contract(
    *,
    source_path: Path | None = None,
    target_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    verify_endpoints: bool | None = None,
    endpoint_probe: EndpointProbe | None = None,
    force_reseed: bool | None = None,
) -> Path | None:
    """Render the deployed Bifrost delegation contract and return its path.

    Args:
        source_path: Path to the packaged bifrost_delegation.yaml source.
            Defaults to the omnimarket-bundled copy.
        target_path: Path to write the rendered contract on the runtime volume.
            Defaults to /app/data/delegation/bifrost_delegation.yaml.
            Pass ``None`` to accept the env-configured target; returns None
            when BIFROST_CONTRACT_PATH resolves to an empty path (services
            that deliberately skip rendering, e.g. projection-api).
        environ: Mapping used for env-var reads. Defaults to os.environ.
        verify_endpoints: When True, probe each populated endpoint via
            GET /v1/models and reject backends whose model is not listed.
            Defaults to the BIFROST_VERIFY_ENDPOINTS env flag.
        endpoint_probe: Probe callable used for endpoint verification. Defaults
            to _probe_openai_model_endpoint.
        force_reseed: When True (or when BIFROST_FORCE_RESEED=1 is set), always
            re-render from the packaged source, ignoring any existing volume
            content (OMN-12945). This is the correct mode on container restart:
            the volume copy is never an authority on a fresh boot.

    Returns:
        Path to the rendered contract, or None when rendering is explicitly
        disabled for this service.

    Raises:
        ProtocolConfigurationError: On any rendering failure — missing source,
            malformed YAML, schema validation failure, or zero populated
            endpoints (OMN-12814: fail-loud, never returns a silent empty result).
        FileNotFoundError: If the source contract cannot be found.
        ValueError: If the source YAML fails schema validation.
    """
    env = environ if environ is not None else os.environ
    should_verify = (
        _env_flag(env.get("BIFROST_VERIFY_ENDPOINTS"), default=False)
        if verify_endpoints is None
        else verify_endpoints
    )
    # OMN-12945: BIFROST_FORCE_RESEED bypasses the stale-volume early-return path.
    # The entrypoint sets this flag so every container restart re-seeds from the
    # packaged source. Callers may also pass force_reseed=True explicitly.
    should_force_reseed = (
        _env_flag(env.get("BIFROST_FORCE_RESEED"), default=False)
        if force_reseed is None
        else force_reseed
    )
    probe = endpoint_probe or _probe_openai_model_endpoint
    _source_env = env.get("BIFROST_SOURCE_CONTRACT_PATH", "").strip()
    source = source_path or (Path(_source_env) if _source_env else _DEFAULT_SOURCE_PATH)
    target = _resolve_target_path(target_path=target_path, env=env)
    if target is None:
        return None

    # Cache-hit path: skip re-rendering when the volume already contains a
    # valid, up-to-date contract AND we are not forcing a re-seed.
    # OMN-12945: force_reseed/BIFROST_FORCE_RESEED bypasses this entirely —
    # the entrypoint sets it on every restart to eliminate named-volume drift.
    if (
        not should_force_reseed
        and not should_verify
        and _has_populated_endpoint(target)
    ):
        source_data = _load_yaml(source) if source.exists() else None
        data = _load_yaml(target)
        _validate_bifrost_delegation_data(
            data,
            path=target,
            validate_endpoint_urls=False,
        )
        if _has_complete_endpoint_urls(target) and (
            source_data is None
            or _target_satisfies_declared_env_backends(
                target_data=data,
                source_data=source_data,
                env=env,
            )
        ):
            _strip_render_hint_fields(target)
            return target

    # Re-seed path: always render from the packaged source.
    # OMN-12814: every error below raises ProtocolConfigurationError — no silent
    # empty-result fallback.
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
            "values. Set the four BIFROST_LOCAL_*_ENDPOINT_URL vars from the "
            "committed lane overlay (docker/lane-overlays/<lane>.bifrost.env) "
            "or provide a pre-rendered contract at BIFROST_CONTRACT_PATH. "
            "OMN-12864: these values must come from the committed overlay, "
            "not ephemeral shell exports."
        )

    # Strip render-only hint fields before writing the deployed contract —
    # `required` is used internally by _populate_backend_endpoint but is not
    # part of the deployed ModelDelegationBackendConfig schema.
    for backend in backends:
        if isinstance(backend, dict):
            backend.pop("required", None)
            backend.pop("base_url_env", None)

    target.parent.mkdir(parents=True, exist_ok=True)
    staged_target = target.with_suffix(f"{target.suffix}.tmp")
    staged_target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _validate_bifrost_delegation_config(staged_target)
    staged_target.replace(target)
    return target


def _emit_config_provenance(rendered: Path, *, env: Mapping[str, str]) -> None:
    """Compute, log, and persist config provenance for the rendered contract.

    Provenance compares the deployed (volume) contract that the runtime actually
    loaded against the packaged source resolved from the same rules the renderer
    uses. The single-line summary is logged to stdout so it appears in container
    startup logs; the sidecar JSON lets the drift sweep and proof packets read
    provenance without re-resolving the packaged source path (OMN-12958).
    """
    source_env = env.get("BIFROST_SOURCE_CONTRACT_PATH", "").strip()
    source = Path(source_env) if source_env else _DEFAULT_SOURCE_PATH
    provenance = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=rendered,
        source_path=source,
    )
    sys.stdout.write(f"[entrypoint] {provenance.provenance_line()}\n")
    if provenance.has_drifted:
        sys.stdout.write(
            "[entrypoint] WARNING: deployed Bifrost delegation config drifted "
            "from packaged source; re-seed required (OMN-12958)\n"
        )
    write_provenance_sidecar(provenance, deployed_path=rendered)


def main() -> int:
    rendered = render_bifrost_delegation_contract()
    if rendered is None:
        sys.stdout.write(
            "[entrypoint] Bifrost delegation contract rendering disabled: "
            "BIFROST_CONTRACT_PATH is empty\n"
        )
        return 0
    sys.stdout.write(f"[entrypoint] Bifrost delegation contract ready: {rendered}\n")
    _emit_config_provenance(rendered, env=os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
