# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render the deployed Bifrost contract from its typed lane overlay.

The lane overlay is configuration data, not a dotenv template.  In particular,
this module never reads endpoint or model bindings from the process environment.

OMN-17099: the overlay may ADD a backend the base contract does not declare,
when the binding carries the full declaration (provider, tier, credential). The
added backend is appended to the rendered contract after every base backend, so
it never displaces a rung the base declared. A binding that names an undeclared
backend without that declaration, or a base backend WITH it, fails the render —
never a silent drop and never a default.
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
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.config_provenance import (
    build_config_provenance,
    write_provenance_sidecar,
)
from omnibase_infra.runtime.dogfood_delegation_fault_routes import (
    load_dogfood_delegation_fault_routes,
)
from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    NEW_BACKEND_DECLARATION_FIELDS,
    ModelBifrostLaneBackendBinding,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

_DEFAULT_TARGET_PATH = Path("/app/data/delegation/bifrost_delegation.yaml")
_LANE_OVERLAY_PATH_ENV = "BIFROST_LANE_OVERLAY_PATH"
_CHAT_COMPLETIONS_PATH_SUFFIX = "/chat/completions"
_DEFAULT_ENDPOINT_PROBE_TIMEOUT_SECONDS = 3.0
#: The base contract's own declaration that a backend is served from the lab.
_LOCAL_TIER = "local"

EndpointProbe = Callable[[str, str, float], str | None]


def _resolve_canonical_source_path() -> Path:
    """Resolve the packaged omnimarket Bifrost base contract."""
    try:
        candidate = Path(
            str(
                importlib.resources.files("omnimarket").joinpath(
                    "configs/bifrost_delegation.yaml"
                )
            )
        )
        if candidate.is_file():
            return candidate
    except (ModuleNotFoundError, TypeError, AttributeError) as exc:
        raise ProtocolConfigurationError(
            "Packaged omnimarket Bifrost base contract is unavailable"
        ) from exc
    raise ProtocolConfigurationError(
        f"Packaged omnimarket Bifrost base contract is unavailable: {candidate}"
    )


def _load_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolConfigurationError(f"{label} not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ProtocolConfigurationError(
            f"{label} is malformed: {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ProtocolConfigurationError(f"{label} must have a mapping root: {path}")
    return {str(key): value for key, value in raw.items()}


def _load_lane_overlay(path: Path) -> ModelBifrostLaneOverlay:
    try:
        return ModelBifrostLaneOverlay.model_validate(
            _load_mapping(path, label="Bifrost lane overlay")
        )
    except ValidationError as exc:
        raise ProtocolConfigurationError(
            f"Bifrost lane overlay is invalid: {path}: {exc}"
        ) from exc


def _resolve_overlay_path(*, overlay_path: Path | None, env: Mapping[str, str]) -> Path:
    """Resolve the lane overlay from the explicit argument or the lane's env pin.

    There is deliberately NO default path. The overlay is where a lane's
    endpoint, model, and local operational bindings come from, so a fallback
    to a fixed filename is a fallback to *another lane's* routing config: the
    previous hardcoded default named the dev lane's overlay file, sending
    every lane that did not mount its own overlay through the dev lane's file
    — silently when the file happened to be present, and with a dev-named
    error on lanes that never mounted it (OMN-17150, found on the first cold
    boot of the collaborator lane). Each lane's compose contract now pins its
    own overlay path next to the mount that provides the file, and a lane
    that renders without a pin fails loudly here, naming the lane.
    """
    if overlay_path is not None:
        return overlay_path
    configured = env.get(_LANE_OVERLAY_PATH_ENV, "").strip()
    if not configured:
        lane = env.get("ONEX_ENVIRONMENT", "").strip() or "<ONEX_ENVIRONMENT unset>"
        raise ProtocolConfigurationError(
            f"{_LANE_OVERLAY_PATH_ENV} is not bound for lane {lane!r}: a lane "
            "that renders the Bifrost delegation contract must pin its own "
            "typed overlay path alongside the mount that provides the file. "
            "There is no default — falling through to another lane's overlay "
            "would bind this lane's delegation routing to that lane's "
            "backends (OMN-17150)."
        )
    return Path(configured)


def _resolve_target_path(
    *, target_path: Path | None, env: Mapping[str, str]
) -> Path | None:
    if target_path is not None:
        return target_path
    configured_path = env.get("BIFROST_CONTRACT_PATH")
    if configured_path is None:
        raise ProtocolConfigurationError(
            "BIFROST_CONTRACT_PATH is not bound; pass target_path explicitly or "
            "bind the deployed contract path."
        )
    stripped_path = configured_path.strip()
    return Path(stripped_path) if stripped_path else None


def _probe_openai_model_endpoint(
    endpoint_url: str, model_name: str, timeout_seconds: float
) -> str | None:
    parsed = urlsplit(endpoint_url)
    path = parsed.path.rstrip("/")
    models_path = (
        f"{path[: -len(_CHAT_COMPLETIONS_PATH_SUFFIX)]}/models"
        if path.endswith(_CHAT_COMPLETIONS_PATH_SUFFIX)
        else f"{path}/v1/models"
    )
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, models_path, "", ""))
    try:
        request = Request(endpoint, headers={"accept": "application/json"})  # noqa: S310
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"{endpoint} is not a readable model endpoint: {exc}"
    data = payload.get("data") if isinstance(payload, dict) else None
    model_ids = (
        {item.get("id") for item in data if isinstance(item, dict)}
        if isinstance(data, list)
        else set()
    )
    if model_name not in model_ids:
        return f"{endpoint} does not list required model {model_name!r}"
    return None


def _index_base_backends(base: dict[str, object]) -> dict[str, dict[object, object]]:
    backends = base.get("backends")
    if not isinstance(backends, list):
        raise ProtocolConfigurationError("Bifrost base contract must declare backends")
    by_id: dict[str, dict[object, object]] = {}
    for backend in backends:
        if not isinstance(backend, dict):
            raise ProtocolConfigurationError(
                "Bifrost base contract backend must be a mapping"
            )
        backend_id = backend.get("backend_id")
        if (
            not isinstance(backend_id, str)
            or not backend_id.strip()
            or backend_id in by_id
        ):
            raise ProtocolConfigurationError(
                "Bifrost base contract has duplicate or blank backend_id"
            )
        by_id[backend_id] = backend
    return by_id


def _is_local_backend(backend: dict[object, object]) -> bool:
    """Whether the contract declares this backend as lab-served (its ``tier``)."""
    return backend.get("tier") == _LOCAL_TIER


def _routed_local_backend_ids(
    base: dict[str, object], by_id: dict[str, dict[object, object]]
) -> set[str]:
    """Every local backend the BASE contract routes to.

    A lab lane must bind each of these (OMN-16833: an omitted rung silently
    degrades its task classes to the metered ceiling). The set is read from the
    base contract's own routing surface — ``routing_rules[].backend_ids`` and
    ``default_backends`` — never from a list shipped in this module (OMN-17099).
    """
    routed: set[str] = set()
    rules = base.get("routing_rules")
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, dict) and isinstance(rule.get("backend_ids"), list):
                routed.update(str(item) for item in rule["backend_ids"])
    defaults = base.get("default_backends")
    if isinstance(defaults, list):
        routed.update(str(item) for item in defaults)
    return {
        backend_id
        for backend_id in routed
        if backend_id in by_id and _is_local_backend(by_id[backend_id])
    }


def _append_dogfood_fault_backends(
    base: dict[str, object], *, overlay: ModelBifrostLaneOverlay
) -> None:
    """Materialize dogfood fault routes from their single packaged authority.

    The ci bus-lane resource owns fault endpoint, status, and one-hop policy.
    The dogfood Bifrost overlay chooses the lane; it never repeats those values.
    These backends exist only in the rendered dogfood artifact, not the shipped
    normal/prod Bifrost contract.
    """
    if overlay.lane != "dogfood":
        return
    backends = base.get("backends")
    if not isinstance(backends, list):
        raise ProtocolConfigurationError("Bifrost base contract must declare backends")
    existing = _index_base_backends(base)
    for route in load_dogfood_delegation_fault_routes():
        if route.backend_id in existing:
            raise ProtocolConfigurationError(
                f"dogfood fault route {route.backend_id!r} collides with a base backend"
            )
        backends.append(
            {
                "backend_id": route.backend_id,
                "provider": "dogfood_fault",
                "endpoint_url": route.endpoint_url,
                "model_name": route.backend_id,
                "tier": "dogfood_fault",
                "timeout_ms": route.requested_timeout_seconds * 1000,
                "max_tokens": 1,
                "capabilities": [],
            }
        )


def _disable_local_backends(by_id: dict[str, dict[object, object]]) -> None:
    """Write a cloud lane's absent local rungs into the artifact (OMN-17502).

    The entries are DISABLED, never deleted. ``load_bifrost_delegation_config``
    (omnimarket, OMN-15628) refuses a contract whose ``routing_rules`` or
    ``default_backends`` name a backend the contract does not declare, and the
    base contract's ``code_generation`` rule and ``default_backends`` both name
    ``local-coder``. A null ``endpoint_url`` is the shape the reducer's
    ``_load_bifrost_endpoints`` skips, so the chain of responders never offers a
    local rung while the routing table stays internally consistent.
    """
    for backend in by_id.values():
        if _is_local_backend(backend):
            backend["endpoint_url"] = None


def _added_backend_entry(
    binding: ModelBifrostLaneBackendBinding,
) -> dict[object, object]:
    """The complete base-contract entry for a backend the lane overlay adds."""
    # The binding model guarantees the declaration is all-or-none; the caller
    # has already established it is present.
    assert binding.provider is not None
    assert binding.tier is not None
    assert binding.credential is not None
    return {
        "backend_id": binding.backend_key,
        "provider": binding.provider,
        "endpoint_url": binding.endpoint_url if binding.serving else None,
        "model_name": binding.advertised_model,
        "tier": binding.tier,
        "timeout_ms": binding.timeout_ms,
        "max_tokens": binding.max_tokens,
        "secret_ref": binding.credential.secret_ref,
        "capabilities": list(binding.capabilities),
    }


def _merge_lane_overlay(
    *,
    base: dict[str, object],
    overlay: ModelBifrostLaneOverlay,
    verify: bool,
    endpoint_probe: EndpointProbe,
) -> dict[str, object]:
    _append_dogfood_fault_backends(base, overlay=overlay)
    by_id = _index_base_backends(base)

    if overlay.locale is EnumBifrostLaneLocale.CLOUD:
        _disable_local_backends(by_id)
    else:
        bound = {binding.backend_key for binding in overlay.backends}
        unbound = sorted(_routed_local_backend_ids(base, by_id) - bound)
        if unbound:
            raise ProtocolConfigurationError(
                f"Bifrost lane overlay for lane {overlay.lane!r} declares locale "
                f"{EnumBifrostLaneLocale.LAB.value!r} but does not bind the "
                f"local backend(s) {unbound} that the base contract routes to. "
                "A lab lane that omits a routed rung silently degrades those "
                "task classes to the metered ceiling (OMN-16833); bind it, or "
                "bind it with serving: false if its endpoint is dark."
            )

    added: list[dict[object, object]] = []
    for binding in overlay.backends:
        backend = by_id.get(binding.backend_key)
        if backend is None:
            if not binding.declares_new_backend:
                raise ProtocolConfigurationError(
                    f"Bifrost lane overlay backend {binding.backend_key!r} is not "
                    "declared by the base contract, and the binding does not "
                    f"declare it either: an added backend must declare "
                    f"{list(NEW_BACKEND_DECLARATION_FIELDS)} (OMN-17099). A "
                    "misspelled id of a base backend fails here too."
                )
            backend = _added_backend_entry(binding)
            added.append(backend)
        else:
            if binding.declares_new_backend:
                raise ProtocolConfigurationError(
                    f"Bifrost lane overlay backend {binding.backend_key!r} is "
                    f"declared by the base contract, which owns its "
                    f"{list(NEW_BACKEND_DECLARATION_FIELDS)}: a lane may rebind "
                    "its endpoint and served id, never rewrite its provider, "
                    "tier or credential (OMN-17099)."
                )
            if "model_name" not in backend:
                raise ProtocolConfigurationError(
                    f"Bifrost base backend {binding.backend_key!r} must declare model_name"
                )
            base_model = backend["model_name"]
            if base_model is not None and base_model != binding.advertised_model:
                raise ProtocolConfigurationError(
                    f"Bifrost base backend {binding.backend_key!r} model_name {base_model!r} "
                    f"does not match overlay served_model_id {binding.advertised_model!r}"
                )
        if not binding.serving:
            # OMN-16999: a DECLARED-but-dark rung. Write the disabled shape —
            # the same ``endpoint_url: null`` a cloud lane's local backends get
            # — so ``_load_bifrost_endpoints`` skips it and routing never offers
            # it, while ``routing_rules``/``default_backends`` naming this
            # backend stay resolvable. model_name is still asserted above and
            # still written below, so the binding survives in the artifact and
            # the rung is restored by flipping one flag, not by reconstructing
            # it. The probe is skipped for the obvious reason: probing an
            # endpoint already proven dark would only fail the render.
            backend["endpoint_url"] = None
            backend["model_name"] = binding.advertised_model
            backend["max_tokens"] = binding.max_tokens
            backend["timeout_ms"] = binding.timeout_ms
            continue
        if verify:
            failure = endpoint_probe(
                binding.endpoint_url,
                binding.advertised_model,
                _DEFAULT_ENDPOINT_PROBE_TIMEOUT_SECONDS,
            )
            if failure is not None:
                raise ProtocolConfigurationError(
                    f"Bifrost lane binding {binding.backend_key!r} failed verification: {failure}"
                )
        backend["endpoint_url"] = binding.endpoint_url
        backend["model_name"] = binding.advertised_model
        backend["max_tokens"] = binding.max_tokens
        backend["timeout_ms"] = binding.timeout_ms

    # Endpoint/model env wiring is forbidden in the rendered contract.  This
    # strips stale source hints even for disabled backends such as local-reasoner.
    for backend in by_id.values():
        backend.pop("endpoint_url_env", None)
        backend.pop("required", None)
    if added:
        backends = base["backends"]
        assert isinstance(backends, list)
        backends.extend(added)
    return base


def _validate_rendered_contract(
    data: dict[str, object], *, overlay: ModelBifrostLaneOverlay
) -> None:
    backends = data.get("backends")
    if not isinstance(backends, list):
        raise ProtocolConfigurationError(
            "Rendered Bifrost contract must declare backends"
        )
    active = 0
    local_with_endpoint: list[str] = []
    for backend in backends:
        if not isinstance(backend, dict):
            raise ProtocolConfigurationError(
                "Rendered Bifrost backend must be a mapping"
            )
        if "endpoint_url_env" in backend:
            raise ProtocolConfigurationError(
                "Rendered Bifrost contract must not contain endpoint_url_env"
            )
        endpoint_url = backend.get("endpoint_url")
        if isinstance(endpoint_url, str) and endpoint_url.strip():
            if not endpoint_url.rstrip("/").endswith(_CHAT_COMPLETIONS_PATH_SUFFIX):
                raise ProtocolConfigurationError(
                    f"Rendered Bifrost endpoint must be complete: {endpoint_url!r}"
                )
            active += 1
            backend_id = backend.get("backend_id")
            if isinstance(backend_id, str) and _is_local_backend(backend):
                local_with_endpoint.append(backend_id)
    if active == 0:
        raise ProtocolConfigurationError(
            "Rendered Bifrost contract has no active endpoint"
        )
    if overlay.locale is EnumBifrostLaneLocale.CLOUD and local_with_endpoint:
        raise ProtocolConfigurationError(
            f"Lane {overlay.lane!r} renders with locale "
            f"{EnumBifrostLaneLocale.CLOUD.value!r} but the rendered contract "
            f"still binds local backend(s) {sorted(local_with_endpoint)} — a "
            "cloud lane must reach no lab endpoint (OMN-17502)"
        )


def render_bifrost_delegation_contract(
    *,
    source_path: Path | None = None,
    overlay_path: Path | None = None,
    target_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    verify_endpoints: bool | None = None,
    endpoint_probe: EndpointProbe | None = None,
) -> Path | None:
    """Render the base contract merged with the required typed lane overlay.

    ``environ`` is used only for the target path, the lane's overlay path pin
    (``BIFROST_LANE_OVERLAY_PATH``), and the endpoint-verification flag;
    endpoint, model, and local operational bindings always come from the
    resolved overlay file, never from the environment.

    The overlay's execution locale decides what "merged" means (OMN-17502): a
    ``lab`` lane binds every local backend the base contract routes to and may
    add fully declared backends of its own (OMN-17099), while a ``cloud``
    lane declares none and renders the base contract's cloud backends with every
    local rung explicitly disabled.
    """
    env = environ if environ is not None else os.environ
    target = _resolve_target_path(target_path=target_path, env=env)
    if target is None:
        return None
    source = source_path or _resolve_canonical_source_path()
    overlay = _load_lane_overlay(
        _resolve_overlay_path(overlay_path=overlay_path, env=env)
    )
    should_verify = (
        env.get("BIFROST_VERIFY_ENDPOINTS", "").strip().lower()
        in {"1", "true", "yes", "on"}
        if verify_endpoints is None
        else verify_endpoints
    )
    data = _merge_lane_overlay(
        base=_load_mapping(source, label="Bifrost base contract"),
        overlay=overlay,
        verify=should_verify,
        endpoint_probe=endpoint_probe or _probe_openai_model_endpoint,
    )
    _validate_rendered_contract(data, overlay=overlay)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged_target = target.with_suffix(f"{target.suffix}.tmp")
    staged_target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    staged_target.replace(target)
    return target


def _emit_config_provenance(*, rendered: Path, source_path: Path) -> None:
    provenance = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=rendered,
        source_path=source_path,
    )
    sys.stdout.write(f"[entrypoint] {provenance.provenance_line()}\n")
    write_provenance_sidecar(provenance, deployed_path=rendered)


def main() -> int:
    source_path = _resolve_canonical_source_path()
    rendered = render_bifrost_delegation_contract(source_path=source_path)
    if rendered is None:
        return 0
    sys.stdout.write(f"[entrypoint] Bifrost delegation contract ready: {rendered}\n")
    _emit_config_provenance(rendered=rendered, source_path=source_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
