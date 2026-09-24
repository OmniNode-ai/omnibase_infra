# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16987 -- the provider-rung canary OBSERVER. Runs INSIDE the deployed runtime.

WHAT THIS IS
    The half of the provider-rung liveness canary that executes in the deployed
    interpreter of the dev lane's ``omninode-runtime-effects`` container, as
    the runtime's own user. ``provider_rung_canary_probe.py`` pipes this file
    to ``docker exec -i -u <user> <container> python - <contract>`` and grades
    what it prints. This file GRADES NOTHING: it reports raw facts per request
    (status, exception family, response shape, provider error code) and the
    verdict is decided on the runner, by a grader tested offline against
    recorded observations.

WHY INSIDE THE RUNTIME
    Resolvability is not liveness. The runtime resolves a backend's key NAME
    through its own lane secret mapping, and a key that resolves can still be
    unfunded, entitled on the wrong product, expired, or quota-capped. The only
    reading that answers "would a delegation on this rung authenticate right
    now" is one taken with the runtime's own contract, its own resolver and its
    own transport. All three are imported from the deployed package here:

      * the contract: ``load_bifrost_delegation_config`` over the path the
        runtime binds as ``BIFROST_CONTRACT_PATH``, named by the caller;
      * the key: ``resolve_api_key_with_source_async`` with the same
        ``secret_ref`` / ``api_key_env`` pair the delegation call effect
        passes, ``required=False`` so a miss is REPORTED, not raised;
      * the request: ``post_chat_completion``, which posts the contract URL
        verbatim.

ONE REQUEST PER DISTINCT RUNG, PLUS ONE WRONG-KEY CONTROL PER ENDPOINT
    Backends that share an endpoint, a model and a key are one rung for this
    purpose (``cloud-gemini-pro`` and ``cloud-glm-judge`` do today), so the
    canary sends ONE minimal completion (``max_tokens`` 8) for them and credits
    the answer to each. A free-tier cap measured in tens of requests a day is
    exactly what an over-eager liveness probe would itself exhaust.

    Each distinct endpoint also receives ONE request carrying a deliberately
    invalid key. The grader requires that request to be refused as an auth
    failure. That is what makes a green readable: a probe that reads 200 no
    matter what credential it sends is not measuring the credential, and this
    control catches it in the same run.

A DECLARED KEY THAT DOES NOT RESOLVE SENDS NOTHING
    No request is issued for such a rung (an unauthenticated call to a paid
    provider is never a probe). The observation says ``secret_resolved:
    false`` and the grader records ``UNRESOLVED``: never LIVE, never PASS.

NO KEY MATERIAL LEAVES THIS PROCESS
    Every resolved value is held in memory only. The single JSON line this
    prints is scrubbed of every resolved value (and of the deliberately invalid
    key) before it is written, and the number of scrubs is reported so a
    provider that echoes a key back is visible rather than silently cleaned.
    Nothing is logged.

OUTPUT
    Exactly one JSON object on stdout, last line. A failure to load the
    contract or import the subject is reported as ``{"error": ...}`` so the
    runner can tell "could not look" (exit 2) from "a rung is dead" (exit 1).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import sys
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

# A trivial request. The answer is never graded, only that one came back.
PROMPT = "Reply with the single word OK."
MAX_TOKENS = 8
# Hard ceiling on one probe request, whatever the contract's per-backend
# timeout says: a liveness answer that takes longer than this is not one.
MAX_TIMEOUT_SECONDS = 30.0

# The deliberately invalid credential for the wrong-key control. It is not a
# secret: its whole purpose is to be refused.
INVALID_KEY = "onex-rung-canary-deliberately-invalid-key-0000"  # nosec B105

REDACTED = "<redacted>"

SKIPPED_NO_ENDPOINT = "SKIPPED_NO_ENDPOINT"
SKIPPED_NO_SECRET_REF = "SKIPPED_NO_SECRET_REF"
PROBE = "PROBE"

ResolveFn = Callable[[str, str | None], Awaitable[str | None]]
# (endpoint_url, payload, headers, timeout_seconds) -> raw facts
PostFn = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def _exc(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:600]}


def backend_secret_ref(backend: dict[str, Any]) -> str | None:
    """The same canonical ref ``ModelDelegationBackendConfig.resolved_secret_ref`` returns."""
    for key in ("secret_ref", "api_key_ref", "api_key_env"):
        value = backend.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def probe_key(backend: dict[str, Any]) -> str:
    """Backends sharing endpoint, model and key are ONE rung for liveness."""
    return "|".join(
        [
            str(backend.get("endpoint_url")),
            str(backend.get("model_name")),
            str(backend_secret_ref(backend)),
            str(backend.get("api_key_env")),
        ]
    )


def plan(
    backends: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify every declared backend and group the probed ones into rungs.

    Returns ``(rows, probes)``: one row per declared backend, and one probe per
    distinct rung, in declaration order.
    """
    rows: list[dict[str, Any]] = []
    probes: dict[str, dict[str, Any]] = {}
    for backend in backends:
        ref = backend_secret_ref(backend)
        endpoint = backend.get("endpoint_url")
        if not isinstance(endpoint, str) or not endpoint.strip():
            disposition = SKIPPED_NO_ENDPOINT
        elif ref is None:
            disposition = SKIPPED_NO_SECRET_REF
        else:
            disposition = PROBE
        key = probe_key(backend) if disposition == PROBE else None
        rows.append(
            {
                "backend_id": backend.get("backend_id"),
                "provider": backend.get("provider"),
                "tier": backend.get("tier"),
                "endpoint_url": endpoint,
                "model_name": backend.get("model_name"),
                "secret_ref": ref,
                "disposition": disposition,
                "probe_key": key,
            }
        )
        if key is None:
            continue
        if key not in probes:
            timeout_ms = backend.get("timeout_ms")
            timeout = (
                min(float(timeout_ms) / 1000.0, MAX_TIMEOUT_SECONDS)
                if isinstance(timeout_ms, int | float) and timeout_ms > 0
                else MAX_TIMEOUT_SECONDS
            )
            probes[key] = {
                "probe_key": key,
                "backend_ids": [],
                "endpoint_url": endpoint,
                "model_name": backend.get("model_name"),
                "secret_ref": ref,
                "api_key_env": backend.get("api_key_env"),
                "extra_headers": dict(backend.get("extra_headers") or {}),
                "timeout_seconds": timeout,
            }
        probes[key]["backend_ids"].append(backend.get("backend_id"))
    return rows, list(probes.values())


def payload_for(model_name: Any) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }


def _headers(extra: dict[str, str], key: str) -> dict[str, str]:
    headers = dict(extra)
    headers["Authorization"] = f"Bearer {key}"
    return headers


def _public(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe_key": probe["probe_key"],
        "backend_ids": list(probe["backend_ids"]),
        "endpoint_url": probe["endpoint_url"],
        "endpoint_host": urlsplit(str(probe["endpoint_url"])).hostname,
        "model_name": probe["model_name"],
        "secret_ref": probe["secret_ref"],
    }


async def observe_probes(
    probes: list[dict[str, Any]],
    *,
    resolve: ResolveFn,
    post: PostFn,
    secrets: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one authenticated request per rung and one wrong-key request per endpoint.

    ``secrets`` collects every resolved value so the caller can scrub output.
    ``resolve`` and ``post`` are injected: the deployed ones in production, fakes
    in the offline tests.
    """
    results: list[dict[str, Any]] = []
    for probe in probes:
        out = _public(probe)
        try:
            value = await resolve(probe["secret_ref"], probe["api_key_env"])
        except Exception as exc:  # noqa: BLE001 - the refusal IS the observation
            value = None
            out["resolver_error"] = type(exc).__name__
        if not value:
            out["secret_resolved"] = False
            out["request_sent"] = False
            results.append(out)
            continue
        secrets.add(value)
        out["secret_resolved"] = True
        out["request_sent"] = True
        out.update(
            post(
                probe["endpoint_url"],
                payload_for(probe["model_name"]),
                _headers(probe["extra_headers"], value),
                probe["timeout_seconds"],
            )
        )
        results.append(out)

    controls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for probe in probes:
        endpoint = probe["endpoint_url"]
        if endpoint in seen:
            continue
        seen.add(endpoint)
        ctl = {
            "endpoint_url": endpoint,
            "endpoint_host": urlsplit(str(endpoint)).hostname,
            "model_name": probe["model_name"],
            "credential": "deliberately_invalid",
            "request_sent": True,
        }
        ctl.update(
            post(
                endpoint,
                payload_for(probe["model_name"]),
                _headers(probe["extra_headers"], INVALID_KEY),
                probe["timeout_seconds"],
            )
        )
        controls.append(ctl)
    return results, controls


def error_fields(body: Any) -> dict[str, Any]:
    """Provider error code, status and message out of an error body, any shape.

    z.ai answers ``{"error": {"code": "1310", "message": ...}}``; the Gemini
    OpenAI-compatible surface wraps the same in a one-element list; OpenRouter
    uses an integer code. Only the fields are lifted; nothing is interpreted.
    """
    if isinstance(body, list) and body:
        body = body[0]
    if not isinstance(body, dict):
        return {}
    err = body.get("error", body)
    if not isinstance(err, dict):
        return {"provider_error_message": str(err)[:300]}
    out: dict[str, Any] = {}
    code = err.get("code")
    if code is not None:
        out["provider_error_code"] = str(code)
    status = err.get("status") or err.get("type")
    if status is not None:
        out["provider_error_status"] = str(status)
    message = err.get("message")
    if message is not None:
        out["provider_error_message"] = str(message)[:300]
    return out


def response_fields(status: int, body: Any, latency_ms: int | None) -> dict[str, Any]:
    out: dict[str, Any] = {"http_status": status, "latency_ms": latency_ms}
    if 200 <= status < 300:
        choices = body.get("choices") if isinstance(body, dict) else None
        model = body.get("model") if isinstance(body, dict) else None
        out["body_is_chat_completion"] = isinstance(choices, list) and bool(choices)
        out["model_echo"] = model if isinstance(model, str) and model else None
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            out["usage"] = {
                k: usage[k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance(usage.get(k), int)
            }
    else:
        out.update(error_fields(body))
    return out


def deployed_post() -> PostFn:
    """The runtime's own transport, wrapped to report facts instead of raising."""
    import time

    import httpx
    from omnimarket.nodes.node_llm_delegation_call_effect.handlers import transport

    def post(
        endpoint_url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            resp = transport.post_chat_completion(
                endpoint_url=endpoint_url,
                payload=payload,
                timeout_seconds=timeout,
                extra_headers=headers,
            )
        except httpx.HTTPStatusError as exc:
            try:
                body: Any = exc.response.json()
            except ValueError:
                body = exc.response.text[:300]
            return response_fields(
                exc.response.status_code,
                body,
                int((time.monotonic() - t0) * 1000),
            )
        except (httpx.TransportError, OSError) as exc:
            return {"exception": type(exc).__name__, "exception_family": "transport"}
        except ValueError as exc:
            # A 2xx whose body is not JSON (JSONDecodeError is a ValueError),
            # or a contract URL the transport refuses as not http(s).
            return {"exception": type(exc).__name__, "exception_family": "decode"}
        except Exception as exc:  # noqa: BLE001 - reported, graded as a probe error
            return {"exception": type(exc).__name__, "exception_family": "other"}
        return response_fields(resp.status_code, resp.json_body, resp.latency_ms)

    return post


def deployed_resolve() -> ResolveFn:
    from omnimarket.inference.secret_store_resolver import (
        resolve_api_key_with_source_async,
    )

    async def resolve(ref: str, env_fallback: str | None) -> str | None:
        value, _source = await resolve_api_key_with_source_async(
            ref, required=False, env_var_fallback=env_fallback
        )
        return value.get_secret_value() if value is not None else None

    return resolve


def quota_policy(config: Any) -> list[dict[str, Any]]:
    policy = getattr(config, "provider_quota_policy", None)
    if policy is None:
        return []
    return [
        {
            "provider_id": p.provider_id,
            "match_endpoint_host": p.match_endpoint_host,
            "codes": [
                {
                    "code": c.code,
                    "disposition": str(getattr(c.disposition, "value", c.disposition)),
                }
                for c in p.codes
            ],
        }
        for p in policy.providers
    ]


def scrub(text: str, secrets: set[str]) -> tuple[str, int]:
    count = 0
    for value in sorted(secrets, key=len, reverse=True):
        if value and value in text:
            count += text.count(value)
            text = text.replace(value, REDACTED)
    return text, count


def observe(contract_path: str, secrets: set[str]) -> dict[str, Any]:
    from pathlib import Path

    from omnimarket.adapters.llm.bifrost.config_loader_bifrost_delegation import (
        load_bifrost_delegation_config,
    )

    config = load_bifrost_delegation_config(config_path=Path(contract_path))
    backends = [b.model_dump(mode="json") for b in config.backends]
    rows, probes = plan(backends)
    secrets.add(INVALID_KEY)
    results, controls = asyncio.run(
        observe_probes(
            probes, resolve=deployed_resolve(), post=deployed_post(), secrets=secrets
        )
    )
    obs = {
        "omnimarket_version": importlib.metadata.version("omnimarket"),
        "contract_path": contract_path,
        # The runtime's own binding, compared here so the grader can refuse a
        # reading taken from a contract the runtime does not route with.
        "runtime_binds_contract": os.environ.get("BIFROST_CONTRACT_PATH")
        == contract_path,
        "config_version": config.config_version,
        "backends": rows,
        "probes": results,
        "controls": controls,
        "quota_policy": quota_policy(config),
    }
    return obs


def render(obs: dict[str, Any], secrets: set[str]) -> str:
    """The one output line, with every secret value scrubbed and counted.

    The invalid control key is scrubbed too, but it is not counted: a provider
    quoting a rejected credential back is not a leak of a real one.
    """
    text = json.dumps(obs, sort_keys=True)
    real = {s for s in secrets if s != INVALID_KEY}
    text, leaked = scrub(text, real)
    text, _ = scrub(text, {INVALID_KEY})
    if leaked:
        obs = json.loads(text)
        obs["redactions"] = leaked
        text = json.dumps(obs, sort_keys=True)
    return text


def main() -> int:
    contract_path = sys.argv[1] if len(sys.argv) > 1 else ""
    secrets: set[str] = set()
    try:
        if not contract_path:
            raise ValueError("contract path argument is required")
        obs = observe(contract_path, secrets)
        line = render(obs, secrets)
    except Exception as exc:  # noqa: BLE001 - reported, graded as could-not-run
        line, _ = scrub(json.dumps({"error": _exc(exc)}), secrets | {INVALID_KEY})
    sys.stdout.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
