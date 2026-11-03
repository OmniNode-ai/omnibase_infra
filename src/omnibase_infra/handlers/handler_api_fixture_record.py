# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HTTP capture effect with fail-closed coverage, masking and lifecycle checks.

The command contains paths only. A private overlay owns target addresses,
credential references, seeded resources and cleanup. The terminal event never
contains captured bodies. No request is made until the complete plan validates.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from contextvars import ContextVar
from ipaddress import ip_address
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from pydantic import JsonValue, TypeAdapter

from omnibase_core.utils.util_safe_yaml_loader import load_yaml_mapping_no_duplicates
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_api_fixture_http_step import ModelApiFixtureHttpStep
from omnibase_infra.models.model_api_fixture_overlay import ModelApiFixtureOverlay
from omnibase_infra.models.model_api_fixture_record_request import (
    ModelApiFixtureRecordRequest,
)
from omnibase_infra.models.model_api_fixture_record_result import (
    ModelApiFixtureRecordResult,
)

_OBJECT = TypeAdapter(dict[str, JsonValue])
_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_ENV = re.compile(r"[A-Z][A-Z0-9_]*")


def _policy(path: Path) -> dict[str, JsonValue]:
    contract = load_yaml_mapping_no_duplicates(path.read_text(), source=path.name)
    return _object(_OBJECT.validate_python(contract["config"])["recording_policy"])


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("expected_object")
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError("expected_array")
    return value


def _pointer_segment(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def inventory_identities(
    fixture: dict[str, JsonValue], policy: dict[str, JsonValue]
) -> Counter[tuple[str, str]]:
    """Read mounted identities from the inventory without copying its metadata."""
    result: Counter[tuple[str, str]] = Counter()
    for value in _array(fixture.get("routes")):
        route = _object(value)
        method, path = route.get("method"), route.get("path")
        if not isinstance(method, str) or not isinstance(path, str):
            raise ValueError("invalid_route_identity")
        if method not in {*_READ_METHODS, "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("invalid_route_method")
        if not path.startswith("/"):
            raise ValueError("invalid_route_path")
        result[(method, path)] += 1
    if len(result) != policy["expected_route_count"] or any(
        count != 1 for count in result.values()
    ):
        raise ValueError("inventory_identity_count")
    return result


def _rule(key: str) -> str:
    name = key.lower().replace("-", "_")
    if any(
        word in name
        for word in (
            "secret",
            "token",
            "password",
            "authorization",
            "cookie",
            "signature",
            "api_key",
            "credential",
        )
    ):
        return "credential"
    if any(word in name for word in ("name", "email", "slug", "principal", "subject")):
        return "identity"
    if name in {"id", "sub", "jti"} or name.endswith(("_id", "_ref")):
        return "identifier"
    if name in {
        "date",
        "iat",
        "exp",
        "nbf",
        "created",
        "updated",
        "timestamp",
    } or name.endswith(("_at", "_seconds", "_ms")):
        return "volatile"
    return "private_text"


def mask_value(
    value: JsonValue,
    *,
    pointer: str,
    approved_keys: frozenset[str],
    masks: dict[str, str],
    key: str = "",
) -> JsonValue:
    """Preserve JSON structure; mask text and sensitive scalars with named rules.

    Free-form object keys are refused because a key can itself be a name or a
    token. Non-JSON bodies are treated as one private string, never persisted raw.
    """
    if isinstance(value, dict):
        if not value.keys() <= approved_keys:
            raise ValueError("unapproved_body_key")
        return {
            field: mask_value(
                child,
                pointer=pointer + "/" + _pointer_segment(field),
                approved_keys=approved_keys,
                masks=masks,
                key=field,
            )
            for field, child in value.items()
        }
    if isinstance(value, list):
        return [
            mask_value(
                child,
                pointer=f"{pointer}/{index}",
                approved_keys=approved_keys,
                masks=masks,
                key=key,
            )
            for index, child in enumerate(value)
        ]
    rule = _rule(key)
    if isinstance(value, str) or (value is not None and rule != "private_text"):
        masks[pointer] = rule
        return f"<MASK:{rule}>"
    return value


def _mask_annotations(value: JsonValue, pointer: str = "") -> dict[str, str]:
    found = {}
    if isinstance(value, str) and "<MASK:" in value:
        if not re.fullmatch(r"<MASK:[a-z_]+>", value):
            raise ValueError("malformed_mask")
        found[pointer] = value[6:-1]
    elif isinstance(value, dict):
        for key, child in value.items():
            found.update(
                _mask_annotations(child, pointer + "/" + _pointer_segment(key))
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_mask_annotations(child, f"{pointer}/{index}"))
    return found


def _no_raw_text(value: JsonValue) -> bool:
    if isinstance(value, str):
        return bool(re.fullmatch(r"<MASK:[a-z_]+>", value))
    if isinstance(value, dict):
        return all(_no_raw_text(child) for child in value.values())
    if isinstance(value, list):
        return all(_no_raw_text(child) for child in value)
    return True


def validate_recorded_set(
    fixture: dict[str, JsonValue],
    recorded: dict[str, JsonValue],
    policy: dict[str, JsonValue],
) -> None:
    """AC1 and AC2 are non-vacuous and reject duplicates as well as missing pairs."""
    expected = inventory_identities(fixture, policy)
    pairs = [_object(pair) for pair in _array(recorded.get("pairs"))]
    if not pairs:
        raise ValueError("empty_recorded_set")
    actual: Counter[tuple[str, str]] = Counter()
    for pair in pairs:
        method, path = pair.get("method"), pair.get("path")
        if not isinstance(method, str) or not isinstance(path, str):
            raise ValueError("invalid_route_identity")
        actual[(method, path)] += 1
    if actual != expected:
        raise ValueError("recorded_identity_mismatch")
    rules = _object(policy["masking_rules"])
    cleanup = _array(recorded.get("cleanup_names"))
    if any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in cleanup):
        raise ValueError("invalid_cleanup_name")
    for pair in pairs:
        request, response = _object(pair.get("request")), _object(pair.get("response"))
        if set(request) != {"method", "path", "query", "headers", "body"}:
            raise ValueError("invalid_recorded_request")
        if request["method"] != pair["method"] or request["path"] != pair["path"]:
            raise ValueError("request_identity_mismatch")
        if set(response) != {"status", "headers", "body"}:
            raise ValueError("invalid_recorded_response")
        status = response["status"]
        if type(status) is not int or not 100 <= status <= 599:
            raise ValueError("invalid_response_status")
        found = _mask_annotations({"request": request, "response": response})
        if (
            _object(pair.get("masks")) != found
            or not set(found.values()) <= rules.keys()
        ):
            raise ValueError("unnamed_mask")
        for captured in (request, response):
            if not _no_raw_text(captured["body"]) or not _no_raw_text(
                captured["headers"]
            ):
                raise ValueError("unmasked_private_text")
        if not _no_raw_text(request["query"]):
            raise ValueError("unmasked_private_text")
        names = _array(pair.get("cleanup"))
        if pair["method"] not in _READ_METHODS and not names:
            raise ValueError("mutation_cleanup_missing")
        if any(name not in cleanup for name in names):
            raise ValueError("cleanup_not_declared")


def _expand(value: JsonValue, variables: dict[str, str]) -> JsonValue:
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, dict):
        return {key: _expand(child, variables) for key, child in value.items()}
    if isinstance(value, list):
        return [_expand(child, variables) for child in value]
    return value


# Credential values come from the private credentials file the request names,
# loaded once per handle() call; never from the process environment.
_CREDENTIALS: ContextVar[Mapping[str, str] | None] = ContextVar(
    "api_fixture_credentials", default=None
)


def _load_credentials(path: str) -> dict[str, str]:
    """Read the private credentials file: a JSON object of reference -> value.

    The file must be owner-only (no group or other permission bits), because
    it holds throwaway account and signing values for the recording target.
    """
    if not path:
        return {}
    location = Path(path)
    if location.stat().st_mode & 0o077:
        raise ValueError("credentials_file_not_private")
    loaded: JsonValue = TypeAdapter(JsonValue).validate_json(location.read_bytes())
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and _ENV.fullmatch(key) and isinstance(value, str)
        for key, value in loaded.items()
    ):
        raise ValueError("credentials_file_invalid")
    return cast("dict[str, str]", loaded)


def _credential(reference: str) -> str:
    value = (_CREDENTIALS.get() or {}).get(reference, "")
    if not _ENV.fullmatch(reference) or not value:
        raise ValueError("credential_reference_unavailable")
    return value


def _validate_plan(
    overlay: ModelApiFixtureOverlay,
    fixture: dict[str, JsonValue],
    policy: dict[str, JsonValue],
) -> None:
    if not overlay.sanctioned or "api" not in overlay.targets:
        raise ValueError("sanctioned_target_required")
    expected = inventory_identities(fixture, policy)
    if Counter((route.step.method, route.path) for route in overlay.routes) != expected:
        raise ValueError("request_plan_identity_mismatch")
    for base in overlay.targets.values():
        parsed = urlsplit(base)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("invalid_target_origin")
        if parsed.path not in {"", "/"} or not parsed.hostname:
            raise ValueError("invalid_target_origin")
        try:
            loopback = ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname == "localhost"
        if parsed.scheme != "https" and not (
            parsed.scheme == "http"
            and loopback
            and overlay.target_kind == "throwaway_slot"
        ):
            raise ValueError("secure_target_required")
    names = [step.name for step in overlay.cleanup]
    if (
        not names
        or len(names) != len(set(names))
        or any(not _NAME.fullmatch(n) for n in names)
    ):
        raise ValueError("cleanup_plan_required")
    if Counter(probe.resource for probe in overlay.counts) != Counter(
        cast("list[str]", policy["count_resources"])
    ):
        raise ValueError("all_resource_counts_required")
    steps = [
        *overlay.seeds,
        *overlay.cleanup,
        *(route.step for route in overlay.routes),
        *(probe.step for probe in overlay.counts),
    ]
    for step in steps:
        if step.target not in overlay.targets or step.path.startswith("//"):
            raise ValueError("undeclared_step_target")
        if overlay.target_kind == "dev" and step.scope == "isolated_slot":
            raise ValueError("isolated_slot_required")
        if not step.expected_status or any(
            not 100 <= n <= 599 for n in step.expected_status
        ):
            raise ValueError("expected_status_required")
        for header, reference in step.credential_headers.items():
            if not re.fullmatch(r"[a-zA-Z0-9-]+", header):
                raise ValueError("invalid_credential_header")
            _credential(reference)
        for variable, pointer in step.bindings.items():
            if (
                not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", variable)
                or variable in {"scope", "delivery_id", "event_id"}
                or not pointer.startswith("/")
            ):
                raise ValueError("invalid_response_binding")
    if any(step.bindings for step in overlay.cleanup):
        raise ValueError("cleanup_must_not_bind_variables")
    for step in [*overlay.seeds, *(route.step for route in overlay.routes)]:
        if step.method not in _READ_METHODS and not step.cleanup:
            raise ValueError("mutation_cleanup_missing")
        if not set(step.cleanup) <= set(names):
            raise ValueError("cleanup_not_declared")
    for probe in overlay.counts:
        if probe.step.method != "GET":
            raise ValueError("count_must_be_read_only")
        if probe.step.bindings:
            raise ValueError("count_must_not_bind_variables")
        serialized = json.dumps(
            {
                "path": probe.step.path,
                "query": probe.step.query,
                "body": probe.step.body,
            }
        )
        if "{scope}" not in serialized:
            raise ValueError("count_scope_required")
    for route in overlay.routes:
        if route.webhook:
            if route.step.method != "POST" or not route.signing_secret_env:
                raise ValueError("webhook_recipe_invalid")
            _credential(route.signing_secret_env)


async def _send(
    client: httpx.AsyncClient,
    overlay: ModelApiFixtureOverlay,
    step: ModelApiFixtureHttpStep,
    variables: dict[str, str],
    *,
    signed: tuple[bytes, dict[str, str]] | None = None,
) -> httpx.Response:
    path = cast("str", _expand(step.path, variables))
    if path.startswith("//") or "?" in path or "#" in path:
        raise ValueError("invalid_request_path")
    headers = {
        header: _credential(ref) for header, ref in step.credential_headers.items()
    }
    headers["content-type"] = "application/json"
    body = json.dumps(_expand(step.body, variables), allow_nan=False).encode()
    if signed is not None:
        body, webhook_headers = signed
        headers.update(webhook_headers)
    response = await client.request(
        step.method,
        overlay.targets[step.target].rstrip("/") + path,
        params=cast("dict[str, str]", _expand(dict(step.query), variables)),
        headers=headers,
        content=body if step.body is not None or signed is not None else None,
    )
    if response.status_code not in step.expected_status:
        raise ValueError("unexpected_http_status")
    if step.bindings:
        body_value: JsonValue = TypeAdapter(JsonValue).validate_json(response.content)
        resolved: dict[str, str] = {}
        for variable, pointer in step.bindings.items():
            selected = body_value
            for segment in pointer[1:].split("/"):
                key = segment.replace("~1", "/").replace("~0", "~")
                selected = _object(selected)[key]
            if not isinstance(selected, str) and type(selected) is not int:
                raise ValueError("response_binding_must_be_scalar")
            resolved[variable] = str(selected)
        variables.update(resolved)
    return response


async def _counts(
    client: httpx.AsyncClient,
    overlay: ModelApiFixtureOverlay,
    variables: dict[str, str],
) -> dict[str, int]:
    result: dict[str, int] = {}
    failed = False
    for probe in overlay.counts:
        try:
            response = await _send(client, overlay, probe.step, variables)
            value: JsonValue = TypeAdapter(JsonValue).validate_json(response.content)
            if probe.pointer:
                if not probe.pointer.startswith("/"):
                    raise ValueError("invalid_count_pointer")
                for segment in probe.pointer[1:].split("/"):
                    key = segment.replace("~1", "/").replace("~0", "~")
                    value = _object(value)[key]
            if probe.measure == "length":
                value = len(_array(value))
            if type(value) is not int or value < 0:
                raise ValueError("invalid_resource_count")
            result[probe.resource] = value
        except (ValueError, KeyError, httpx.HTTPError):
            failed = True
    if failed:
        raise ValueError("resource_count_unavailable")
    return result


def _capture(
    method: str,
    path: str,
    response: httpx.Response,
    overlay: ModelApiFixtureOverlay,
    policy: dict[str, JsonValue],
) -> tuple[dict[str, JsonValue], dict[str, JsonValue], dict[str, str]]:
    masks: dict[str, str] = {}
    approved = frozenset(overlay.approved_body_keys)
    try:
        request_body: JsonValue = TypeAdapter(JsonValue).validate_json(
            response.request.content
        )
    except ValueError:
        request_body = response.request.content.decode(errors="replace") or None
    try:
        response_body: JsonValue = TypeAdapter(JsonValue).validate_json(
            response.content
        )
    except ValueError:
        response_body = response.text or None
    request_headers: dict[str, JsonValue] = {}
    for name, value in response.request.headers.items():
        # Host and transport defaults are not replay inputs. Credential and
        # signature header values are retained only as named markers.
        if name in {
            "host",
            "accept",
            "accept-encoding",
            "connection",
            "user-agent",
            "content-length",
        }:
            continue
        rule = _rule(name)
        masks["/request/headers/" + _pointer_segment(name)] = rule
        request_headers[name] = f"<MASK:{rule}>"
    response_headers: dict[str, JsonValue] = {}
    for name in cast("list[str]", policy["response_headers"]):
        if name in response.headers:
            rule = _rule(name)
            masks["/response/headers/" + _pointer_segment(name)] = rule
            response_headers[name] = f"<MASK:{rule}>"
    request: dict[str, JsonValue] = {
        "method": method,
        "path": path,
        "query": {},
        "headers": request_headers,
        "body": mask_value(
            request_body, pointer="/request/body", approved_keys=approved, masks=masks
        ),
    }
    query: dict[str, JsonValue] = {}
    for name in response.request.url.params:
        query[name] = mask_value(
            list(response.request.url.params.get_list(name)),
            pointer="/request/query/" + _pointer_segment(name),
            approved_keys=approved,
            masks=masks,
            key=name,
        )
    request["query"] = query
    captured: dict[str, JsonValue] = {
        "status": response.status_code,
        "headers": response_headers,
        "body": mask_value(
            response_body, pointer="/response/body", approved_keys=approved, masks=masks
        ),
    }
    return request, captured, masks


def _private_destination(path: Path, directory: Path) -> Path:
    root = directory.resolve(strict=True)
    public_root = Path(__file__).resolve().parents[3]
    if root.is_relative_to(public_root) or not path.resolve().is_relative_to(root):
        raise ValueError("private_output_directory_required")
    if path.is_symlink() or path.exists() or path.parent.resolve() != root:
        raise ValueError("fresh_private_output_required")
    return root


def _write_private(path: Path, directory: Path, document: dict[str, JsonValue]) -> None:
    root = _private_destination(path, directory)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, delete=False
        ) as stream:
            temporary = stream.name
            os.fchmod(stream.fileno(), 0o600)
            json.dump(document, stream, allow_nan=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and refuses an existing destination, unlike replace.
        os.link(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def build_signed_webhook(
    kind: str, *, secret: bytes, delivery: UUID, timestamp: int
) -> tuple[bytes, dict[str, str]]:
    """Sign exact raw bytes: GitHub ping or an inert Stripe fixture event.

    Stripe records even an unhandled event, so its request recipe must name
    cleanup for the planted event row. GitHub ping performs signature checking
    without publishing a delivery into a tenant's workflow stream.
    """
    if not secret:
        raise ValueError("signing_credential_missing")
    if kind == "github":
        body = json.dumps({"zen": "fixture-probe", "hook_id": 0}).encode()
        signature = hmac.new(secret, body, hashlib.sha256).hexdigest()
        return body, {
            "x-github-event": "ping",
            "x-github-delivery": str(delivery),
            "x-hub-signature-256": "sha256=" + signature,
        }
    if kind == "stripe":
        body = json.dumps(
            {
                "id": "evt_" + delivery.hex,
                "object": "event",
                "type": "fixture.recording",
                "created": timestamp,
                "data": {"object": {}},
            },
            separators=(",", ":"),
        ).encode()
        signed = str(timestamp).encode() + b"." + body
        signature = hmac.new(secret, signed, hashlib.sha256).hexdigest()
        return body, {"stripe-signature": f"t={timestamp},v1={signature}"}
    raise ValueError("unsupported_webhook_kind")


class HandlerApiFixtureRecord:
    """Capture via the contract's dispatch route; clean up even after a failed call."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelApiFixtureRecordRequest
    ) -> ModelApiFixtureRecordResult:
        try:
            policy = _policy(Path(request.policy_path))
            fixture_bytes = Path(request.fixture_path).read_bytes()
            fixture = _OBJECT.validate_json(fixture_bytes)
            if request.mode == "validate":
                recorded = _OBJECT.validate_json(
                    Path(request.recorded_path).read_bytes()
                )
                validate_recorded_set(fixture, recorded, policy)
                if (
                    recorded.get("fixture_sha256")
                    != hashlib.sha256(fixture_bytes).hexdigest()
                ):
                    raise ValueError("fixture_digest_mismatch")
                lifecycle = _object(recorded.get("lifecycle"))
                before, after = (
                    _object(lifecycle.get("before")),
                    _object(lifecycle.get("after")),
                )
                if (
                    set(before) != set(cast("list[str]", policy["count_resources"]))
                    or before != after
                ):
                    raise ValueError("resource_count_delta")
                if lifecycle.get("cleanup_completed") is not True or any(
                    type(n) is not int or n < 0 for n in before.values()
                ):
                    raise ValueError("cleanup_not_completed")
                return ModelApiFixtureRecordResult(
                    success=True,
                    pair_count=len(_array(recorded["pairs"])),
                    before=cast("dict[str, int]", before),
                    after=cast("dict[str, int]", after),
                    cleanup_completed=True,
                )
            overlay = ModelApiFixtureOverlay.model_validate_json(
                Path(request.overlay_path).read_bytes()
            )
            _CREDENTIALS.set(_load_credentials(request.credentials_path))
            _validate_plan(overlay, fixture, policy)
            _private_destination(
                Path(request.recorded_path), Path(overlay.private_output_directory)
            )
            return await self._record(
                request,
                overlay,
                fixture,
                hashlib.sha256(fixture_bytes).hexdigest(),
                policy,
            )
        except (ValueError, OSError, KeyError, httpx.HTTPError):
            # Exceptions can contain response text, a concrete path, or a
            # credential. Never reflect their text into the terminal event.
            return ModelApiFixtureRecordResult(
                success=False, error_code="recording_precondition_failed"
            )

    async def _record(
        self,
        request: ModelApiFixtureRecordRequest,
        overlay: ModelApiFixtureOverlay,
        fixture: dict[str, JsonValue],
        digest: str,
        policy: dict[str, JsonValue],
    ) -> ModelApiFixtureRecordResult:
        delivery = uuid4()
        variables = {
            "scope": overlay.scope_prefix,
            "tenant_id": str(uuid4()),
            "delivery_id": str(delivery),
            "event_id": "evt_" + delivery.hex,
        }
        pairs: list[JsonValue] = []
        before: dict[str, int] = {}
        after: dict[str, int] = {}
        cleanup_ok = True
        capture_ok = False
        # No idle keep-alive connection: a route that fails inside the service
        # answers 500 and closes its connection, which would otherwise break the
        # next request on the pooled connection and abort the recording.
        async with httpx.AsyncClient(
            timeout=overlay.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client:
            # A missing baseline stops before the first mutation.
            before = await _counts(client, overlay, variables)
            try:
                for step in overlay.seeds:
                    await _send(client, overlay, step, variables)
                for route in overlay.routes:
                    signed = None
                    if route.webhook:
                        signed = build_signed_webhook(
                            route.webhook,
                            secret=_credential(route.signing_secret_env).encode(),
                            delivery=delivery,
                            timestamp=int(time.time()),
                        )
                    response = await _send(
                        client, overlay, route.step, variables, signed=signed
                    )
                    captured_request, captured_response, masks = _capture(
                        route.step.method, route.path, response, overlay, policy
                    )
                    pairs.append(
                        {
                            "method": route.step.method,
                            "path": route.path,
                            "request": captured_request,
                            "response": captured_response,
                            "masks": dict(masks),
                            "cleanup": list(route.step.cleanup),
                        }
                    )
                capture_ok = True
            except (ValueError, OSError, KeyError, httpx.HTTPError):
                capture_ok = False
            finally:
                # Every cleanup is predeclared, including cleanup for a seed or
                # request that created a resource and then lost its response.
                # One failed cleanup never prevents the remaining cleanups or
                # the final count reads. Shield this phase from cancellation.
                async def finish() -> tuple[bool, dict[str, int]]:
                    clean = True
                    for step in reversed(overlay.cleanup):
                        try:
                            await _send(client, overlay, step, variables)
                        except (ValueError, OSError, KeyError, httpx.HTTPError):
                            clean = False
                    try:
                        counts = await _counts(client, overlay, variables)
                    except (ValueError, OSError, KeyError, httpx.HTTPError):
                        return False, {}
                    return clean, counts

                task = asyncio.create_task(finish())
                try:
                    cleanup_ok, after = await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
        if not cleanup_ok or before != after or not capture_ok:
            code = (
                "cleanup_failed"
                if not cleanup_ok
                else "resource_count_delta"
                if before != after
                else "capture_failed"
            )
            return ModelApiFixtureRecordResult(
                success=False,
                pair_count=len(pairs),
                error_code=code,
                before=before,
                after=after,
                cleanup_completed=cleanup_ok,
            )
        document: dict[str, JsonValue] = {
            "schema_version": 1,
            "fixture_sha256": digest,
            "pairs": pairs,
            "cleanup_names": [step.name for step in overlay.cleanup],
            "lifecycle": {
                "before": dict(before),
                "after": dict(after),
                "cleanup_completed": True,
            },
        }
        validate_recorded_set(fixture, document, policy)
        _write_private(
            Path(request.recorded_path),
            Path(overlay.private_output_directory),
            document,
        )
        return ModelApiFixtureRecordResult(
            success=True,
            pair_count=len(pairs),
            before=before,
            after=after,
            cleanup_completed=True,
        )
