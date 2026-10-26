# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Minimal etcd v3 client over the JSON gRPC gateway (OMN-21421).

Covers the calls placement needs: lease grant, single keepalive, revoke, an
atomic create-if-absent transaction, key and prefix reads. The gateway takes
base64 keys and values and returns 64-bit integers as JSON strings.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping

import httpx

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

_JsonObject = dict[str, object]


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _unb64(text: object) -> str:
    return base64.b64decode(str(text)).decode() if text else ""


def _prefix_end(prefix: str) -> str:
    raw = bytearray(prefix.encode())
    raw[-1] += 1
    return base64.b64encode(bytes(raw)).decode()


class HandlerPlacementEtcd:
    """Async etcd v3 JSON-gateway client; every request is bounded by ``timeout_s``."""

    def __init__(self, endpoint: str, *, timeout_s: float = 1.0) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, body: Mapping[str, object]) -> _JsonObject:
        try:
            response = await self._client.post(f"{self._endpoint}{path}", json=body)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=f"etcd{path}",
            )
            raise InfraConnectionError(
                f"etcd request {path} failed: {type(exc).__name__}", context=context
            ) from exc
        if not isinstance(payload, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=f"etcd{path}",
            )
            raise InfraConnectionError(
                f"etcd request {path} returned a non-object body", context=context
            )
        return payload

    async def lease_grant(self, ttl_s: int) -> tuple[int, int]:
        """Grant a lease; returns ``(lease_id, granted_ttl_s)``."""
        payload = await self._post("/v3/lease/grant", {"TTL": ttl_s})
        return int(str(payload["ID"])), int(str(payload["TTL"]))

    async def lease_keepalive(self, lease_id: int) -> int:
        """Renew a lease once; returns the remaining TTL, ``0`` when it has expired."""
        payload = await self._post("/v3/lease/keepalive", {"ID": str(lease_id)})
        result = payload.get("result")
        if not isinstance(result, dict):
            return 0
        return int(str(result.get("TTL", "0")))

    async def lease_revoke(self, lease_id: int) -> None:
        await self._post("/v3/lease/revoke", {"ID": str(lease_id)})

    async def put(self, key: str, value: str, *, lease_id: int) -> int:
        payload = await self._post(
            "/v3/kv/put",
            {"key": _b64(key), "value": _b64(value), "lease": str(lease_id)},
        )
        header = payload.get("header")
        return int(str(header.get("revision", "0"))) if isinstance(header, dict) else 0

    async def create_if_absent(
        self, key: str, value: str, *, lease_id: int
    ) -> tuple[bool, int]:
        """Atomically create ``key`` bound to ``lease_id`` when it does not exist.

        Returns ``(created, revision)``; on success the revision is the store
        revision of the grant, which is the election epoch.
        """
        payload = await self._post(
            "/v3/kv/txn",
            {
                "compare": [
                    {
                        "key": _b64(key),
                        "target": "CREATE",
                        "result": "EQUAL",
                        "createRevision": "0",
                    }
                ],
                "success": [
                    {
                        "requestPut": {
                            "key": _b64(key),
                            "value": _b64(value),
                            "lease": str(lease_id),
                        }
                    }
                ],
                "failure": [],
            },
        )
        header = payload.get("header")
        revision = (
            int(str(header.get("revision", "0"))) if isinstance(header, dict) else 0
        )
        return bool(payload.get("succeeded", False)), revision

    async def get(self, key: str) -> tuple[str, int] | None:
        """Read one key; returns ``(value, create_revision)`` or ``None``."""
        payload = await self._post("/v3/kv/range", {"key": _b64(key)})
        kvs = payload.get("kvs")
        if not isinstance(kvs, list) or not kvs:
            return None
        entry = kvs[0]
        if not isinstance(entry, dict):
            return None
        return _unb64(entry.get("value")), int(str(entry.get("create_revision", "0")))

    async def get_prefix(self, prefix: str) -> dict[str, str]:
        payload = await self._post(
            "/v3/kv/range", {"key": _b64(prefix), "range_end": _prefix_end(prefix)}
        )
        kvs = payload.get("kvs")
        if not isinstance(kvs, list):
            return {}
        return {
            _unb64(entry.get("key")): _unb64(entry.get("value"))
            for entry in kvs
            if isinstance(entry, dict)
        }


__all__ = ["HandlerPlacementEtcd"]
