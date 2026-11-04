# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The OTLP HTTP traces URL for a configured base endpoint (OMN-22097)."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, field_validator

_TRACES_PATH = "/v1/traces"


class ModelOtlpTracesEndpoint(BaseModel):
    """A base endpoint URL and the traces URL ``OTLPSpanExporter`` must be given.

    ``OTLPSpanExporter(endpoint=...)`` uses its value verbatim, so the signal path
    is added to the URL path: a path prefix is kept, a trailing ``/v1/traces`` is
    not doubled, and the query string stays after the path. A value that is not
    an http(s) URL with a host is refused; the error never echoes the value.
    The signal-specific ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` is not read: the
    runtime's endpoint is the single contract-declared value.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        hide_input_in_errors=True,
    )

    base_url: str

    @field_validator("base_url")
    @classmethod
    def _require_http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("OTLP endpoint must be an http(s) URL with a host")
        return value

    @property
    def traces_url(self) -> str:
        parts = urlsplit(self.base_url)
        path = parts.path.rstrip("/").removesuffix(_TRACES_PATH)
        return urlunsplit(
            (parts.scheme, parts.netloc, f"{path}{_TRACES_PATH}", parts.query, "")
        )


__all__ = ["ModelOtlpTracesEndpoint"]
