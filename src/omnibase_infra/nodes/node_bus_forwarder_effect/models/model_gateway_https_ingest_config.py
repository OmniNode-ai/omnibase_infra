# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolved config for the forwarder's OUTBOUND HTTPS ingest leg (OMN-16459).

Operator ruling 2026-08-24 (verbatim, via team-lead): *"the gateway on the cloud
should be configurable to point at whatever env we want. we can use the forwarder
to accelerate moving our work to the cloud, but it should be replaced with the
https doors as soon as possible."*

Operator scope ruling 2026-08-30 (verbatim): *"the cloud leg shouldn't need
anything new. all we need is one ingress for all calls right?"* -- ONE
authenticated batch-ingest route on the already-hosted cloud gateway. Hence a
single ``ingest_url_ref`` with a batch bound, never a per-event-class endpoint
map, and idempotency asserted on the route keyed on the content-addressed
envelope id rather than delegated to a dedupe service or to the sink.

SCOPE, stated here because the ticket's AC5 is easy to over-read: this leg
replaces the OUTBOUND publish boundary only. ``mirror_topics.inbound`` is still
a Kafka pull from the cloud broker, so declaring this block does NOT by itself
make the OMN-16449 dnsmasq bastion deletable.
"""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A contract reference is a dotted lowercase path (``gateway.cloud.https.ingest_url``).
# Anything else in a ``*_ref`` field is an operator pasting a value where a
# reference belongs -- a JWT, a base64 blob, a bare URL. Refused at load.
_REFERENCE_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")


class ModelGatewayHttpsIngestConfig(BaseModel):
    """The single contract-declared HTTPS ingest route for the outbound leg."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ingest_url: str
    ingest_url_ref: str
    ingest_auth_ref: str
    idempotency_key: Literal["envelope_id"]
    max_batch_records: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    retry_initial_seconds: float = Field(gt=0)
    retry_max_seconds: float = Field(gt=0)

    @field_validator("ingest_url")
    @classmethod
    def _validate_ingest_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https":
            raise ValueError(
                "gateway ingest_url must use the https scheme -- this leg exists "
                "to replace a broker-protocol path with the gateway's HTTPS door, "
                "and a cleartext door would carry tenant traffic unencrypted"
            )
        if not parsed.netloc:
            raise ValueError("gateway ingest_url must carry a host")
        if not parsed.path or parsed.path == "/":
            raise ValueError(
                "gateway ingest_url must name the ingest route path, not a bare host"
            )
        return value

    @field_validator("ingest_url_ref", "ingest_auth_ref")
    @classmethod
    def _validate_reference(cls, value: str) -> str:
        if not _REFERENCE_PATTERN.match(value):
            raise ValueError(
                f"{value[:12]!r}... is not a contract reference; gateway ingest "
                "credentials and addresses are named by dotted reference and "
                "resolved at the effect boundary -- a value in config is refused"
            )
        return value

    @model_validator(mode="after")
    def _validate_retry_window(self) -> ModelGatewayHttpsIngestConfig:
        if self.retry_max_seconds < self.retry_initial_seconds:
            raise ValueError(
                "retry_max_seconds must be greater than or equal to "
                "retry_initial_seconds"
            )
        return self

    @property
    def ingest_host(self) -> str:
        """Host of the resolved ingest route, for cross-leg distinctness checks."""
        return urlparse(self.ingest_url).netloc.split(":")[0]


__all__ = ["ModelGatewayHttpsIngestConfig"]
