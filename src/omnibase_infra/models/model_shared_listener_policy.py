# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Policy values of the runtime shared listener (OMN-22377)."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The runtime reads the values the compose policy renders into its process
# environment, like every other rendered runtime policy value.
ENV_PREFIX = "ONEX_RUNTIME_"


class ModelSharedListenerPolicy(BaseModel):
    """Bounds of the shared listener, all declared by policy and never defaulted.

    Attributes:
        pool: Pool name the broker keys the listening socket by.
        listener_first_request_seconds: How long a connection that has not sent
            its first request is kept open while draining (A3).
        last_attached_hold_seconds: How long the last confirmed acceptor holds
            a request while not ready (A6).
        detach_successor_wait_seconds: How long a stopping runtime waits for a
            ``granted`` before it detaches anyway (A6).
        passive_ready_watch_ms: Readiness monitor period (A2); also the retry
            period for attach and detach asks.
        probe_client_timeout_seconds: The probe client's own timeout (5 for the
            ``curl --max-time 5`` of AC4).
        response_margin_seconds: Margin the hold bound leaves inside the probe
            client timeout.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    pool: str = Field(min_length=1)
    listener_first_request_seconds: float = Field(gt=0)
    last_attached_hold_seconds: float = Field(gt=0)
    detach_successor_wait_seconds: float = Field(gt=0)
    passive_ready_watch_ms: int = Field(gt=0)
    probe_client_timeout_seconds: float = Field(gt=0)
    response_margin_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _hold_fits_inside_probe_client_timeout(self) -> ModelSharedListenerPolicy:
        needed = self.last_attached_hold_seconds + self.response_margin_seconds
        if needed >= self.probe_client_timeout_seconds:
            msg = (
                f"last_attached_hold_seconds ({self.last_attached_hold_seconds}) "
                f"plus response_margin_seconds ({self.response_margin_seconds}) must "
                f"be under probe_client_timeout_seconds "
                f"({self.probe_client_timeout_seconds})"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> ModelSharedListenerPolicy:
        """Build the policy from the rendered process environment; fail loudly."""
        names = {
            "pool": "HANDOVER_POOL",
            "listener_first_request_seconds": "LISTENER_FIRST_REQUEST_SECONDS",
            "last_attached_hold_seconds": "LAST_ATTACHED_HOLD_SECONDS",
            "detach_successor_wait_seconds": "DETACH_SUCCESSOR_WAIT_SECONDS",
            "passive_ready_watch_ms": "PASSIVE_READY_WATCH_MS",
            "probe_client_timeout_seconds": "PROBE_CLIENT_TIMEOUT_SECONDS",
            "response_margin_seconds": "LISTENER_RESPONSE_MARGIN_SECONDS",
        }
        missing = [
            ENV_PREFIX + n for n in names.values() if not environ.get(ENV_PREFIX + n)
        ]
        if missing:
            msg = f"shared listener policy values not rendered: {sorted(missing)}"
            raise ValueError(msg)
        return cls.model_validate(
            {field: environ[ENV_PREFIX + name] for field, name in names.items()}
        )


__all__: list[str] = ["ENV_PREFIX", "ModelSharedListenerPolicy"]
