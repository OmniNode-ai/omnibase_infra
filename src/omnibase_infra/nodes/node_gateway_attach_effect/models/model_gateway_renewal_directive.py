# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The server-computed renewal cycle handed to the client at attach.

OMN-15952. Before this model, ``gateway.attach`` told a runtime how often to
heartbeat and nothing else. The runtime could read ``expires_at`` off the
session it got back, but every other term of the renewal cycle -- how early
to start, how much to spread, and above all *what renewal even is* -- was
undeclared. An unattended runtime cannot infer a policy it was never told,
and each independent client that guessed would guess differently.

This directive makes the cycle a contract term instead of client folklore:

  * ``mode`` names the mechanism (``RE_ATTACH``: re-grant, then attach again
    for a NEW ``session_id``). It is on the wire so no client has to assume
    that a heartbeat extends anything -- it does not.
  * ``renew_at`` is the deadline. Re-grant and re-attach must have COMPLETED
    by then, not started.
  * ``renew_not_before`` opens the jitter window. A fleet bootstrapped in one
    batch must not converge on a single wall-clock second against Keycloak's
    token endpoint, so each runtime picks its own moment uniformly in
    ``[renew_not_before, renew_at]``.
  * ``session_expires_at`` is echoed here rather than only on the session so
    the directive is self-contained -- a client that logs, forwards, or
    persists only the directive still holds the ceiling it is racing.

The ordering invariant below (``renew_not_before <= renew_at <
session_expires_at``) is the machine-checkable form of "renewal completes
before expiry, and expiry never moves." It is enforced in the model rather
than in the builder so that any construction path -- handler, test, a future
second caller, or a payload deserialized off the wire -- is subject to it.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_renewal_mode import (
    EnumGatewayRenewalMode,
)


class ModelGatewayRenewalDirective(BaseModel):
    """Server-declared renewal cycle for one attached session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: EnumGatewayRenewalMode
    # Echo of the session ceiling this cycle is racing. Never moved by any
    # subsequent call on this session.
    session_expires_at: datetime
    # Earliest moment the runtime should begin its re-grant + re-attach.
    renew_not_before: datetime
    # Latest moment by which re-grant + re-attach must have COMPLETED.
    renew_at: datetime
    # The terms that produced the two timestamps above, echoed so a client
    # can recompute the cycle after a clock correction without a second
    # round trip, and so drift between the config and the wire is visible.
    margin_seconds: int = Field(gt=0)
    jitter_seconds: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> ModelGatewayRenewalDirective:
        """Fail closed on any directive that would tell a client to renew late.

        A directive whose ``renew_at`` is at or past ``session_expires_at``
        is worse than no directive: it instructs the runtime to attempt a
        re-attach with a session already dead, and reads as a deliberate
        policy rather than a bug. Constructing one raises here instead.
        """
        if self.renew_not_before > self.renew_at:
            raise ValueError(
                "renew_not_before must not be later than renew_at "
                f"({self.renew_not_before.isoformat()} > {self.renew_at.isoformat()})"
            )
        if self.renew_at >= self.session_expires_at:
            raise ValueError(
                "renew_at must be strictly before session_expires_at -- renewal "
                "completes before expiry and never extends it "
                f"({self.renew_at.isoformat()} >= "
                f"{self.session_expires_at.isoformat()})"
            )
        return self


__all__ = ["ModelGatewayRenewalDirective"]
