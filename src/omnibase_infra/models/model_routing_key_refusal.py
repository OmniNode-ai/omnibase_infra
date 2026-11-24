# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One instance the router refuses for its agent key (OMN-26485)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelRoutingKeyRefusal(BaseModel):
    """An available instance whose agent key check is not the lane runtime's.

    The key check is the 16-hex one-way value of a signing key, never the key
    (``deploy_agent.auth.hmac_key_check``); both values are named here and neither is
    secret. ``reason`` is ``key-check-differs`` when the instance answered another
    value and ``key-check-unread`` when no answer is on hand (an agent with no key
    commits to none, so an unread value is refused, never assumed equal).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance: str
    instance_key_check: str
    runtime_key_check: str
    reason: Literal["key-check-differs", "key-check-unread"]

    @property
    def message(self) -> str:
        shown = self.instance_key_check or "<unread>"
        return (
            f"instance {self.instance} agent key check {shown} is not the lane "
            f"runtime's {self.runtime_key_check} ({self.reason})"
        )
