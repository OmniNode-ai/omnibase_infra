# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The claim -> run -> post result -> ack cycle, plus the attach/heartbeat loop.

Fail-closed rules this module enforces:

- A claimed envelope whose payload does not carry the fields a chat request
  needs is nacked, never guessed into a request.
- A local-model failure publishes a typed failure envelope AND nacks the
  source message -- it never acks a message it failed to complete.
- A heartbeat that reports the session as terminated stops the loop; it is
  never treated as "keep going, retry the heartbeat later."
- The renewal directive's ``mode`` is asserted to be ``RE_ATTACH`` before
  this module acts on it. A control plane that ever starts declaring an
  in-place renewal mode this client doesn't implement must fail loud here,
  not silently skip renewal and run past its session ceiling.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from omnibase_core.types import JsonType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraProtocolError,
    InfraRequestRejectedError,
    InfraUnavailableError,
)
from scripts.edge_delegation_worker.credential_loader import load_worker_credential
from scripts.edge_delegation_worker.delegation_channel import ProtocolDelegationChannel
from scripts.edge_delegation_worker.gateway_control_plane import (
    attach,
    detach,
    heartbeat,
    resolve_access_token,
)
from scripts.edge_delegation_worker.local_model_bridge import run_chat_completion
from scripts.edge_delegation_worker.models import (
    ModelDelegationEnvelope,
    ModelGatewaySession,
    ModelLocalInferenceRequest,
)
from scripts.edge_delegation_worker.topic_constants import (
    DELEGATION_COMPLETED_TOPIC,
    DELEGATION_FAILED_TOPIC,
)

logger = logging.getLogger(__name__)

_RENEWAL_MODE_RE_ATTACH = "RE_ATTACH"


def _envelope_to_inference_request(
    envelope: ModelDelegationEnvelope, *, model_name: str
) -> ModelLocalInferenceRequest:
    """Build a chat-completion request from a claimed envelope's payload.

    Fail-closed: raises ``ValueError`` for any payload shape this worker
    does not recognize rather than fabricating a plausible-looking request.
    Accepts either an already-OpenAI-shaped ``messages`` list, or a bare
    ``prompt`` string that is wrapped into a single user message.
    """
    payload = envelope.payload
    messages_raw = payload.get("messages")
    messages: tuple[dict[str, JsonType], ...]
    if isinstance(messages_raw, list) and messages_raw:
        messages = tuple(m for m in messages_raw if isinstance(m, dict))
        if len(messages) != len(messages_raw):
            raise ValueError("payload.messages contained a non-object entry")
    else:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                "payload has neither a non-empty 'messages' list nor a "
                "non-empty string 'prompt'"
            )
        messages = ({"role": "user", "content": prompt},)

    max_tokens = payload.get("max_tokens")
    temperature = payload.get("temperature")
    return ModelLocalInferenceRequest(
        correlation_id=envelope.correlation_id,
        model=model_name,
        messages=messages,
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        temperature=temperature if isinstance(temperature, (int, float)) else None,
    )


async def run_single_cycle(
    *,
    channel: ProtocolDelegationChannel,
    http_client: httpx.AsyncClient,
    model_base: str,
    model_name: str,
) -> bool:
    """Run one claim/infer/publish/ack cycle. Returns ``True`` iff a message was claimed."""
    envelope = await channel.claim()
    if envelope is None:
        return False

    try:
        request = _envelope_to_inference_request(envelope, model_name=model_name)
    except ValueError as exc:
        logger.warning(
            "edge_delegation_worker: unusable payload shape for correlation_id=%s: %s",
            envelope.correlation_id,
            exc,
        )
        await channel.publish_result(
            topic=DELEGATION_FAILED_TOPIC,
            correlation_id=envelope.correlation_id,
            event_type="omnibase-infra.delegation-failed",
            payload={"reason": f"unusable payload shape: {exc}"},
        )
        await channel.nack(envelope, reason=f"unusable payload shape: {exc}")
        return True

    try:
        result = await run_chat_completion(model_base, request, http_client=http_client)
    except (InfraRequestRejectedError, InfraProtocolError) as exc:
        logger.warning(
            "edge_delegation_worker: local inference failed for correlation_id=%s: %s",
            envelope.correlation_id,
            exc,
        )
        await channel.publish_result(
            topic=DELEGATION_FAILED_TOPIC,
            correlation_id=envelope.correlation_id,
            event_type="omnibase-infra.delegation-failed",
            payload={"reason": str(exc)},
        )
        await channel.nack(envelope, reason=str(exc))
        return True

    await channel.publish_result(
        topic=DELEGATION_COMPLETED_TOPIC,
        correlation_id=envelope.correlation_id,
        event_type="omnibase-infra.delegation-completed",
        payload={
            "content": result.content,
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
    )
    await channel.ack(envelope)
    return True


class SessionTerminatedError(RuntimeError):
    """Raised when a heartbeat reports the session as no longer active."""


class UnsupportedRenewalModeError(RuntimeError):
    """Raised when the control plane declares a renewal mode this client cannot run."""


@dataclass(frozen=True)
class _AttachedSession:
    """A gateway session plus the bearer token it was attached with.

    Bundled together because every re-attach (initial + each renewal) mints
    both at once -- keeping them paired avoids the caller ever holding a
    session_id from one attach next to an access_token from a different one.
    """

    session: ModelGatewaySession
    access_token: str


async def _fresh_attach(
    *,
    api_base: str,
    credential_path: Path,
    edge_instance_id: str,
    http_client: httpx.AsyncClient,
) -> _AttachedSession:
    """Perform a fresh client_credentials grant + attach, minting a NEW session_id.

    Renewal is never in-place -- see the module docstring and OMN-15952: a
    runtime that wants to keep working past ``expires_at`` re-grants and
    attaches again rather than extending a live session on a heartbeat.
    """
    credential = load_worker_credential(credential_path)
    access_token = await resolve_access_token(credential, http_client=http_client)
    session = await attach(
        api_base,
        access_token=access_token,
        edge_instance_id=edge_instance_id,
        http_client=http_client,
    )
    if session.renewal is not None and session.renewal.mode != _RENEWAL_MODE_RE_ATTACH:
        raise UnsupportedRenewalModeError(
            f"gateway declared renewal mode {session.renewal.mode!r}; "
            f"this worker only implements {_RENEWAL_MODE_RE_ATTACH!r}"
        )
    return _AttachedSession(session=session, access_token=access_token)


async def run_worker_loop(
    *,
    api_base: str,
    model_base: str,
    model_name: str,
    credential_path: Path,
    edge_instance_id: str,
    channel: ProtocolDelegationChannel,
    http_client: httpx.AsyncClient,
    poll_interval_seconds: float,
    max_cycles: int | None = None,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    """Attach, then loop claim/infer/publish/ack with heartbeat + renewal.

    ``max_cycles`` is test/bounded-run support (``None`` means run until a
    fatal error or external cancellation). This function does not catch
    ``asyncio.CancelledError`` -- graceful shutdown is the caller's
    responsibility (cancel the task, let ``finally`` detach).

    ``now_fn`` defaults to the real wall clock; tests inject a fake to make
    heartbeat/renewal timing deterministic without a real sleep.
    """
    attached = await _fresh_attach(
        api_base=api_base,
        credential_path=credential_path,
        edge_instance_id=edge_instance_id,
        http_client=http_client,
    )
    session = attached.session
    access_token = attached.access_token
    last_heartbeat = now_fn()
    cycles = 0

    try:
        while max_cycles is None or cycles < max_cycles:
            now = now_fn()

            if session.renewal is not None and now >= session.renewal.renew_not_before:
                attached = await _fresh_attach(
                    api_base=api_base,
                    credential_path=credential_path,
                    edge_instance_id=edge_instance_id,
                    http_client=http_client,
                )
                session = attached.session
                access_token = attached.access_token
                last_heartbeat = now

            elif (
                now - last_heartbeat
            ).total_seconds() >= session.heartbeat_interval_seconds:
                result = await heartbeat(
                    api_base,
                    access_token=access_token,
                    session_id=session.session_id,
                    http_client=http_client,
                )
                if result.is_terminated:
                    raise SessionTerminatedError(
                        f"session {session.session_id} terminated: "
                        f"{result.termination_reason}"
                    )
                last_heartbeat = now

            processed = await run_single_cycle(
                channel=channel,
                http_client=http_client,
                model_base=model_base,
                model_name=model_name,
            )
            cycles += 1
            if not processed:
                await asyncio.sleep(poll_interval_seconds)
    finally:
        try:
            await detach(
                api_base,
                access_token=access_token,
                session_id=session.session_id,
                reason="worker shutdown",
                http_client=http_client,
            )
        except (
            InfraAuthenticationError,
            InfraUnavailableError,
        ):  # cleanup-resilience-ok: best-effort teardown on any exit path
            logger.warning(
                "edge_delegation_worker: detach on shutdown failed for session_id=%s",
                session.session_id,
                exc_info=True,
            )
