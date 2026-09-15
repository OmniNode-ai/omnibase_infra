# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An Infisical read failure must name what caused it (OMN-17372).

WHY THIS EXISTS. ``SecretResolutionError`` hardcodes
``EnumCoreErrorCode.RESOURCE_NOT_FOUND`` for every cause it wraps
(``omnibase_infra/errors/error_infra.py``), and both Infisical read paths used
to raise it with the fixed string ``"Failed to resolve secret from Infisical"``,
discarding the chained exception. So a rate limit, a 5xx, a malformed response
and a genuinely absent secret all reached the caller as the same sentence and
the same code.

MEASURED. The scheduled customer-pass walk failed a real customer-key OpenRouter
delegation with exactly that string on three consecutive runs -- 2026-09-14
18:35Z, 2026-09-15 01:15Z, 2026-09-15 06:47Z -- against successes on either side
of them. The durable ``gateway_workflows`` record carried an empty
``terminal_failure_class``, an empty ``terminal_failure_code`` and that one
sentence as ``terminal_failure_reason``; the worker pods holding the logs had
been replaced before anyone looked. The condition could not be named at all.

WHAT IS PINNED. The cause's CLASS NAME reaches the message, and the message is
the only part that survives to a delegation terminal. The cause's own TEXT does
not: a class name cannot carry a secret, an adapter or SDK message can.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import SecretResolutionError
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

pytestmark = pytest.mark.unit


class _ProviderRateLimitedError(RuntimeError):
    """Stands in for any non-auth, non-timeout adapter failure."""


#: Text the cause carries that must NOT be copied into the raised message. A
#: real adapter message can hold a folder, a key name or a value; this one holds
#: all three shapes at once so a naive ``str(e)`` interpolation is visible.
_CAUSE_TEXT = (
    "429 for /tenant-inference-credentials/cred_abc_openrouter_def = sk-live-XXXX"
)


class _ThrowingHandlerSync:
    """An Infisical handler whose SYNC read raises."""

    def get_secret_sync(self, *, secret_name: str, secret_path: str | None) -> Any:
        raise _ProviderRateLimitedError(_CAUSE_TEXT)


class _ThrowingHandlerAsync:
    """An Infisical handler whose ASYNC ``execute`` raises."""

    async def execute(self, envelope: dict[str, object]) -> Any:
        raise _ProviderRateLimitedError(_CAUSE_TEXT)


def _additional_context(error: SecretResolutionError) -> dict[str, Any]:
    """The bundled infra context, as it survives onto ``ModelOnexError``.

    ``ModelInfraErrorContext`` is flattened into ``context["additional_context"]``
    rather than kept as a model, so a test that reads attributes off
    ``error.context`` is asserting against a dict and passes vacuously.
    """
    context = error.context
    assert isinstance(context, dict)
    additional = context.get("additional_context")
    assert isinstance(additional, dict)
    return additional


def _resolver(handler: object) -> SecretResolver:
    return SecretResolver(
        config=ModelSecretResolverConfig(mappings=[]),
        infisical_handler=handler,  # type: ignore[arg-type]
    )


class TestInfisicalReadFailureNamesItsCause:
    """Both read paths must carry the cause type out of the except block."""

    def test_sync_read_failure_names_the_cause_type(self) -> None:
        resolver = _resolver(_ThrowingHandlerSync())

        with pytest.raises(SecretResolutionError) as excinfo:
            resolver._read_infisical_secret_sync("SOME_KEY", "some.logical.name")

        message = str(excinfo.value)
        assert "Failed to resolve secret from Infisical" in message
        assert "_ProviderRateLimitedError" in message, (
            "the message is the only part of this failure a delegation terminal "
            "carries, so a cause recorded nowhere in it cannot be diagnosed later"
        )

    def test_async_read_failure_names_the_cause_type(self) -> None:
        resolver = _resolver(_ThrowingHandlerAsync())

        with pytest.raises(SecretResolutionError) as excinfo:
            asyncio.run(
                resolver._read_infisical_secret_async("SOME_KEY", "some.logical.name")
            )

        message = str(excinfo.value)
        assert "Failed to resolve secret from Infisical" in message
        assert "_ProviderRateLimitedError" in message

    def test_sync_read_failure_records_original_error_type_in_context(self) -> None:
        resolver = _resolver(_ThrowingHandlerSync())

        with pytest.raises(SecretResolutionError) as excinfo:
            resolver._read_infisical_secret_sync("SOME_KEY", "some.logical.name")

        context = _additional_context(excinfo.value)
        assert context["original_error_type"] == "_ProviderRateLimitedError"
        assert context["transport_type"] == EnumInfraTransportType.INFISICAL.value

    def test_async_read_failure_records_original_error_type_in_context(self) -> None:
        resolver = _resolver(_ThrowingHandlerAsync())

        with pytest.raises(SecretResolutionError) as excinfo:
            asyncio.run(
                resolver._read_infisical_secret_async("SOME_KEY", "some.logical.name")
            )

        context = _additional_context(excinfo.value)
        assert context["original_error_type"] == "_ProviderRateLimitedError"


class TestInfisicalReadFailureLeaksNothing:
    """The TYPE is carried out. The cause's own text is not."""

    @pytest.mark.parametrize(
        ("handler", "call"),
        [
            (_ThrowingHandlerSync(), "sync"),
            (_ThrowingHandlerAsync(), "async"),
        ],
    )
    def test_cause_text_is_never_copied_into_the_message(
        self, handler: object, call: str
    ) -> None:
        resolver = _resolver(handler)

        with pytest.raises(SecretResolutionError) as excinfo:
            if call == "sync":
                resolver._read_infisical_secret_sync("SOME_KEY", "some.logical.name")
            else:
                asyncio.run(
                    resolver._read_infisical_secret_async(
                        "SOME_KEY", "some.logical.name"
                    )
                )

        message = str(excinfo.value)
        assert _CAUSE_TEXT not in message
        assert "sk-live-XXXX" not in message
        assert "cred_abc_openrouter_def" not in message
        assert "/tenant-inference-credentials" not in message

    def test_the_chained_cause_is_still_available_for_a_local_traceback(self) -> None:
        """Preserved on ``__cause__``, where a pod log can print it and a
        customer-facing record cannot."""
        resolver = _resolver(_ThrowingHandlerSync())

        with pytest.raises(SecretResolutionError) as excinfo:
            resolver._read_infisical_secret_sync("SOME_KEY", "some.logical.name")

        assert isinstance(excinfo.value.__cause__, _ProviderRateLimitedError)
