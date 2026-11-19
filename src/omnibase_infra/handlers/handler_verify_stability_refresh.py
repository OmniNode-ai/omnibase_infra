# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability refresh health gate's Kafka API reads, authenticated (OMN-25927).

Once the stability broker requires SASL (OMN-23363), an anonymous ``rpk`` call inside
the broker container is refused, so a refresh whose health gate reads anonymously
fails and rolls back with SASL left on. The gate reads as the declared superuser
principal ``stability-refresh-gate``. Its password is the operator env's
``STABILITY_TEST_KAFKA_SASL_PASSWORD_STABILITY_REFRESH_GATE``, which the refresh
exports (``set -a`` over the operator env file); docker copies it into the container
from ``RPK_PASS``, so only the variable NAME reaches the command line. With no
password set the call stays anonymous, as on a PLAINTEXT broker.

``scripts/runtime_build/verify_stability_refresh.py`` runs from a checkout under any
interpreter, so it loads this module by path and builds every rpk argv with
:func:`rpk_exec_argv`; this module imports nothing outside the standard library at
load time. The node path (``handle``) runs one read and returns its exit code and
output.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
    from omnibase_infra.models.model_verify_stability_refresh_rpk_request import (
        ModelVerifyStabilityRefreshRpkRequest,
    )
    from omnibase_infra.models.model_verify_stability_refresh_rpk_result import (
        ModelVerifyStabilityRefreshRpkResult,
    )

GATE_PRINCIPAL = "stability-refresh-gate"
GATE_PASSWORD_VAR = "STABILITY_TEST_KAFKA_SASL_PASSWORD_STABILITY_REFRESH_GATE"
SASL_MECHANISM = "SCRAM-SHA-256"


def rpk_exec_argv(
    broker_container: str,
    *args: str,
    environ: MutableMapping[str, str] | None = None,
) -> list[str]:
    """``docker exec`` argv for one rpk call inside the broker container.

    When the gate's password is set, it is copied to ``RPK_PASS`` in ``environ``
    (the process environment by default), which docker reads by name.
    """
    env = os.environ if environ is None else environ
    password = env.get(GATE_PASSWORD_VAR, "")
    login: list[str] = []
    if password:
        env["RPK_PASS"] = password
        login = [
            "-e",
            f"RPK_USER={GATE_PRINCIPAL}",
            "-e",
            f"RPK_SASL_MECHANISM={SASL_MECHANISM}",
            "-e",
            "RPK_PASS",
        ]
    return ["docker", "exec", *login, broker_container, "rpk", *args]


class HandlerVerifyStabilityRefresh:
    """The node's single entry point: one authenticated rpk read of the stability broker."""

    @property
    def handler_type(self) -> EnumHandlerType:
        from omnibase_infra.enums import EnumHandlerType

        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        from omnibase_infra.enums import EnumHandlerTypeCategory

        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelVerifyStabilityRefreshRpkRequest
    ) -> ModelVerifyStabilityRefreshRpkResult:
        """Run one rpk read inside the broker container as the gate's principal."""
        from omnibase_infra.models.model_verify_stability_refresh_rpk_result import (
            ModelVerifyStabilityRefreshRpkResult,
        )

        argv = rpk_exec_argv(request.broker_container, *request.rpk_args)
        authenticated = "RPK_PASS" in argv
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ModelVerifyStabilityRefreshRpkResult(
                exit_code=124, authenticated=authenticated, detail="rpk read timed out"
            )
        return ModelVerifyStabilityRefreshRpkResult(
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            detail=(completed.stderr or "").strip()[-400:],
            authenticated=authenticated,
        )
