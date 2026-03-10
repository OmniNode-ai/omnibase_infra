# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that bootstraps Infisical via provision + seed scripts.

Runs ``provision-infisical.py`` (unless ``skip_identity=True``) and
``seed-infisical.py`` (with ``--dry-run`` when ``dry_run=True``) in order.

Invariant I3 — Monkeypatch discipline:
    - ``import asyncio`` at module top.
    - Subprocess calls go through ``asyncio.create_subprocess_exec``.
    - Tests patch via ``monkeypatch.setattr(handler_mod, "asyncio", ...)``.

Ticket: OMN-3494
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.nodes.node_setup_infisical_effect.models.model_infisical_setup_effect_output import (
    ModelInfisicalSetupEffectOutput,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

# Resolved relative to the package install root (omnibase_infra/scripts/).
_SCRIPTS_DIR = Path(__file__).resolve().parents[7] / "scripts"


def _resolve_scripts_dir() -> Path:
    """Return the path to the scripts directory."""
    return _SCRIPTS_DIR


async def _run_script(
    script_path: Path,
    extra_args: list[str],
    env_override: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Run a Python script via the current interpreter.

    Returns:
        Tuple of (success: bool, output: str).
    """
    cmd = [sys.executable, str(script_path), *extra_args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout_bytes, _ = await proc.communicate()
    stdout_text = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    success = proc.returncode == 0
    return success, stdout_text


class HandlerInfisicalFullSetup:
    """Run Infisical provision + seed scripts in sequence.

    Attributes:
        handler_type: ``NODE_HANDLER``
        handler_category: ``EFFECT``
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the handler."""
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler."""
        self._initialized = True
        logger.info("HandlerInfisicalFullSetup initialized")

    async def shutdown(self) -> None:
        """Shut down the handler."""
        self._initialized = False
        logger.info("HandlerInfisicalFullSetup shutdown")

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[ModelInfisicalSetupEffectOutput]:
        """Execute Infisical provision and seed in order.

        Envelope keys:
            correlation_id: UUID for tracing.
            skip_identity: bool — if True, skip provision-infisical.py.
            dry_run: bool — if True, pass ``--dry-run`` to seed-infisical.py.
            infisical_addr: str | None — override Infisical address.
            scripts_dir: str | Path | None — override scripts directory (for tests).
        """
        correlation_id_raw = envelope.get("correlation_id")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id_raw
            if isinstance(correlation_id_raw, UUID)
            else None,
            transport_type=EnumInfraTransportType.INFISICAL,
            operation="run_full_setup",
            target_name="infisical_scripts",
        )
        corr_id = context.correlation_id
        if corr_id is None:
            corr_id = uuid4()

        skip_identity = bool(envelope.get("skip_identity", False))
        dry_run = bool(envelope.get("dry_run", False))
        infisical_addr = envelope.get("infisical_addr")
        scripts_dir_raw = envelope.get("scripts_dir")
        scripts_dir = (
            Path(str(scripts_dir_raw))
            if scripts_dir_raw is not None
            else _resolve_scripts_dir()
        )

        provision_script = scripts_dir / "provision-infisical.py"
        seed_script = scripts_dir / "seed-infisical.py"

        identity_provisioned = False
        secrets_seeded = False
        errors: list[str] = []

        # Step 1: provision-infisical.py (unless skip_identity=True)
        if not skip_identity:
            provision_args: list[str] = []
            if infisical_addr and isinstance(infisical_addr, str):
                provision_args += ["--addr", infisical_addr]
            logger.info("Running provision-infisical.py: %s", provision_script)
            success, output = await _run_script(provision_script, provision_args)
            if success:
                identity_provisioned = True
                logger.info("provision-infisical.py succeeded")
            else:
                errors.append(f"provision-infisical.py failed: {output.strip()[:500]}")
                logger.error("provision-infisical.py failed: %s", output[:200])
                return ModelHandlerOutput.for_compute(
                    input_envelope_id=uuid4(),
                    correlation_id=corr_id,
                    handler_id="handler-infisical-full-setup",
                    result=ModelInfisicalSetupEffectOutput(
                        success=False,
                        correlation_id=corr_id,
                        status="failed",
                        infisical_addr=infisical_addr
                        if isinstance(infisical_addr, str)
                        else None,
                        error="; ".join(errors),
                    ),
                )
        else:
            logger.info("Skipping provision-infisical.py (skip_identity=True)")
            identity_provisioned = False

        # Step 2: seed-infisical.py
        seed_args: list[str] = []
        if dry_run:
            seed_args.append("--dry-run")
        else:
            seed_args.append("--execute")
        logger.info("Running seed-infisical.py: %s", seed_script)
        success, output = await _run_script(seed_script, seed_args)
        if success:
            secrets_seeded = True
            logger.info("seed-infisical.py succeeded")
        else:
            errors.append(f"seed-infisical.py failed: {output.strip()[:500]}")
            logger.error("seed-infisical.py failed: %s", output[:200])
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-infisical-full-setup",
                result=ModelInfisicalSetupEffectOutput(
                    success=False,
                    correlation_id=corr_id,
                    status="failed",
                    infisical_addr=infisical_addr
                    if isinstance(infisical_addr, str)
                    else None,
                    error="; ".join(errors),
                ),
            )

        logger.info(
            "HandlerInfisicalFullSetup complete: identity_provisioned=%s, secrets_seeded=%s",
            identity_provisioned,
            secrets_seeded,
        )

        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=corr_id,
            handler_id="handler-infisical-full-setup",
            result=ModelInfisicalSetupEffectOutput(
                success=True,
                correlation_id=corr_id,
                status="completed",
                infisical_addr=infisical_addr
                if isinstance(infisical_addr, str)
                else None,
                error=None,
            ),
        )


__all__: list[str] = ["HandlerInfisicalFullSetup"]
