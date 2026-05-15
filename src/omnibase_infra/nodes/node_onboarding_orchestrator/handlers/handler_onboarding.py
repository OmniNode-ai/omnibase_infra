# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handler for the onboarding orchestrator node (OMN-5270, OMN-10784).

Two execution paths:
- **DAG path** (default): load graph -> resolve policy -> execute
  verification for each step -> render output.
- **Interactive path**: load interactive policy by name -> drive
  ``InteractiveExecutor`` with an injected adapter -> optionally
  write env config via ``ConfigWriter``.

The ``input_adapter`` is a function parameter injected by the caller,
NOT part of the Pydantic model (DI outside models — OMN-10784 GPT #1).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from omnibase_infra.nodes.node_onboarding_orchestrator.models.enum_onboarding_status import (
    EnumOnboardingStatus,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_input import (
    ModelOnboardingInput,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_output import (
    ModelOnboardingOutput,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_step_result import (
    ModelStepResult,
)
from omnibase_infra.onboarding.config_writer import ConfigWriter
from omnibase_infra.onboarding.interactive_executor import InteractiveExecutor
from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy
from omnibase_infra.onboarding.model_onboarding_step import ModelOnboardingStep
from omnibase_infra.onboarding.policy_resolver import (
    load_builtin_policies,
    resolve_policy,
)
from omnibase_infra.onboarding.protocol_input_adapter import ProtocolInputAdapter
from omnibase_infra.onboarding.renderers.renderer_markdown import (
    RendererOnboardingMarkdown,
)
from omnibase_infra.probes.model_verification_spec import ModelVerificationSpec
from omnibase_infra.probes.verification_executor import execute_verification


class OnboardingHandlerError(ValueError):
    """Raised when the handler encounters an unrecoverable input error."""


def _load_interactive_policy(policy_name: str) -> ModelInteractivePolicy:
    """Load an interactive policy by name from the built-in policies directory.

    Raises:
        OnboardingHandlerError: If the policy is not found or not interactive.
    """
    policies = load_builtin_policies()
    raw = policies.get(policy_name)
    if raw is None:
        msg = f"Policy '{policy_name}' not found in built-in policies"
        raise OnboardingHandlerError(msg)

    policy_type = raw.get("policy_type")
    if policy_type != "interactive":
        msg = (
            f"Policy '{policy_name}' has policy_type={policy_type!r}, "
            f"expected 'interactive'"
        )
        raise OnboardingHandlerError(msg)

    try:
        return ModelInteractivePolicy.model_validate(raw)
    except ValidationError as exc:
        msg = f"Built-in policy '{policy_name}' is invalid"
        raise OnboardingHandlerError(msg) from exc


async def _handle_interactive(
    input_model: ModelOnboardingInput,
    input_adapter: ProtocolInputAdapter,
) -> ModelOnboardingOutput:
    """Execute the interactive onboarding path.

    Args:
        input_model: Input specifying the policy and dry_run/env_output_path.
        input_adapter: Adapter for collecting user input (injected by caller).

    Returns:
        Output with interactive provenance.
    """
    assert input_model.policy_name is not None  # caller guarantees this

    policy = _load_interactive_policy(input_model.policy_name)
    executor = InteractiveExecutor(policy, input_adapter)
    result = await executor.execute()

    # Convert interactive step results to handler-level step results
    handler_step_results = [
        ModelStepResult(
            step_key=sr.step_key,
            passed=True,
            message="Response captured",
        )
        for sr in result.step_results
    ]

    # Optionally write env config
    env_output_path_written: str | None = None
    if not input_model.dry_run:
        if input_model.env_output_path is None:
            msg = "env_output_path is required when dry_run=False"
            raise OnboardingHandlerError(msg)
        target_path = Path(input_model.env_output_path)
        writer = ConfigWriter()
        writer.write(result.env_dict, target_path)
        env_output_path_written = str(target_path)

    # Render env output as markdown for display
    env_lines = [f"  {k}={v}" for k, v in sorted(result.env_dict.items())]
    rendered = (
        f"# Interactive Onboarding: {input_model.policy_name}\n\n"
        f"Terminal step: {result.terminal_step}\n"
        f"Completed: {result.completed}\n\n"
        f"## Environment Output\n\n```\n" + "\n".join(env_lines) + "\n```\n"
    )
    if input_model.dry_run:
        rendered += "\n*Dry run — no files written.*\n"
    else:
        rendered += f"\n*Written to: {env_output_path_written}*\n"

    visited_steps = [sr.step_key for sr in result.step_results]

    return ModelOnboardingOutput(
        success=result.completed,
        total_steps=len(visited_steps) + 1,  # +1 for terminal step
        completed_steps=len(visited_steps) + 1,
        step_results=handler_step_results,
        rendered_output=rendered,
        provenance=result,
        policy_name=input_model.policy_name,
        policy_type="interactive",
        visited_steps=visited_steps,
        terminal_step=result.terminal_step,
        dry_run=input_model.dry_run,
        env_output_path_written=env_output_path_written,
    )


async def _handle_dag(
    input_model: ModelOnboardingInput,
) -> ModelOnboardingOutput:
    """Execute the existing DAG-based onboarding path (unchanged from OMN-5270)."""
    graph = load_canonical_graph()
    steps = resolve_policy(
        graph,
        target_capabilities=input_model.target_capabilities,
        skip_steps=input_model.skip_steps,
    )

    step_results: list[ModelStepResult] = []
    completed_steps: list[ModelOnboardingStep] = []
    failed = False

    for step in steps:
        if failed and not input_model.continue_on_failure:
            step_results.append(
                ModelStepResult(
                    step_key=step.step_key,
                    passed=False,
                    message="Skipped due to previous failure",
                )
            )
            continue

        if step.verification:
            spec = ModelVerificationSpec(
                check_type=step.verification.check_type,
                target=step.verification.target,
                timeout_seconds=step.verification.timeout_seconds or 10,
            )
            result = await execute_verification(spec)
            step_results.append(
                ModelStepResult(
                    step_key=step.step_key,
                    passed=result.passed,
                    message=result.message,
                    elapsed_ms=result.elapsed_ms,
                )
            )
            if result.passed:
                completed_steps.append(step)
            else:
                failed = True
        else:
            step_results.append(
                ModelStepResult(
                    step_key=step.step_key,
                    passed=True,
                    message="No verification defined",
                )
            )
            completed_steps.append(step)

    # Render output — pass all steps (not just completed) so failed/skipped appear
    renderer = RendererOnboardingMarkdown()
    rendered = renderer.render(steps, step_results, title="Onboarding Progress")

    # Compute capability evidence from step results
    verified_caps: list[str] = []
    unmet_caps: list[str] = []
    for step, result in zip(steps, step_results, strict=True):
        if result.passed:
            verified_caps.extend(step.produces_capabilities)
        else:
            unmet_caps.extend(step.produces_capabilities)

    # Determine status
    all_passed = all(r.passed for r in step_results)
    if all_passed:
        status = EnumOnboardingStatus.PASSED
    elif any(r.message == "Skipped due to previous failure" for r in step_results):
        status = EnumOnboardingStatus.BLOCKED
    else:
        status = EnumOnboardingStatus.FAILED

    return ModelOnboardingOutput(
        success=all_passed,
        total_steps=len(steps),
        completed_steps=len(completed_steps),
        step_results=step_results,
        rendered_output=rendered,
        status=status,
        verified_capabilities=verified_caps,
        unmet_capabilities=unmet_caps,
    )


async def handle_onboarding(
    input_model: ModelOnboardingInput,
    input_adapter: ProtocolInputAdapter | None = None,
) -> ModelOnboardingOutput:
    """Execute the onboarding orchestration.

    Routes to the interactive or DAG path based on ``policy_name``.

    Args:
        input_model: Input with policy name and/or target capabilities.
        input_adapter: Adapter for interactive input collection.
            Required when ``policy_name`` is set and the policy is interactive.
            Injected via function parameter, NOT in the Pydantic model.

    Returns:
        Output with step results and rendered output.

    Raises:
        OnboardingHandlerError: If interactive path is requested but
            ``input_adapter`` is not provided, or if the policy is not found.
    """
    if input_model.policy_name is not None:
        # Interactive path
        if input_adapter is None:
            msg = (
                f"input_adapter is required for interactive policy "
                f"'{input_model.policy_name}'"
            )
            raise OnboardingHandlerError(msg)
        return await _handle_interactive(input_model, input_adapter)

    # DAG path (existing behavior)
    return await _handle_dag(input_model)


class HandlerOnboarding:
    """Class wrapper for handle_onboarding — required for OMN-8735 auto-wiring.

    The auto-wiring framework requires a class (not a bare function) so it can
    inspect the constructor signature. This wrapper delegates to
    ``handle_onboarding`` and requires no constructor arguments.
    """

    def __init__(self) -> None:  # stub-ok: stateless init
        """Initialize the handler (stateless)."""

    async def handle(
        self,
        input_model: ModelOnboardingInput,
        input_adapter: ProtocolInputAdapter | None = None,
    ) -> ModelOnboardingOutput:
        """Execute the onboarding orchestration."""
        return await handle_onboarding(input_model, input_adapter=input_adapter)


__all__ = ["OnboardingHandlerError", "handle_onboarding", "HandlerOnboarding"]
