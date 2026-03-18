# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handler for the onboarding orchestrator node (OMN-5270).

Orchestrates: load graph -> resolve policy -> execute verification
for each step -> render output.
"""

from __future__ import annotations

from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_input import (
    ModelOnboardingInput,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_output import (
    ModelOnboardingOutput,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_step_result import (
    ModelStepResult,
)
from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.model_onboarding_step import ModelOnboardingStep
from omnibase_infra.onboarding.policy_resolver import resolve_policy
from omnibase_infra.onboarding.renderers.renderer_markdown import (
    RendererOnboardingMarkdown,
)
from omnibase_infra.probes.model_verification_spec import ModelVerificationSpec
from omnibase_infra.probes.verification_executor import execute_verification


async def handle_onboarding(
    input_model: ModelOnboardingInput,
) -> ModelOnboardingOutput:
    """Execute the onboarding orchestration.

    Args:
        input_model: Input with policy name and target capabilities.

    Returns:
        Output with step results and rendered output.
    """
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

    # Render output
    renderer = RendererOnboardingMarkdown()
    rendered = renderer.render(completed_steps, title="Onboarding Progress")

    all_passed = all(r.passed for r in step_results)
    return ModelOnboardingOutput(
        success=all_passed,
        total_steps=len(steps),
        completed_steps=len(completed_steps),
        step_results=step_results,
        rendered_output=rendered,
    )


__all__ = ["handle_onboarding"]
