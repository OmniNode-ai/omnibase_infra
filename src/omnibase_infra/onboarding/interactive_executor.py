# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Interactive executor — drives the full onboarding flow via adapter + reducer.

Pure except for adapter I/O (input collection). Never writes files.
Produces a ``ModelInteractiveResult`` with env dict, step results, and provenance.

OMN-10782 / Task 5 of the interactive-onboarding-executor plan.
"""

from __future__ import annotations

from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy
from omnibase_infra.onboarding.model_interactive_result import (
    ModelInteractiveResult,
    ModelStepResult,
)
from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep
from omnibase_infra.onboarding.protocol_input_adapter import ProtocolInputAdapter
from omnibase_infra.onboarding.transition_reducer import (
    TransitionError,
    TransitionReducer,
)


class InteractiveExecutorError(Exception):
    """Raised when the executor encounters an unrecoverable error."""


class InteractiveExecutor:
    """Drives an interactive onboarding policy to completion.

    Composes the ``TransitionReducer`` (pure state machine) with a
    ``ProtocolInputAdapter`` (I/O for input collection).  The executor
    is pure except for adapter calls — it never writes files.

    Args:
        policy: The interactive onboarding policy to execute.
        adapter: Input adapter for collecting user responses.

    Raises:
        InteractiveExecutorError: If the policy graph is invalid
            (wraps ``TransitionError`` from the reducer).
    """

    def __init__(
        self,
        policy: ModelInteractivePolicy,
        adapter: ProtocolInputAdapter,
    ) -> None:
        self._policy = policy
        self._adapter = adapter
        try:
            self._reducer = TransitionReducer(policy)
        except TransitionError as exc:
            raise InteractiveExecutorError(str(exc)) from exc
        self._steps_by_id: dict[str, ModelInteractiveStep] = {
            s.id: s for s in policy.steps
        }

    async def execute(self) -> ModelInteractiveResult:
        """Run the interactive flow from start to terminal step.

        Returns:
            ``ModelInteractiveResult`` containing the env dict, ordered step
            results, and provenance metadata.  Does NOT write any files.

        Raises:
            InteractiveExecutorError: If the start step is missing or
                an unknown step is encountered during traversal.
        """
        start_step = self._policy.start_step
        if start_step is None:
            msg = "Policy has no start_step"
            raise InteractiveExecutorError(msg)

        current_step_id = start_step
        state: dict[str, object] = {}
        step_results: list[ModelStepResult] = []

        while True:
            step = self._steps_by_id.get(current_step_id)
            if step is None:
                msg = f"Unknown step: '{current_step_id}'"
                raise InteractiveExecutorError(msg)

            # Terminal step: notify (action display) and produce env output.
            if self._reducer.is_terminal(current_step_id):
                if step.type == "action":
                    await self._adapter.notify_action(step)
                env_dict = self._reducer.get_env_output(current_step_id, state)
                return ModelInteractiveResult(
                    env_dict=env_dict,
                    step_results=step_results,
                    policy_name=self._policy.policy_name,
                    completed=True,
                    terminal_step=current_step_id,
                )

            # Non-terminal step: collect input via adapter.
            response = await self._collect_response(step)

            # Record step result (action steps have no user response).
            step_results.append(
                ModelStepResult(
                    step_key=step.id,
                    step_title=step.prompt,
                    response=response,
                )
            )

            # Advance through the reducer.
            current_step_id, state = self._reducer.advance(
                current_step_id, response, state
            )

    async def _collect_response(self, step: ModelInteractiveStep) -> str | list[str]:
        """Dispatch to the appropriate adapter method based on step type.

        Returns:
            The user's response: ``str`` for choice/text, ``list[str]`` for
            multi_choice.

        Raises:
            InteractiveExecutorError: If the step type is unsupported.
        """
        if step.type == "choice":
            return await self._adapter.collect_choice(step)
        if step.type == "multi_choice":
            return await self._adapter.collect_multi_choice(step)
        if step.type == "text":
            return await self._adapter.collect_text(step)
        if step.type == "action":
            await self._adapter.notify_action(step)
            # Action steps are notification-only — return empty string as
            # a no-op response for the reducer.
            return ""

        msg = f"Unsupported step type: '{step.type}'"
        raise InteractiveExecutorError(msg)


__all__ = ["InteractiveExecutor", "InteractiveExecutorError"]
