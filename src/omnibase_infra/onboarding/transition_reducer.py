# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Transition reducer — pure state machine for interactive onboarding.

Given ``(current_step, response, state)`` returns ``(next_step, updated_state)``
or identifies the step as terminal.  No I/O — the reducer evaluates
the transition table and condition expressions declaratively.

OMN-10780 / Task 3 of the interactive-onboarding-executor plan.
"""

from __future__ import annotations

import re

from omnibase_infra.onboarding.condition_evaluator import (
    ConditionEvaluationError,
    evaluate_condition,
)
from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy
from omnibase_infra.onboarding.model_transition import ModelTransition
from omnibase_infra.onboarding.model_transition_branch import ModelTransitionBranch

# Type aliases to avoid repeating union types in every method signature.
type StepResponse = str | list[str]
type StateDict = dict[str, object]


class TransitionError(Exception):
    """Raised when no transition branch matches or the step is unknown."""


class InterpolationError(Exception):
    """Raised when env output interpolation fails."""


# Regex for {state.key}, {state.key|}, {state.key|default}
_INTERPOLATION_RE = re.compile(r"\{state\.(\w+)(?:\|([^}]*))?\}")


class TransitionReducer:
    """Pure state machine that advances through interactive onboarding steps.

    Validates graph integrity at construction.  All methods are pure — no I/O.
    """

    def __init__(self, policy: ModelInteractivePolicy) -> None:
        self._policy = policy
        self._step_ids = {s.id for s in policy.steps}
        self._transitions_by_step: dict[str, ModelTransition] = {}
        self._terminal_steps: set[str] = set()

        for t in policy.transitions:
            self._transitions_by_step[t.from_step] = t
            if t.terminal:
                self._terminal_steps.add(t.from_step)

        # Collect all option values from steps so they can be injected as
        # self-referential literal tokens during condition evaluation.
        # E.g. step options ["kafka", "llm_inference"] → state tokens
        # {"kafka": "kafka", "llm_inference": "llm_inference"} so that
        # conditions like ``llm_inference in response`` resolve correctly.
        self._option_literals: dict[str, str] = {}
        for step in policy.steps:
            for opt in step.options:
                self._option_literals[opt] = opt

        self._validate_graph()

    # ------------------------------------------------------------------
    # Graph validation (GPT #4 — construction-time)
    # ------------------------------------------------------------------

    def _validate_graph(self) -> None:
        """Validate the transition graph is well-formed.

        Checks:
        - Every transition ``from_step`` references a declared step.
        - No duplicate ``from_step`` entries.
        - Every branch ``next`` references a declared step.
        - Every terminal step has a matching ``env_output`` entry.
        - All steps are reachable from the start step.
        """
        # Structural checks before reachability.
        seen_from_steps: set[str] = set()
        for transition in self._policy.transitions:
            if transition.from_step not in self._step_ids:
                msg = f"Transition from unknown step '{transition.from_step}'"
                raise TransitionError(msg)
            if transition.from_step in seen_from_steps:
                msg = (
                    f"Duplicate transition definition for step '{transition.from_step}'"
                )
                raise TransitionError(msg)
            seen_from_steps.add(transition.from_step)

            targets: list[str] = []
            if transition.responses:
                targets.extend(branch.next for branch in transition.responses.values())
            if transition.on_submit:
                targets.extend(branch.next for branch in transition.on_submit)

            for target in targets:
                if target not in self._step_ids:
                    msg = (
                        f"Transition from '{transition.from_step}' targets "
                        f"unknown step '{target}'"
                    )
                    raise TransitionError(msg)

            if (
                transition.terminal
                and transition.from_step not in self._policy.env_output
            ):
                msg = f"Terminal step '{transition.from_step}' missing env_output"
                raise TransitionError(msg)

        # Reachability: BFS from start_step through all branch targets.
        start = self._policy.start_step
        if start is None:
            msg = "Policy has no start_step"
            raise TransitionError(msg)

        if start not in self._step_ids:
            msg = f"start_step '{start}' is not a declared step"
            raise TransitionError(msg)

        reachable: set[str] = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            if current in reachable:
                continue
            reachable.add(current)
            transition = self._transitions_by_step.get(current)
            if transition is None:
                continue
            if transition.terminal:
                continue
            for branch in (transition.responses or {}).values():
                if branch.next not in reachable:
                    frontier.append(branch.next)
            for branch in transition.on_submit or []:
                if branch.next not in reachable:
                    frontier.append(branch.next)

        # Check for unreachable non-start steps
        unreachable = self._step_ids - reachable
        if unreachable:
            msg = f"Unreachable steps detected: {sorted(unreachable)}"
            raise TransitionError(msg)

    # ------------------------------------------------------------------
    # Core advance
    # ------------------------------------------------------------------

    def advance(
        self, step_id: str, response: StepResponse, state: StateDict
    ) -> tuple[str, StateDict]:
        """Evaluate transitions from *step_id* given *response* + *state*.

        Returns ``(next_step_id, updated_state)``.

        Raises:
            TransitionError: If step_id is unknown, terminal, or no branch matches.
        """
        if step_id not in self._step_ids:
            msg = f"Unknown step: '{step_id}'"
            raise TransitionError(msg)

        transition = self._transitions_by_step.get(step_id)
        if transition is None:
            msg = f"No transition defined for step '{step_id}'"
            raise TransitionError(msg)

        if transition.terminal:
            msg = f"Step '{step_id}' is terminal — cannot advance"
            raise TransitionError(msg)

        branch = self._match_branch(transition, response, state)
        updated_state = self._apply_set_state(branch, response, state)
        return branch.next, updated_state

    def is_terminal(self, step_id: str) -> bool:
        """Return True if *step_id* is a terminal step."""
        return step_id in self._terminal_steps

    # ------------------------------------------------------------------
    # Branch matching
    # ------------------------------------------------------------------

    def _match_branch(
        self,
        transition: ModelTransition,
        response: StepResponse,
        state: StateDict,
    ) -> ModelTransitionBranch:
        """Find the first matching branch for the given response and state.

        For ``responses`` dict (choice steps): look up by response string key.
        For ``on_submit`` list (multi-choice / text): evaluate conditions in order.
        """
        # Choice-style: response is a key into the responses dict
        if transition.responses is not None:
            if isinstance(response, list):
                msg = (
                    f"Step '{transition.from_step}' uses responses dict "
                    f"but received list response"
                )
                raise TransitionError(msg)
            branch = transition.responses.get(response)
            if branch is None:
                msg = (
                    f"No transition from '{transition.from_step}' "
                    f"for response '{response}'"
                )
                raise TransitionError(msg)
            return branch

        # on_submit-style: evaluate conditions against state + response
        if transition.on_submit is not None:
            # Inject 'response' into a temporary state for condition evaluation
            eval_state = self._build_eval_state(state, response)
            for branch in transition.on_submit:
                try:
                    if evaluate_condition(branch.condition, eval_state):
                        return branch
                except ConditionEvaluationError as exc:
                    msg = (
                        f"Condition evaluation failed for step "
                        f"'{transition.from_step}': {exc}"
                    )
                    raise TransitionError(msg) from exc

            msg = (
                f"No matching branch for step '{transition.from_step}' — "
                f"all conditions evaluated to False"
            )
            raise TransitionError(msg)

        msg = (
            f"Transition from '{transition.from_step}' has neither "
            f"responses nor on_submit"
        )
        raise TransitionError(msg)

    def _build_eval_state(self, state: StateDict, response: StepResponse) -> StateDict:
        """Build the state dict used for condition evaluation.

        Injects:
        - ``response`` so conditions like ``llm_inference in response`` work.
        - Self-referential literal tokens for step option values so that
          bare identifiers (e.g. ``llm_inference``) resolve to their own
          string name when used as the LHS of ``in`` / ``not in`` checks.
          Actual state values take precedence over option literals.
        """
        eval_state: StateDict = dict(self._option_literals)
        eval_state.update(state)
        eval_state["response"] = response
        return eval_state

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_set_state(
        branch: ModelTransitionBranch,
        response: StepResponse,
        state: StateDict,
    ) -> StateDict:
        """Apply *branch.set_state* to produce an updated state dict.

        ``{response}`` placeholders in values are replaced with the actual
        response.  Multi-choice responses are stored as lists.
        """
        updated = dict(state)
        for key, value in branch.set_state.items():
            if isinstance(value, str) and value == "{response}":
                # Store the raw response — list for multi-choice, str for single
                updated[key] = response
            else:
                updated[key] = value
        return updated

    # ------------------------------------------------------------------
    # Env output interpolation
    # ------------------------------------------------------------------

    def get_env_output(self, terminal_step_id: str, state: StateDict) -> dict[str, str]:
        """Collect env output for a terminal step with state interpolation.

        Supports:
        - ``{state.key}`` — required; raises if key is not in state.
        - ``{state.key|}`` — optional with empty-string default.
        - ``{state.key|default_value}`` — optional with explicit default.

        Returns:
            Dict mapping env var names to interpolated string values.

        Raises:
            InterpolationError: If a required key is missing from state.
            TransitionError: If terminal_step_id has no env_output entry.
        """
        if terminal_step_id not in self._policy.env_output:
            msg = f"No env_output defined for step '{terminal_step_id}'"
            raise TransitionError(msg)

        template = self._policy.env_output[terminal_step_id]
        result: dict[str, str] = {}

        for env_key, env_template in template.items():
            result[env_key] = self._interpolate(env_template, state)

        return result

    @staticmethod
    def _interpolate(template: str, state: StateDict) -> str:
        """Interpolate ``{state.key}`` / ``{state.key|}`` / ``{state.key|default}``."""

        def _replacer(match: re.Match[str]) -> str:
            key = match.group(1)
            default_group = match.group(2)  # None if no pipe, "" if pipe with no value

            if default_group is not None:
                # Has a pipe — optional key with default
                value = state.get(key)
                if value is None:
                    return default_group
                return str(value)
            else:
                # No pipe — required key
                if key not in state:
                    msg = f"Required state key '{key}' not found during interpolation"
                    raise InterpolationError(msg)
                return str(state[key])

        # Check for unknown interpolation syntax (e.g. {bad.syntax} without state. prefix)
        # We allow literal strings to pass through unchanged.
        return _INTERPOLATION_RE.sub(_replacer, template)


__all__ = [
    "InterpolationError",
    "TransitionError",
    "TransitionReducer",
]
