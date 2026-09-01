# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handler for the onboarding orchestrator node (OMN-5270, OMN-10784).

Two execution paths:
- **DAG path** (default): load graph -> resolve policy -> execute
  verification for each step -> render output.
- **Interactive path**: load interactive policy by name -> drive
  ``InteractiveExecutor`` with an injected adapter -> optionally
  write env config via ``ConfigWriter`` and the 0600 JSON credentials
  artifact via ``CredentialsWriter`` (OMN-16035, OMN-16038).

The ``input_adapter`` is a function parameter injected by the caller,
NOT part of the Pydantic model (DI outside models — OMN-10784 GPT #1).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import SecretStr, ValidationError

from omnibase_core.types import StrictJsonType
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
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
from omnibase_infra.onboarding.credentials_writer import (
    CredentialsWriter,
    select_credential_entries,
)
from omnibase_infra.onboarding.interactive_executor import InteractiveExecutor
from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.model_credential_store_write import (
    ModelCredentialStoreWrite,
)
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
from omnibase_infra.runtime.overlay.overlay_from_env import overlay_from_env_dict
from omnibase_infra.runtime.overlay.overlay_writer import OverlayWriter


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


def _write_credential_store(
    credential: ModelCredentialStoreWrite,
    credentials_output_path: str | None,
) -> str:
    """Store the collected credential through the reader's own writer.

    The store owns two files under one ``~/.onex`` root -- the reference-only
    ``config.yaml`` block and the 0600 ``credentials.json`` -- and writing them
    through it is the whole point of this path: it is what makes the credential
    onboarding collects the credential ``onex auth status`` resolves, instead
    of two files that merely coexist (OMN-17028).

    The root is the DIRECTORY of the caller's credentials path, and the file
    name must be the store's own. That is not a guess about the caller's
    intent: the store defines both file names relative to one root, so a
    caller naming ``.../my-creds.json`` is naming a file this store will never
    read back, and writing it anyway would rebuild the exact writer/reader
    split this function exists to close.

    Raises:
        OnboardingHandlerError: If no credentials output path was supplied, or
            the path names a file the store does not read.
    """
    if credentials_output_path is None:
        msg = (
            "credentials_output_path is required when the policy declares "
            "credential_store_output and dry_run=False -- refusing to discard "
            "collected secret material"
        )
        raise OnboardingHandlerError(msg)

    requested = Path(credentials_output_path)
    store = StoreGatewayCredential(onex_home=requested.parent)
    if requested.name != store.credentials_path.name:
        msg = (
            f"credentials_output_path names '{requested.name}', but the "
            f"credential store reads '{store.credentials_path.name}' under the "
            f"~/.onex root. Point it at "
            f"'{requested.parent / store.credentials_path.name}' -- a file the "
            f"store does not read is the writer/reader split this path closes"
        )
        raise OnboardingHandlerError(msg)

    store.save_api_key(
        tenant_slug=credential.tenant_slug,
        # The one line that unwraps the key, and it goes straight to the 0600
        # file the store writes.
        api_key=credential.api_key.get_secret_value(),
        base_url=credential.base_url,
    )
    return str(store.credentials_path)


def _write_declared_credentials(
    credentials: Mapping[str, SecretStr],
    credentials_output_path: str | None,
) -> str:
    """Write a policy's declared secrets as the flat ``{ref: secret}`` document.

    Refuses rather than silently dropping the secrets when the caller gave no
    path: the policy declared credentials, so a live run that quietly wrote
    none of them would leave the operator believing the machine is attached
    when the very next gateway call cannot find a credential.

    Raises:
        OnboardingHandlerError: If no credentials output path was supplied.
    """
    if credentials_output_path is None:
        msg = (
            "credentials_output_path is required when the policy declares "
            "credentials_output and dry_run=False — refusing to discard "
            "collected secret material"
        )
        raise OnboardingHandlerError(msg)

    credentials_path = Path(credentials_output_path)
    payload: dict[str, StrictJsonType] = {
        ref: secret.get_secret_value() for ref, secret in credentials.items()
    }
    CredentialsWriter().write(payload, credentials_path)
    return str(credentials_path)


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

    # Optionally write overlay, env config, and the credentials artifact
    env_output_path_written: str | None = None
    overlay_output_path_written: str | None = None
    credentials_output_path_written: str | None = None
    if not input_model.dry_run:
        # ``env_output_path`` is required only where it is actually consumed:
        # for the legacy .env write, and as the directory the overlay path is
        # derived from when no explicit overlay path was given. Requiring it
        # unconditionally contradicted this model's own validator, which
        # accepts ``legacy_env_output=False`` + an explicit overlay path with
        # no env path at all — the combination a credentials-only policy such
        # as ``connect_cloud`` uses.
        target_path = (
            Path(input_model.env_output_path)
            if input_model.env_output_path is not None
            else None
        )

        # Overlay YAML, written only where a destination was actually named.
        # A credential-only policy names none: its whole output is the stored
        # credential, and writing an overlay anyway would put a second,
        # unread copy of the same coordinates on disk — which is the exact
        # shape (a file the reader never opens) this lane exists to stop
        # producing (OMN-17028).
        overlay_path: Path | None = None
        if input_model.overlay_output_path is not None:
            overlay_path = Path(input_model.overlay_output_path)
        elif target_path is not None:
            overlay_path = target_path.parent / "overlay.yaml"
        if overlay_path is not None:
            overlay = overlay_from_env_dict(
                result.env_dict, environment="dev", return_warnings=False
            )
            OverlayWriter().write(overlay, overlay_path)
            overlay_output_path_written = str(overlay_path)

        # Legacy .env output behind flag
        if input_model.legacy_env_output:
            if target_path is None:  # pragma: no cover - rejected by the validator
                msg = "env_output_path is required when legacy_env_output=True"
                raise OnboardingHandlerError(msg)
            writer = ConfigWriter()
            writer.write(result.env_dict, target_path)
            env_output_path_written = str(target_path)

        # Credentials artifact: explicit invocation only — it fires only when
        # the caller named a path, so an onboarding run without secrets writes
        # no 0600 file. Three sources, most-declared first:
        #
        #   result.credential_store_write (OMN-17028) — the policy named the
        #     credential STORE, so the store writes both of its files and the
        #     credential is readable by `onex auth status` the moment the run
        #     ends. This is the only branch that leaves the machine
        #     authenticated; the two below leave files on disk that the
        #     credential reader never opens on their own.
        #
        #   result.credentials_dict (OMN-16038) — the policy said, in its
        #     credentials_output block, exactly which refs carry secrets. The
        #     document is written FLAT ({ref: secret}), because that is the
        #     shape StoreGatewayCredential reads; nesting it under the policy
        #     name would produce a file `onex auth` cannot resolve.
        #
        #   select_credential_entries(env_dict) (OMN-16035) — the fallback for
        #     policies that predate credentials_output: infer secrets from env
        #     key names and nest them under the policy name.
        #
        # A policy that declares either block has stated its secrets exactly,
        # so the name-shaped guess is not also applied to it.
        if result.credential_store_write is not None:
            credentials_output_path_written = _write_credential_store(
                result.credential_store_write, input_model.credentials_output_path
            )
        elif result.credentials_dict:
            credentials_output_path_written = _write_declared_credentials(
                result.credentials_dict, input_model.credentials_output_path
            )
        elif input_model.credentials_output_path is not None:
            credential_entries = select_credential_entries(result.env_dict)
            if credential_entries:
                credentials_path = Path(input_model.credentials_output_path)
                entries: dict[str, StrictJsonType] = dict(credential_entries)
                credentials_payload: dict[str, StrictJsonType] = {
                    "credentials": {input_model.policy_name: entries}
                }
                CredentialsWriter().write(credentials_payload, credentials_path)
                credentials_output_path_written = str(credentials_path)

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
        if overlay_output_path_written:
            rendered += f"\n*Overlay written to: {overlay_output_path_written}*\n"
        if env_output_path_written:
            rendered += f"\n*Env written to: {env_output_path_written}*\n"
        if credentials_output_path_written:
            rendered += (
                f"\n*Credentials written to: {credentials_output_path_written} "
                f"(mode 0600)*\n"
            )
        # Naming config.yaml explicitly is not decoration: the whole defect
        # this path closes was an operator being told onboarding succeeded
        # while the file the credential reader opens had never been written.
        # Saying which files now exist is what makes that visible.
        if (
            result.credential_store_write is not None
            and credentials_output_path_written
        ):
            store_root = Path(credentials_output_path_written).parent
            rendered += (
                f"\n*Gateway credential stored under: {store_root} "
                f"— verify with `onex auth status`*\n"
            )

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
        overlay_output_path_written=overlay_output_path_written,
        credentials_output_path_written=credentials_output_path_written,
    )


def _skipped_step_result(step: ModelOnboardingStep) -> ModelStepResult:
    return ModelStepResult(
        step_key=step.step_key,
        passed=False,
        message="Skipped due to previous failure",
    )


async def _execute_dag_step(step: ModelOnboardingStep) -> ModelStepResult:
    if step.verification is None:
        return ModelStepResult(
            step_key=step.step_key,
            passed=True,
            message="No verification defined",
        )

    spec = ModelVerificationSpec(
        check_type=step.verification.check_type,
        target=step.verification.target,
        timeout_seconds=step.verification.timeout_seconds or 10,
    )
    result = await execute_verification(spec)
    return ModelStepResult(
        step_key=step.step_key,
        passed=result.passed,
        message=result.message,
        elapsed_ms=result.elapsed_ms,
    )


def _collect_capability_evidence(
    steps: list[ModelOnboardingStep],
    step_results: list[ModelStepResult],
) -> tuple[list[str], list[str]]:
    verified_caps: list[str] = []
    unmet_caps: list[str] = []
    for step, step_res in zip(steps, step_results, strict=True):
        target = verified_caps if step_res.passed else unmet_caps
        target.extend(step.produces_capabilities)
    return verified_caps, unmet_caps


def _status_from_step_results(
    step_results: list[ModelStepResult],
) -> EnumOnboardingStatus:
    if all(r.passed for r in step_results):
        return EnumOnboardingStatus.PASSED
    if any(r.message == "Skipped due to previous failure" for r in step_results):
        return EnumOnboardingStatus.BLOCKED
    if any(r.passed for r in step_results):
        return EnumOnboardingStatus.PARTIAL
    return EnumOnboardingStatus.FAILED


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
            step_results.append(_skipped_step_result(step))
            continue

        step_result = await _execute_dag_step(step)
        step_results.append(step_result)
        if step_result.passed:
            completed_steps.append(step)
        else:
            failed = True

    # Render output — pass all steps (not just completed) so failed/skipped appear
    renderer = RendererOnboardingMarkdown()
    rendered = renderer.render(steps, step_results, title="Onboarding Progress")

    verified_caps, unmet_caps = _collect_capability_evidence(steps, step_results)
    status = _status_from_step_results(step_results)

    return ModelOnboardingOutput(
        success=status == EnumOnboardingStatus.PASSED,
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
