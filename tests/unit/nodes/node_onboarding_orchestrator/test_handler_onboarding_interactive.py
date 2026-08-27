# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for interactive handler integration (OMN-10784 Task 7).

Verifies:
- Interactive path with FakeInputAdapter produces correct output
- dry_run=True (default) does not call ConfigWriter
- dry_run=False calls ConfigWriter with correct args
- policy_name=None falls through to existing DAG path
- DAG path regression (output unchanged by handler modifications)
- Missing adapter raises OnboardingHandlerError
- Unknown policy name raises OnboardingHandlerError
- env_output_path required when dry_run=False
- credentials artifact wiring, permissions, and merge behavior (OMN-16035)
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
    OnboardingHandlerError,
    handle_onboarding,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_input import (
    ModelOnboardingInput,
)
from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput
from omnibase_infra.onboarding.model_interactive_result import ModelInteractiveResult
from omnibase_infra.probes.model_verification_result import ModelVerificationResult

pytestmark = pytest.mark.unit


# --- Fixtures ---


@pytest.fixture
def local_no_llm_adapter() -> AdapterFakeInput:
    """Adapter driving the local-no-llm path through interactive_onboarding."""
    return AdapterFakeInput(
        responses={
            "choose_deployment_mode": "local",
            "configure_local_services": ["kafka", "postgres"],
        }
    )


@pytest.fixture
def cloud_aws_adapter() -> AdapterFakeInput:
    """Adapter driving the cloud-aws path."""
    return AdapterFakeInput(
        responses={
            "choose_deployment_mode": "cloud",
            "configure_cloud_provider": "aws",
            "configure_aws_region": "us-east-1",
        }
    )


def _make_passing_result(
    check_type: str = "command_exit_0", target: str = "test"
) -> ModelVerificationResult:
    return ModelVerificationResult(
        passed=True,
        check_type=check_type,
        target=target,
        message="Passed",
        elapsed_ms=10,
    )


# --- Interactive path tests ---


class TestHandlerInteractivePath:
    """Tests for interactive executor dispatch via handle_onboarding."""

    @pytest.mark.asyncio
    async def test_interactive_local_no_llm_produces_correct_output(
        self,
        local_no_llm_adapter: AdapterFakeInput,
    ) -> None:
        """Interactive local path produces correct env output and provenance."""
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=True,
        )
        output = await handle_onboarding(
            input_model, input_adapter=local_no_llm_adapter
        )

        assert output.success is True
        assert output.policy_name == "interactive_onboarding"
        assert output.policy_type == "interactive"
        assert output.terminal_step == "write_config_local"
        assert output.dry_run is True
        assert output.env_output_path_written is None

        # Provenance carries full interactive result
        assert output.provenance is not None
        assert output.provenance.env_dict["ONEX_DEPLOYMENT_MODE"] == "local"
        assert (
            output.provenance.env_dict["KAFKA_BOOTSTRAP_SERVERS"] == "localhost:19092"
        )
        assert output.provenance.completed is True

        # Visited steps are recorded
        assert "choose_deployment_mode" in output.visited_steps
        assert "configure_local_services" in output.visited_steps

        # Rendered output contains env values
        assert "ONEX_DEPLOYMENT_MODE=local" in output.rendered_output
        assert "Dry run" in output.rendered_output

    @pytest.mark.asyncio
    async def test_interactive_cloud_aws_produces_correct_output(
        self,
        cloud_aws_adapter: AdapterFakeInput,
    ) -> None:
        """Interactive cloud-aws path produces correct env output."""
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=True,
        )
        output = await handle_onboarding(input_model, input_adapter=cloud_aws_adapter)

        assert output.success is True
        assert output.terminal_step == "write_config_cloud"
        assert output.provenance is not None
        assert output.provenance.env_dict["ONEX_DEPLOYMENT_MODE"] == "cloud"
        assert output.provenance.env_dict["CLOUD_PROVIDER"] == "aws"
        assert output.provenance.env_dict["AWS_REGION"] == "us-east-1"

    @pytest.mark.asyncio
    async def test_dry_run_true_does_not_write_files(
        self,
        local_no_llm_adapter: AdapterFakeInput,
    ) -> None:
        """dry_run=True (default) must not invoke ConfigWriter.write."""
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=True,
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.ConfigWriter"
        ) as mock_writer_cls:
            output = await handle_onboarding(
                input_model, input_adapter=local_no_llm_adapter
            )

        # ConfigWriter should not have been instantiated at all
        mock_writer_cls.assert_not_called()
        assert output.env_output_path_written is None
        assert output.dry_run is True

    @pytest.mark.asyncio
    async def test_dry_run_false_calls_config_writer(
        self,
        local_no_llm_adapter: AdapterFakeInput,
        tmp_path: object,
    ) -> None:
        """dry_run=False invokes ConfigWriter.write with correct args."""
        from pathlib import Path

        target = Path(str(tmp_path)) / "test.env"
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=False,
            env_output_path=str(target),
        )

        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.ConfigWriter"
        ) as mock_writer_cls:
            mock_instance = mock_writer_cls.return_value
            mock_instance.write.return_value = "rendered content"
            output = await handle_onboarding(
                input_model, input_adapter=local_no_llm_adapter
            )

        mock_instance.write.assert_called_once()
        call_args = mock_instance.write.call_args
        env_dict_arg = call_args[0][0]
        path_arg = call_args[0][1]

        assert env_dict_arg["ONEX_DEPLOYMENT_MODE"] == "local"
        assert path_arg == target
        assert output.env_output_path_written == str(target)
        assert output.dry_run is False

    @pytest.mark.asyncio
    async def test_dry_run_false_writes_overlay_and_sets_path(
        self,
        local_no_llm_adapter: AdapterFakeInput,
        tmp_path: object,
    ) -> None:
        """dry_run=False writes overlay YAML and sets overlay_output_path_written."""
        from pathlib import Path

        target = Path(str(tmp_path)) / "test.env"
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=False,
            env_output_path=str(target),
        )
        output = await handle_onboarding(
            input_model, input_adapter=local_no_llm_adapter
        )

        assert output.overlay_output_path_written is not None
        overlay_path = Path(output.overlay_output_path_written)
        assert overlay_path.exists()
        assert overlay_path.suffix == ".yaml"

    @pytest.mark.asyncio
    async def test_dry_run_true_overlay_output_path_written_is_none(
        self,
        local_no_llm_adapter: AdapterFakeInput,
    ) -> None:
        """dry_run=True must leave overlay_output_path_written as None."""
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=True,
        )
        output = await handle_onboarding(
            input_model, input_adapter=local_no_llm_adapter
        )

        assert output.overlay_output_path_written is None

    async def test_dry_run_false_without_env_output_path_raises(
        self,
        local_no_llm_adapter: AdapterFakeInput,
    ) -> None:
        """dry_run=False without env_output_path is rejected at model construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="env_output_path is required"):
            ModelOnboardingInput(
                policy_name="interactive_onboarding",
                dry_run=False,
                env_output_path=None,
            )

    def test_dry_run_false_blank_env_output_path_raises(self) -> None:
        """Provided env_output_path must not be blank."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="env_output_path cannot be blank"):
            ModelOnboardingInput(
                policy_name="interactive_onboarding",
                dry_run=False,
                env_output_path="   ",
                overlay_output_path="overlay.yaml",
            )

    def test_dry_run_false_blank_overlay_output_path_raises(self) -> None:
        """Provided overlay_output_path must not be blank."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError, match="overlay_output_path cannot be blank"
        ):
            ModelOnboardingInput(
                policy_name="interactive_onboarding",
                dry_run=False,
                env_output_path="onboarding.env",
                overlay_output_path="   ",
            )

    @pytest.mark.asyncio
    async def test_interactive_without_adapter_raises(self) -> None:
        """Interactive policy without input_adapter raises OnboardingHandlerError."""
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
        )
        with pytest.raises(OnboardingHandlerError, match="input_adapter is required"):
            await handle_onboarding(input_model, input_adapter=None)

    @pytest.mark.asyncio
    async def test_unknown_policy_raises(self) -> None:
        """Unknown policy_name raises OnboardingHandlerError."""
        adapter = AdapterFakeInput(responses={})
        input_model = ModelOnboardingInput(
            policy_name="nonexistent_policy",
        )
        with pytest.raises(OnboardingHandlerError, match="not found"):
            await handle_onboarding(input_model, input_adapter=adapter)

    @pytest.mark.asyncio
    async def test_handler_class_wrapper_delegates_interactive(
        self,
        local_no_llm_adapter: AdapterFakeInput,
    ) -> None:
        """HandlerOnboarding class wrapper correctly delegates interactive calls."""
        from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
            HandlerOnboarding,
        )

        handler = HandlerOnboarding()
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=True,
        )
        output = await handler.handle(input_model, input_adapter=local_no_llm_adapter)

        assert output.success is True
        assert output.policy_name == "interactive_onboarding"
        assert output.provenance is not None


# --- DAG path regression tests ---


class TestHandlerDAGPathRegression:
    """Verify the DAG path is unchanged by the interactive handler modifications."""

    @pytest.mark.asyncio
    async def test_dag_path_with_policy_name_none(self) -> None:
        """policy_name=None routes to DAG path, not interactive."""
        input_model = ModelOnboardingInput(
            target_capabilities=["first_node_running"],
            policy_name=None,
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            new_callable=AsyncMock,
            return_value=_make_passing_result(),
        ):
            output = await handle_onboarding(input_model)

        assert output.success is True
        assert output.total_steps == 5
        assert output.completed_steps == 5
        assert len(output.step_results) == 5
        assert all(r.passed for r in output.step_results)
        assert "Onboarding Progress" in output.rendered_output

        # Interactive provenance fields should be None/empty
        assert output.provenance is None
        assert output.policy_name is None
        assert output.policy_type is None
        assert output.visited_steps == []
        assert output.terminal_step is None

    @pytest.mark.asyncio
    async def test_dag_path_default_input(self) -> None:
        """Default ModelOnboardingInput (no policy_name) routes to DAG path."""
        input_model = ModelOnboardingInput(
            target_capabilities=["first_node_running"],
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            new_callable=AsyncMock,
            return_value=_make_passing_result(),
        ):
            output = await handle_onboarding(input_model)

        assert output.success is True
        assert output.provenance is None

    @pytest.mark.asyncio
    async def test_dag_path_failure_still_stops(self) -> None:
        """DAG path failure behavior is unchanged by handler modifications."""
        input_model = ModelOnboardingInput(
            target_capabilities=["first_node_running"],
        )
        mock = AsyncMock(
            side_effect=[
                _make_passing_result(),
                ModelVerificationResult(
                    passed=False,
                    check_type="command_exit_0",
                    target="test",
                    message="Failed",
                    elapsed_ms=10,
                ),
                _make_passing_result(),
                _make_passing_result(),
                _make_passing_result(),
            ]
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            mock,
        ):
            output = await handle_onboarding(input_model)

        assert output.success is False
        assert output.completed_steps == 1
        skipped = [r for r in output.step_results if "Skipped" in r.message]
        assert len(skipped) == 3

    @pytest.mark.asyncio
    async def test_dag_rendered_output_format_unchanged(self) -> None:
        """DAG rendered output format is byte-compatible with pre-interactive handler."""
        input_model = ModelOnboardingInput(
            target_capabilities=["first_node_running"],
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            new_callable=AsyncMock,
            return_value=_make_passing_result(),
        ):
            output = await handle_onboarding(input_model)

        # Same format markers as original handler
        assert "Onboarding Progress" in output.rendered_output
        assert "[x]" in output.rendered_output
        assert "steps passed" in output.rendered_output


# --- Credentials artifact wiring (OMN-16035) ---


def _credential_bearing_result() -> ModelInteractiveResult:
    """Interactive result whose env output carries credential-shaped entries.

    The shipped ``interactive_onboarding`` policy emits no secrets today, so the
    executor is stubbed to exercise the credentials call site directly.
    """
    return ModelInteractiveResult(
        env_dict={
            "ONEX_DEPLOYMENT_MODE": "local",
            "KAFKA_BOOTSTRAP_SERVERS": "localhost:19092",
            "POSTGRES_PASSWORD": "pg-secret",
            "ONEX_API_TOKEN": "tok-123",
        },
        step_results=[],
        policy_name="interactive_onboarding",
        completed=True,
        terminal_step="write_config_local",
    )


def _patch_executor() -> object:
    """Patch InteractiveExecutor so execute() returns the credential-bearing result."""
    patcher = patch(
        "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.InteractiveExecutor"
    )
    mock_cls = patcher.start()
    mock_cls.return_value.execute = AsyncMock(return_value=_credential_bearing_result())
    return patcher


class TestHandlerCredentialsArtifact:
    """The credentials writer is wired next to ConfigWriter, explicit-invocation only."""

    @pytest.mark.asyncio
    async def test_credentials_written_at_mode_0600_with_only_secret_keys(
        self,
        local_no_llm_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        credentials_path = tmp_path / "credentials.json"
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=False,
            env_output_path=str(tmp_path / "test.env"),
            credentials_output_path=str(credentials_path),
        )

        patcher = _patch_executor()
        try:
            output = await handle_onboarding(
                input_model, input_adapter=local_no_llm_adapter
            )
        finally:
            patcher.stop()  # type: ignore[attr-defined]

        assert output.credentials_output_path_written == str(credentials_path)
        assert stat.S_IMODE(credentials_path.stat().st_mode) == 0o600

        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
        entries = payload["credentials"]["interactive_onboarding"]
        assert entries == {
            "POSTGRES_PASSWORD": "pg-secret",
            "ONEX_API_TOKEN": "tok-123",
        }
        assert "ONEX_DEPLOYMENT_MODE" not in entries
        assert "KAFKA_BOOTSTRAP_SERVERS" not in entries
        assert "mode 0600" in output.rendered_output

    @pytest.mark.asyncio
    async def test_credentials_merge_preserves_other_policy_entries(
        self,
        local_no_llm_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        credentials_path = tmp_path / "credentials.json"
        credentials_path.write_text(
            json.dumps({"credentials": {"other_policy": {"OTHER_TOKEN": "keep"}}}),
            encoding="utf-8",
        )
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=False,
            env_output_path=str(tmp_path / "test.env"),
            credentials_output_path=str(credentials_path),
        )

        patcher = _patch_executor()
        try:
            await handle_onboarding(input_model, input_adapter=local_no_llm_adapter)
        finally:
            patcher.stop()  # type: ignore[attr-defined]

        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
        assert payload["credentials"]["other_policy"] == {"OTHER_TOKEN": "keep"}
        assert "ONEX_API_TOKEN" in payload["credentials"]["interactive_onboarding"]

    @pytest.mark.asyncio
    async def test_no_credentials_path_writes_no_credentials_file(
        self,
        local_no_llm_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        """Explicit-invocation only: no path means the writer never runs."""
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=False,
            env_output_path=str(tmp_path / "test.env"),
        )

        patcher = _patch_executor()
        try:
            with patch(
                "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.CredentialsWriter"
            ) as mock_writer_cls:
                output = await handle_onboarding(
                    input_model, input_adapter=local_no_llm_adapter
                )
        finally:
            patcher.stop()  # type: ignore[attr-defined]

        mock_writer_cls.assert_not_called()
        assert output.credentials_output_path_written is None

    @pytest.mark.asyncio
    async def test_dry_run_never_writes_credentials(
        self,
        local_no_llm_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        credentials_path = tmp_path / "credentials.json"
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=True,
            credentials_output_path=str(credentials_path),
        )

        patcher = _patch_executor()
        try:
            with patch(
                "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.CredentialsWriter"
            ) as mock_writer_cls:
                output = await handle_onboarding(
                    input_model, input_adapter=local_no_llm_adapter
                )
        finally:
            patcher.stop()  # type: ignore[attr-defined]

        mock_writer_cls.assert_not_called()
        assert output.credentials_output_path_written is None
        assert not credentials_path.exists()

    @pytest.mark.asyncio
    async def test_run_without_credential_shaped_keys_writes_nothing(
        self,
        local_no_llm_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        """The shipped policy emits no secrets, so no 0600 file is created."""
        credentials_path = tmp_path / "credentials.json"
        input_model = ModelOnboardingInput(
            policy_name="interactive_onboarding",
            dry_run=False,
            env_output_path=str(tmp_path / "test.env"),
            credentials_output_path=str(credentials_path),
        )

        output = await handle_onboarding(
            input_model, input_adapter=local_no_llm_adapter
        )

        assert output.credentials_output_path_written is None
        assert not credentials_path.exists()

    def test_blank_credentials_output_path_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError, match="credentials_output_path cannot be blank"
        ):
            ModelOnboardingInput(
                policy_name="interactive_onboarding",
                dry_run=False,
                env_output_path="out.env",
                credentials_output_path="   ",
            )


# --- connect_cloud policy end-to-end through the handler (OMN-16038) ---

_CONNECT_CLOUD_SECRET = "s3cr3t-not-a-real-client-secret-0f9a"


@pytest.fixture
def connect_cloud_adapter() -> AdapterFakeInput:
    """Adapter driving the full connect_cloud prompt flow."""
    return AdapterFakeInput(
        responses={
            "gateway_base_url": "https://api.omninode.ai",
            "tenant_slug": "acme",
            "gateway_client_id": "ga-acme-edge",
            "gateway_token_endpoint": (
                "https://keycloak.omninode.ai/realms/onex/protocol/openid-connect/token"
            ),
            "gateway_client_secret": _CONNECT_CLOUD_SECRET,
        }
    )


class TestConnectCloudPolicy:
    """The connect_cloud policy's two operator-visible modes: dry-run and live."""

    @pytest.mark.asyncio
    async def test_dry_run_renders_the_flow_and_writes_nothing(
        self,
        connect_cloud_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        credentials_path = tmp_path / "credentials.json"
        input_model = ModelOnboardingInput(
            policy_name="connect_cloud",
            dry_run=True,
            credentials_output_path=str(credentials_path),
        )

        output = await handle_onboarding(
            input_model, input_adapter=connect_cloud_adapter
        )

        assert output.success is True
        assert output.visited_steps == [
            "gateway_base_url",
            "tenant_slug",
            "gateway_client_id",
            "gateway_token_endpoint",
            "gateway_client_secret",
        ]
        assert not credentials_path.exists()
        assert output.credentials_output_path_written is None
        assert "Dry run" in output.rendered_output

    @pytest.mark.asyncio
    async def test_live_run_writes_the_credential_in_the_store_shape(
        self,
        connect_cloud_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        """The artifact must be the flat {ref: secret} document ``onex auth`` reads."""
        credentials_path = tmp_path / "credentials.json"
        input_model = ModelOnboardingInput(
            policy_name="connect_cloud",
            dry_run=False,
            legacy_env_output=False,
            overlay_output_path=str(tmp_path / "overlay.yaml"),
            credentials_output_path=str(credentials_path),
        )

        output = await handle_onboarding(
            input_model, input_adapter=connect_cloud_adapter
        )

        assert output.credentials_output_path_written == str(credentials_path)
        assert stat.S_IMODE(credentials_path.stat().st_mode) == 0o600

        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
        assert payload == {"acme-gateway": _CONNECT_CLOUD_SECRET}

    @pytest.mark.asyncio
    async def test_live_run_credential_resolves_through_the_gateway_store(
        self,
        connect_cloud_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        """Proof the shape is right: StoreGatewayCredential reads it back."""
        from omnibase_infra.gateway.client.store_gateway_credential import (
            StoreGatewayCredential,
        )

        onex_home = tmp_path / ".onex"
        input_model = ModelOnboardingInput(
            policy_name="connect_cloud",
            dry_run=False,
            legacy_env_output=False,
            overlay_output_path=str(tmp_path / "overlay.yaml"),
            credentials_output_path=str(onex_home / "credentials.json"),
        )

        output = await handle_onboarding(
            input_model, input_adapter=connect_cloud_adapter
        )
        assert output.credentials_output_path_written is not None

        # The policy owns the secret file; config.yaml is written by
        # ``onex auth login``. Compose the two and the credential resolves.
        store = StoreGatewayCredential(onex_home=onex_home)
        env = output.provenance.env_dict if output.provenance is not None else {}
        store.config_path.write_text(
            "gateway:\n"
            f"  tenant_slug: {env['ONEX_GATEWAY_TENANT_SLUG']}\n"
            f"  client_id: {env['ONEX_GATEWAY_CLIENT_ID']}\n"
            f"  client_secret_ref: {env['ONEX_GATEWAY_CLIENT_SECRET_REF']}\n"
            f"  token_endpoint: {env['ONEX_GATEWAY_TOKEN_ENDPOINT']}\n"
            f"  base_url: {env['ONEX_GATEWAY_BASE_URL']}\n"
            "  edge_instance_id: test-host\n"
        )

        credential = store.load()
        assert credential.client_id == "ga-acme-edge"
        assert credential.client_secret.get_secret_value() == _CONNECT_CLOUD_SECRET

    @pytest.mark.asyncio
    async def test_secret_is_absent_from_every_returned_surface(
        self,
        connect_cloud_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        """AC3 — no secret in the render, the step receipts, or any dump."""
        input_model = ModelOnboardingInput(
            policy_name="connect_cloud",
            dry_run=False,
            legacy_env_output=False,
            overlay_output_path=str(tmp_path / "overlay.yaml"),
            credentials_output_path=str(tmp_path / "credentials.json"),
        )

        output = await handle_onboarding(
            input_model, input_adapter=connect_cloud_adapter
        )

        assert _CONNECT_CLOUD_SECRET not in output.rendered_output
        assert _CONNECT_CLOUD_SECRET not in output.model_dump_json()
        assert _CONNECT_CLOUD_SECRET not in (tmp_path / "overlay.yaml").read_text()

    @pytest.mark.asyncio
    async def test_secret_is_absent_from_the_dry_run_render(
        self,
        connect_cloud_adapter: AdapterFakeInput,
    ) -> None:
        output = await handle_onboarding(
            ModelOnboardingInput(policy_name="connect_cloud", dry_run=True),
            input_adapter=connect_cloud_adapter,
        )

        assert _CONNECT_CLOUD_SECRET not in output.rendered_output
        assert _CONNECT_CLOUD_SECRET not in output.model_dump_json()

    @pytest.mark.asyncio
    async def test_live_run_without_a_credentials_path_refuses(
        self,
        connect_cloud_adapter: AdapterFakeInput,
        tmp_path: Path,
    ) -> None:
        """A policy that emits credentials must not silently discard them."""
        input_model = ModelOnboardingInput(
            policy_name="connect_cloud",
            dry_run=False,
            legacy_env_output=False,
            overlay_output_path=str(tmp_path / "overlay.yaml"),
        )

        with pytest.raises(OnboardingHandlerError, match="credentials_output_path"):
            await handle_onboarding(input_model, input_adapter=connect_cloud_adapter)
