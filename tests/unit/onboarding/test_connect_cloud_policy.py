# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the ``connect_cloud`` onboarding policy (OMN-16038).

The policy walks a new operator through attaching an edge to the ONEX cloud
gateway and produces the on-disk credential that ``StoreGatewayCredential``
(OMN-15922) reads back. Three properties are load-bearing and each has a test
that fails if it regresses:

1. The prompt order is the onboarding sequence the ticket specifies.
2. The client secret is collected through the *masked* adapter path and is
   never carried in the step-result receipt, the env output, or any
   serialization of the result model.
3. The emitted credentials dict is keyed by the exact secret ref
   ``<tenant_slug>-gateway`` that ``StoreGatewayCredential.save`` writes, so
   the artifact this policy produces is the artifact ``onex auth`` resolves.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr

from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput
from omnibase_infra.onboarding.interactive_executor import InteractiveExecutor
from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy
from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep
from omnibase_infra.onboarding.transition_reducer import TransitionReducer

pytestmark = pytest.mark.unit

POLICY_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "onboarding"
    / "policies"
    / "connect_cloud.yaml"
)

# A value that is obviously not a real credential but is still a unique token,
# so an assertion that it is absent from rendered text is meaningful.
_FAKE_SECRET = "s3cr3t-not-a-real-client-secret-0f9a"

_RESPONSES: dict[str, str | list[str]] = {
    "gateway_base_url": "https://api.omninode.ai",
    "tenant_slug": "acme",
    "gateway_client_id": "ga-acme-edge",
    "gateway_token_endpoint": (
        "https://keycloak.omninode.ai/realms/onex/protocol/openid-connect/token"
    ),
    "gateway_client_secret": _FAKE_SECRET,
}

_EXPECTED_PROMPT_ORDER = [
    "gateway_base_url",
    "tenant_slug",
    "gateway_client_id",
    "gateway_token_endpoint",
    "gateway_client_secret",
]


@pytest.fixture
def policy() -> ModelInteractivePolicy:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    return ModelInteractivePolicy.model_validate(raw)


@pytest.fixture
def adapter() -> AdapterFakeInput:
    return AdapterFakeInput(responses=dict(_RESPONSES))


class TestPolicyShape:
    def test_policy_file_exists_and_validates(
        self, policy: ModelInteractivePolicy
    ) -> None:
        assert policy.policy_name == "connect_cloud"
        assert policy.policy_type == "interactive"
        assert "gateway_credential_written" in policy.target_capabilities

    def test_only_the_client_secret_step_is_masked(
        self, policy: ModelInteractivePolicy
    ) -> None:
        masked = [step.id for step in policy.steps if step.secret]
        assert masked == ["gateway_client_secret"]

    def test_terminal_step_declares_the_credentials_write_action(
        self, policy: ModelInteractivePolicy
    ) -> None:
        terminal_ids = [t.from_step for t in policy.transitions if t.terminal]
        assert terminal_ids == ["write_credentials"]

        steps: dict[str, ModelInteractiveStep] = {s.id: s for s in policy.steps}
        assert steps["write_credentials"].action == "write_credentials"
        assert steps["write_credentials"].type == "action"

    def test_env_output_carries_no_secret_template(
        self, policy: ModelInteractivePolicy
    ) -> None:
        """The secret is credentials-only; env/overlay must never reference it."""
        for template in policy.env_output.values():
            for value in template.values():
                assert "gateway_client_secret" not in value


class TestReducerCredentialsEmission:
    def test_credentials_output_is_keyed_by_the_store_secret_ref(
        self, policy: ModelInteractivePolicy
    ) -> None:
        reducer = TransitionReducer(policy)
        state: dict[str, object] = {
            "tenant_slug": "acme",
            "gateway_client_secret": _FAKE_SECRET,
        }

        credentials = reducer.get_credentials_output("write_credentials", state)

        # StoreGatewayCredential.save writes f"{tenant_slug}-gateway".
        assert set(credentials) == {"acme-gateway"}
        assert credentials["acme-gateway"].get_secret_value() == _FAKE_SECRET

    def test_credentials_and_env_are_emitted_side_by_side(
        self, policy: ModelInteractivePolicy
    ) -> None:
        reducer = TransitionReducer(policy)
        state: dict[str, object] = {
            "gateway_base_url": "https://api.omninode.ai",
            "tenant_slug": "acme",
            "gateway_client_id": "ga-acme-edge",
            "gateway_token_endpoint": "https://kc/realms/onex/protocol/openid-connect/token",
            "gateway_client_secret": _FAKE_SECRET,
        }

        env = reducer.get_env_output("write_credentials", state)
        credentials = reducer.get_credentials_output("write_credentials", state)

        assert env["ONEX_GATEWAY_CLIENT_ID"] == "ga-acme-edge"
        assert env["ONEX_GATEWAY_CLIENT_SECRET_REF"] == "acme-gateway"
        assert _FAKE_SECRET not in "".join(env.values())
        assert credentials["acme-gateway"].get_secret_value() == _FAKE_SECRET


class TestExecutorFlow:
    @pytest.mark.asyncio
    async def test_prompt_order_matches_the_onboarding_sequence(
        self, policy: ModelInteractivePolicy, adapter: AdapterFakeInput
    ) -> None:
        result = await InteractiveExecutor(policy, adapter).execute()

        assert [sr.step_key for sr in result.step_results] == _EXPECTED_PROMPT_ORDER
        assert result.completed is True
        assert result.terminal_step == "write_credentials"

    @pytest.mark.asyncio
    async def test_secret_is_collected_through_the_masked_adapter_path(
        self, policy: ModelInteractivePolicy, adapter: AdapterFakeInput
    ) -> None:
        await InteractiveExecutor(policy, adapter).execute()

        assert adapter.secret_steps_collected == ["gateway_client_secret"]

    @pytest.mark.asyncio
    async def test_secret_never_appears_in_the_receipt(
        self, policy: ModelInteractivePolicy, adapter: AdapterFakeInput
    ) -> None:
        result = await InteractiveExecutor(policy, adapter).execute()

        # Step-result receipt carries a redaction placeholder, not the value.
        secret_step = next(
            sr for sr in result.step_results if sr.step_key == "gateway_client_secret"
        )
        assert secret_step.response != _FAKE_SECRET

        # Nor does any serialization of the whole result leak it.
        assert _FAKE_SECRET not in result.model_dump_json()
        assert _FAKE_SECRET not in repr(result)
        assert _FAKE_SECRET not in str(result.credentials_dict)

    @pytest.mark.asyncio
    async def test_executor_emits_the_credential_for_the_writer(
        self, policy: ModelInteractivePolicy, adapter: AdapterFakeInput
    ) -> None:
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.credentials_dict == {"acme-gateway": SecretStr(_FAKE_SECRET)}
        assert (
            result.credentials_dict["acme-gateway"].get_secret_value() == _FAKE_SECRET
        )
        assert _FAKE_SECRET not in "".join(result.env_dict.values())
