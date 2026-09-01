# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the ``connect_cloud`` onboarding policy (OMN-16038, OMN-17028).

The policy walks a beta customer through authenticating this machine to the
ONEX cloud gateway and produces the on-disk credential ``onex auth status``
reads back. Four properties are load-bearing and each has a test that fails if
it regresses:

1. The prompt order is the onboarding sequence: origin, tenant, key.
2. The API key is collected through the *masked* adapter path and is never
   carried in the step-result receipt, the env output, or any serialization of
   the result model.
3. The policy hands its credential to the credential STORE, not to a file
   writer — ``credential_store_output``, whose fields are the store's own write
   signature. This is the OMN-17028 property: what onboarding writes is by
   construction what the credential reader reads. A policy that emitted only a
   file would restore the defect.
4. The policy asks for no attach-plane field. ``client_id``,
   ``token_endpoint``, ``client_secret`` and ``edge_instance_id`` belong to the
   gateway attach/relay control plane and are not on the delegation path; the
   previous revision prompted for three of them, which is what made a beta
   customer's onboarding ask for values they do not have.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
_FAKE_KEY = "onxk_not-a-real-key-0f9a"  # pragma: allowlist secret
_BASE_URL = "https://dev.api.omninode.ai"

_RESPONSES: dict[str, str | list[str]] = {
    "gateway_base_url": _BASE_URL,
    "tenant_slug": "acme",
    "gateway_api_key": _FAKE_KEY,
}

_EXPECTED_PROMPT_ORDER = [
    "gateway_base_url",
    "tenant_slug",
    "gateway_api_key",
]

#: Fields of the credential the gateway ATTACH plane resolves. None of them is
#: on the delegation path, so none of them may be prompted for here.
_ATTACH_PLANE_FIELDS = (
    "client_id",
    "client_secret",
    "token_endpoint",
    "edge_instance_id",
)


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

    def test_only_the_api_key_step_is_masked(
        self, policy: ModelInteractivePolicy
    ) -> None:
        masked = [step.id for step in policy.steps if step.secret]
        assert masked == ["gateway_api_key"]

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
        """The key is store-only; env/overlay must never reference it."""
        for template in policy.env_output.values():
            for value in template.values():
                assert "gateway_api_key" not in value

    def test_no_attach_plane_field_is_prompted_for(
        self, policy: ModelInteractivePolicy
    ) -> None:
        """OMN-17028: the delegation path presents a key, and nothing else.

        Asserted over the prompts themselves rather than over the policy's
        declared outputs, because the defect was a customer being ASKED for a
        machine-client secret they cannot obtain — the cost lands at the prompt
        whether or not the value is later stored.
        """
        prompted = {step.id for step in policy.steps}
        for field in _ATTACH_PLANE_FIELDS:
            assert not any(field in step_id for step_id in prompted), (
                f"'{field}' is an attach-plane field; connect_cloud is the "
                f"delegation-path onboarding and must not prompt for it"
            )


class TestCredentialStoreEmission:
    """The OMN-17028 handoff: the policy names a store, not a file."""

    def test_policy_declares_the_store_write_not_a_bare_file(
        self, policy: ModelInteractivePolicy
    ) -> None:
        assert "write_credentials" in policy.credential_store_output
        # Mutually exclusive by construction — one writer owns credentials.json.
        assert policy.credentials_output == {}

    def test_reducer_emits_the_stores_own_write_signature(
        self, policy: ModelInteractivePolicy
    ) -> None:
        reducer = TransitionReducer(policy)
        state: dict[str, object] = {
            "gateway_base_url": _BASE_URL,
            "tenant_slug": "acme",
            "gateway_api_key": _FAKE_KEY,
        }

        write = reducer.get_credential_store_output("write_credentials", state)

        assert write is not None
        assert write.tenant_slug == "acme"
        assert write.base_url == _BASE_URL
        assert write.api_key.get_secret_value() == _FAKE_KEY

    def test_env_names_the_key_ref_the_store_itself_writes(
        self, policy: ModelInteractivePolicy
    ) -> None:
        """The rendered ref must match ``save_api_key``'s, or it names nothing."""
        reducer = TransitionReducer(policy)
        state: dict[str, object] = {
            "gateway_base_url": _BASE_URL,
            "tenant_slug": "acme",
            "gateway_api_key": _FAKE_KEY,
        }

        env = reducer.get_env_output("write_credentials", state)

        assert env["ONEX_GATEWAY_BASE_URL"] == _BASE_URL
        assert env["ONEX_GATEWAY_TENANT_SLUG"] == "acme"
        assert env["ONEX_GATEWAY_API_KEY_REF"] == "acme-api-key"
        assert _FAKE_KEY not in "".join(env.values())


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
    async def test_key_is_collected_through_the_masked_adapter_path(
        self, policy: ModelInteractivePolicy, adapter: AdapterFakeInput
    ) -> None:
        await InteractiveExecutor(policy, adapter).execute()

        assert adapter.secret_steps_collected == ["gateway_api_key"]

    @pytest.mark.asyncio
    async def test_key_never_appears_in_the_receipt(
        self, policy: ModelInteractivePolicy, adapter: AdapterFakeInput
    ) -> None:
        result = await InteractiveExecutor(policy, adapter).execute()

        # Step-result receipt carries a redaction placeholder, not the value.
        secret_step = next(
            sr for sr in result.step_results if sr.step_key == "gateway_api_key"
        )
        assert secret_step.response != _FAKE_KEY

        # Nor does any serialization of the whole result leak it.
        assert _FAKE_KEY not in result.model_dump_json()
        assert _FAKE_KEY not in repr(result)
        assert _FAKE_KEY not in str(result.credential_store_write)

    @pytest.mark.asyncio
    async def test_executor_emits_the_credential_for_the_store(
        self, policy: ModelInteractivePolicy, adapter: AdapterFakeInput
    ) -> None:
        result = await InteractiveExecutor(policy, adapter).execute()

        assert result.credential_store_write is not None
        assert result.credential_store_write.api_key.get_secret_value() == _FAKE_KEY
        assert result.credentials_dict == {}
        assert _FAKE_KEY not in "".join(result.env_dict.values())
