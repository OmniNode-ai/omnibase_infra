# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel hands the webhook ingress its SecretResolver only where the lane maps one (OMN-19492).

``service_kernel._build_runtime_handler_dependencies`` is resolver Step 2, the
only place a handler gets a constructor argument the resolver has no provider
for. The ingress takes an optional ``secret_resolver``; without one it refuses
every delivery. These tests drive the real builder against the real runtime
policy contract, so a profile that loses the mapping, or a builder that stops
supplying it, fails here and not as a dev lane that silently refuses every
GitHub delivery.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_github_webhook_ingress_effect.handlers.handler_github_webhook_ingress import (
    WEBHOOK_SECRET_REF,
)
from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

pytestmark = pytest.mark.unit

CONTRACT_PATH = (
    Path(__file__).parents[3]
    / "contracts"
    / "services"
    / "runtime_policy.contract.yaml"
)


def _profile_config(tmp_path: Path, profile: str) -> Path:
    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    mappings = [
        m.model_dump(mode="json")
        for m in contract.profiles[profile].secret_resolver_mappings
    ]
    path = tmp_path / f"{profile}-secret-resolver.yaml"
    path.write_text(
        yaml.safe_dump({"enable_convention_fallback": False, "mappings": mappings}),
        encoding="utf-8",
    )
    return path


def test_the_dev_profile_gives_the_ingress_a_resolver(tmp_path: Path) -> None:
    deps = _build_runtime_handler_dependencies(
        postgres_pool=None,
        kafka_bootstrap_servers=None,
        gateway_secret_resolver_config_path=_profile_config(tmp_path, "dev"),
    )
    assert deps is not None
    assert isinstance(
        deps["HandlerGitHubWebhookIngress"]["secret_resolver"], SecretResolver
    )


@pytest.mark.parametrize("profile", ["stability-test", "judge", "lakshman", "dogfood"])
def test_profiles_without_the_mapping_leave_the_ingress_closed(
    tmp_path: Path, profile: str
) -> None:
    deps = _build_runtime_handler_dependencies(
        postgres_pool=None,
        kafka_bootstrap_servers=None,
        gateway_secret_resolver_config_path=_profile_config(tmp_path, profile),
    )
    assert deps is None or "HandlerGitHubWebhookIngress" not in deps


def test_the_dev_mapping_names_the_logical_secret_the_handler_resolves() -> None:
    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    names = {m.logical_name for m in contract.profiles["dev"].secret_resolver_mappings}
    assert WEBHOOK_SECRET_REF in names
