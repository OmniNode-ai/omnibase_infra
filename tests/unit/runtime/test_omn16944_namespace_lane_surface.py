# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16944: a namespace rule survives the whole lane-config surface.

AC1 requires the rule to be *lane-declared*, so the mechanism is only real if a
namespace declared on a runtime profile reaches the rendered
``secret_resolver.yaml`` the deployed process actually reads, and resolves there
without any further edit.

This walks the real surface end to end:

    ModelRuntimeProfilePolicy.secret_resolver_namespaces
      -> ONEX_SECRET_RESOLVER_CONFIG_JSON (render_runtime_policy_env shape)
      -> render_secret_resolver_config() -> secret_resolver.yaml on disk
      -> ModelSecretResolverConfig -> SecretResolver

and then asserts a credential ref *invented after* the render resolves to a
source, while the same resolver still refuses a ref the rule does not claim.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_process_policy import (
    ModelRuntimeProcessPolicy,
)
from omnibase_infra.runtime.models.model_runtime_profile_policy import (
    ModelRuntimeProfilePolicy,
)
from omnibase_infra.runtime.models.model_secret_namespace_rule import (
    ModelSecretNamespaceRule,
)
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.render_secret_resolver_config import (
    render_secret_resolver_config,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

pytestmark = pytest.mark.unit

TENANT_CREDENTIAL_REF_PATTERN = r"^cred_[A-Za-z0-9._:-]+_[A-Za-z0-9_-]+_[0-9a-f]{32}$"


def _tenant_credential_namespace() -> ModelSecretNamespaceRule:
    return ModelSecretNamespaceRule(
        namespace="tenant_inference_credentials",
        ref_pattern=TENANT_CREDENTIAL_REF_PATTERN,
        source_type="infisical",
        source_path_template="{ref}",
    )


def _process(name: str) -> ModelRuntimeProcessPolicy:
    return ModelRuntimeProcessPolicy.model_validate(
        {
            "runtime_id": f"omn16944-{name}",
            "runtime_address": f"runtime://test/omn16944/{name}",
            "capabilities": ["compute"],
            "bifrost_verify_endpoints": False,
            "omnimemory_enabled": False,
            "replicas": 1,
        }
    )


def _profile() -> ModelRuntimeProfilePolicy:
    return ModelRuntimeProfilePolicy.model_validate(
        {
            "compose_project": "omnibase-infra-test",
            "main_port": 8085,
            "effects_port": 8086,
            "topic_provisioner_max_partitions": 4,
            "boundary_dlq_enabled": False,
            "secret_resolver_config_path": "/app/data/delegation/secret_resolver.yaml",
            "secret_resolver_namespaces": [
                _tenant_credential_namespace().model_dump(mode="json")
            ],
            "services": {
                "main": _process("main").model_dump(mode="json", by_alias=True),
                "effects": _process("effects").model_dump(mode="json", by_alias=True),
                "worker": _process("worker").model_dump(mode="json", by_alias=True),
            },
        }
    )


def test_profile_namespace_reaches_the_rendered_lane_file(tmp_path: Path) -> None:
    profile = _profile()
    assert profile.secret_resolver_namespaces

    # The shape render_runtime_policy_env emits into
    # ONEX_SECRET_RESOLVER_CONFIG_JSON.
    config_json = json.dumps(
        ModelSecretResolverConfig(
            mappings=list(profile.secret_resolver_mappings),
            namespaces=list(profile.secret_resolver_namespaces),
            enable_convention_fallback=False,
        ).model_dump(mode="json", exclude_defaults=True),
        separators=(",", ":"),
        sort_keys=True,
    )
    assert "tenant_inference_credentials" in config_json

    target = tmp_path / "secret_resolver.yaml"
    rendered = render_secret_resolver_config(
        target_path=target,
        environ={"ONEX_SECRET_RESOLVER_CONFIG_JSON": config_json},
    )
    assert rendered == target

    on_disk = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert [rule["namespace"] for rule in on_disk["namespaces"]] == [
        "tenant_inference_credentials"
    ]
    # The rendered lane file must never carry a secret VALUE -- only names,
    # patterns and source paths.
    assert "key_value" not in target.read_text(encoding="utf-8")

    resolver = SecretResolver(config=ModelSecretResolverConfig.model_validate(on_disk))
    assert resolver.list_configured_namespaces() == ["tenant_inference_credentials"]

    # A credential minted AFTER this lane config was rendered -- no manifest
    # edit, no redeploy -- has a source.
    minted_after_render = f"cred_new-tenant_openrouter_{uuid.uuid4().hex}"
    spec = resolver.resolve_namespace_source(minted_after_render)
    assert spec is not None
    assert spec.source_type == "infisical"
    assert spec.source_path == minted_after_render

    # ...and an unrelated name is still unresolvable on this fallback-off lane.
    assert resolver.get_source_info("llm.anthropic.api_key") is None


def test_a_lane_declaring_namespaces_must_name_a_config_path() -> None:
    with pytest.raises(ValueError, match="config path"):
        ModelRuntimeProfilePolicy.model_validate(
            {
                "compose_project": "omnibase-infra-test",
                "main_port": 8085,
                "effects_port": 8086,
                "topic_provisioner_max_partitions": 4,
                "boundary_dlq_enabled": False,
                "secret_resolver_config_path": "",
                "secret_resolver_namespaces": [
                    _tenant_credential_namespace().model_dump(mode="json")
                ],
                "services": {
                    "main": _process("main").model_dump(mode="json", by_alias=True),
                    "effects": _process("effects").model_dump(
                        mode="json", by_alias=True
                    ),
                    "worker": _process("worker").model_dump(mode="json", by_alias=True),
                },
            }
        )
