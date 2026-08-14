# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the x-runtime-env anchor in docker-compose.infra.yml (OMN-4800).

Missing vars from x-runtime-env are a silent failure mode — the container starts
but the var is absent. Per CLAUDE.md: "vars reach containers ONLY if listed in
x-runtime-env anchor".

These tests prevent:
1. Required keys missing from the anchor
2. Docker Compose syntax errors in the anchor
3. Services bypassing the anchor with conflicting environment blocks
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPOSE_PATH = (
    Path(__file__).parent.parent.parent / "docker" / "docker-compose.infra.yml"
)

# Keys that MUST be present in x-runtime-env.
# Add new required runtime vars here — this is the canonical required-key list.
REQUIRED_RUNTIME_KEYS: frozenset[str] = frozenset(
    {
        "POSTGRES_PASSWORD",
        "OMNIBASE_INFRA_DB_URL",
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_BROKER_ALLOWLIST",
        "INFISICAL_ADDR",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
        "ONEX_CONTRACTS_DIR",
        "ONEX_STATE_DIR",
        "ONEX_SECRET_RESOLVER_CONFIG_JSON",
        "ONEX_SECRET_RESOLVER_CONFIG_PATH",
        "ONEX_ACTIVE_RUNTIME_PACKAGES",
        "ONEX_MARKETPLACE_SKILLS_ROOT",
        "ONEX_LOG_LEVEL",
        "ONEX_ENVIRONMENT",
        "USE_EVENT_ROUTING",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        # OMN-15529: OnexBot-OCC-Writer App identity for the OCC companion
        # producer. Absent from the anchor, the credential on the host is
        # invisible to the container and the OMN-15362 cutover is a no-op.
        "ONEXBOT_OCC_APP_ID",
        "ONEXBOT_OCC_PRIVATE_KEY",
        "OMNI_OCC_GITHUB_AUTH_MODE",
        "DEPLOY_AGENT_HMAC_SECRET",
        "LLM_GLM_URL",
        "LLM_GLM_MODEL_NAME",
        "LLM_GLM_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST",
        "VALKEY_HOST",
        "VALKEY_PORT",
        # OMN-15645: omnimarket#2000 (OMN-15628) removed the packaged-default
        # fallback for this key in the delegation routing reducer
        # (resolve_required_path_config("DELEGATION_ROUTING_TIERS_PATH"),
        # omnimarket src/omnimarket/nodes/node_delegation_routing_reducer/
        # handlers/handler_delegation_routing.py:392-393). An unbound key now
        # raises ProtocolConfigurationError at first config read instead of
        # silently defaulting — absent from the anchor, this key is a boot
        # landmine, not a missing-optional-feature. This entry is the
        # code-declared-required-key -> compose-anchor-coverage check AC4 of
        # OMN-15645 asks for, scoped to this one key (registry-based, not a
        # generic code scanner — see the OMN-15645 PR body for the deferral of
        # the fully generic mechanism to OMN-14951).
        "DELEGATION_ROUTING_TIERS_PATH",
    }
)


def _load_compose() -> dict:
    """Load and parse the docker-compose.infra.yml file."""
    assert COMPOSE_PATH.exists(), f"Compose file not found: {COMPOSE_PATH}"
    with COMPOSE_PATH.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), (
        f"Compose file did not parse as a YAML mapping: {COMPOSE_PATH}"
    )
    return data


def _get_runtime_env_keys(data: dict) -> set[str]:
    """Extract keys from the x-runtime-env anchor."""
    runtime_env = data.get("x-runtime-env")
    assert runtime_env is not None, (
        "x-runtime-env anchor not found in docker-compose.infra.yml. "
        "This anchor is required for env var passthrough to containers."
    )
    assert isinstance(runtime_env, dict), (
        f"x-runtime-env is not a YAML mapping (got {type(runtime_env).__name__}). "
        "The anchor must be a key-value mapping."
    )
    return set(runtime_env.keys())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRuntimeEnvAnchorContainsRequiredKeys:
    """Test 1: x-runtime-env anchor contains all required keys."""

    @pytest.mark.unit
    def test_anchor_contains_required_keys(self) -> None:
        """Assert all required runtime keys are present in x-runtime-env."""
        data = _load_compose()
        actual_keys = _get_runtime_env_keys(data)

        missing = REQUIRED_RUNTIME_KEYS - actual_keys
        assert not missing, (
            f"x-runtime-env is missing {len(missing)} required key(s):\n"
            + "\n".join(f"  - {k}" for k in sorted(missing))
            + "\n\nAdd missing keys to x-runtime-env in docker/docker-compose.infra.yml. "
            "Format: KEY: ${KEY:-} (or ${KEY:?error} for required keys)."
        )


class TestOccAppIdentityReachesRuntimeServices:
    """OMN-15529: the OCC App credential must reach the container, not just the anchor.

    Asserting anchor membership alone is not the seam — a service only sees a
    var if the anchor is merged into *its* environment. These tests drive the
    resolved ``services.<svc>.environment`` mapping (PyYAML resolves the
    ``!!merge <<: *runtime-env`` keys), which is the same merge Docker Compose
    performs before interpolation.

    Failure mode this closes (OMN-15362 defect D2): the OnexBot-OCC-Writer key
    was provisioned into ``~/.omnibase/.env`` on the runtime host and was still
    invisible to ``node_occ_companion_effect`` / ``OccCompanionEmitter``,
    because ``x-runtime-env`` is an allowlist rather than a pass-through.
    ``OMNI_OCC_GITHUB_AUTH_MODE=app`` would then fail with
    ``GitHubAppCredentialMissingError`` against a dead runtime path.
    """

    OCC_APP_IDENTITY_KEYS: frozenset[str] = frozenset(
        {
            "ONEXBOT_OCC_APP_ID",
            "ONEXBOT_OCC_PRIVATE_KEY",
            "OMNI_OCC_GITHUB_AUTH_MODE",
        }
    )

    # runtime-effects hosts the OCC companion EFFECT — the producer whose
    # GitHub identity this credential changes. The other two runtime services
    # inherit the same anchor and are asserted for consistency.
    OCC_PRODUCER_SERVICE = "runtime-effects"
    RUNTIME_SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")

    @pytest.mark.unit
    def test_occ_app_identity_reaches_occ_producer_service(self) -> None:
        """The OCC producer container's env exposes all three App identity vars."""
        data = _load_compose()
        service = data["services"][self.OCC_PRODUCER_SERVICE]
        environment = service["environment"]
        assert isinstance(environment, dict)

        missing = self.OCC_APP_IDENTITY_KEYS - set(environment)
        assert not missing, (
            f"Service '{self.OCC_PRODUCER_SERVICE}' does not expose "
            f"{sorted(missing)}. The host credential is invisible to the "
            "containerized OCC companion producer — add the name(s) to "
            "x-runtime-env in docker/docker-compose.infra.yml (OMN-15529)."
        )

    @pytest.mark.unit
    def test_occ_app_identity_reaches_every_runtime_service(self) -> None:
        """Every anchor-inheriting runtime service sees the same App identity vars."""
        data = _load_compose()
        for service_name in self.RUNTIME_SERVICES:
            environment = data["services"][service_name]["environment"]
            assert isinstance(environment, dict)
            missing = self.OCC_APP_IDENTITY_KEYS - set(environment)
            assert not missing, (
                f"Service '{service_name}' is missing {sorted(missing)} from its "
                "resolved environment (x-runtime-env merge)."
            )

    @pytest.mark.unit
    def test_occ_app_identity_is_optional_not_fail_closed(self) -> None:
        """The three vars use the optional form, so lanes without the key still render.

        ``${VAR:?...}`` here would abort ``docker compose config`` on every lane
        that has not been provisioned with the App key. Empty is safe on both
        consumers: the auth-mode readers coalesce ``""`` to ``pat``, and
        ``resolve_api_key(..., required=False)`` treats an empty credential as
        absent (raising ``GitHubAppCredentialMissingError`` in app mode rather
        than silently falling back to the shared PAT).
        """
        raw = COMPOSE_PATH.read_text()
        for key in sorted(self.OCC_APP_IDENTITY_KEYS):
            declaration = f"{key}: ${{{key}:-}}"
            assert declaration in raw, (
                f"{key} must be declared as '{declaration}' in x-runtime-env. "
                "A required (:?) or defaulted (:-value) form would either wedge "
                "unprovisioned lanes or pin a default that belongs to the "
                "consuming code, not to compose."
            )


class TestRuntimeEnvAnchorSyntaxValid:
    """Test 2: docker-compose.infra.yml parses without YAML errors."""

    @pytest.mark.unit
    def test_anchor_syntax_valid(self) -> None:
        """Verify docker-compose.infra.yml loads cleanly as valid YAML."""
        # This primarily catches YAML syntax errors in the anchor definition
        data = _load_compose()
        runtime_env = data.get("x-runtime-env")
        assert runtime_env is not None, "x-runtime-env anchor not found"
        assert isinstance(runtime_env, dict), "x-runtime-env must be a YAML mapping"
        assert len(runtime_env) > 0, (
            "x-runtime-env is empty — no vars will be passed through"
        )

    @pytest.mark.unit
    def test_anchor_has_no_null_values(self) -> None:
        """Assert no key in x-runtime-env resolves to a bare null value.

        Null values in the anchor indicate the key was listed without a default,
        which may cause silent failures in containers.
        """
        data = _load_compose()
        runtime_env = data.get("x-runtime-env", {})
        # yaml.safe_load resolves ${VAR:-} to the literal string (not None)
        # but bare `KEY:` (no value) resolves to None
        null_keys = [k for k, v in runtime_env.items() if v is None]
        assert not null_keys, (
            f"x-runtime-env has {len(null_keys)} key(s) with null values: {null_keys}. "
            "Use KEY: ${{KEY:-}} for optional vars or KEY: ${{KEY:?error}} for required vars."
        )

    @pytest.mark.unit
    def test_anchor_uses_onex_marketplace_skill_roots(self) -> None:
        """Assert legacy OmniClaude runtime roots are not exposed to runtime containers."""
        data = _load_compose()
        runtime_env_keys = _get_runtime_env_keys(data)

        assert "ONEX_ACTIVE_RUNTIME_PACKAGES" in runtime_env_keys
        assert "ONEX_MARKETPLACE_SKILLS_ROOT" in runtime_env_keys
        assert "OMNICLAUDE_CONTRACTS_ROOT" not in runtime_env_keys
        assert "OMNICLAUDE_SKILLS_ROOT" not in runtime_env_keys


class TestRuntimeEnvPassthroughNotBypassed:
    """Test 3: No service bypasses x-runtime-env with conflicting environment blocks."""

    @pytest.mark.unit
    def test_env_passthrough_not_bypassed(self) -> None:
        """Assert runtime services use *runtime-env anchor, not standalone environment blocks.

        Services that inherit <<: *runtime-env should not also define a standalone
        environment: block that duplicates anchor keys — the standalone block would
        shadow or conflict with the anchor values.

        This test checks that runtime-profile services (those with <<: *runtime-env)
        do not have environment: blocks with keys that overlap with x-runtime-env.
        """
        data = _load_compose()
        runtime_env_keys = _get_runtime_env_keys(data)
        services = data.get("services", {})

        violations: list[str] = []

        for svc_name, svc_config in services.items():
            if not isinstance(svc_config, dict):
                continue

            env_block = svc_config.get("environment")
            if not env_block:
                continue

            # Only flag services that also inherit *runtime-env
            # (services with bare environment: blocks that don't inherit are fine)
            # Check if this service inherits the runtime-env anchor via <<: *runtime-env
            # After yaml.safe_load, anchor merges are resolved — check deploy labels or
            # known runtime service names instead
            # Heuristic: if env_block shares many keys with runtime_env_keys, likely merged
            if isinstance(env_block, dict):
                svc_env_keys = set(env_block.keys())
                overlap = svc_env_keys & runtime_env_keys
                if (
                    len(overlap) >= 3
                ):  # 3+ overlapping keys = likely duplicate definition
                    violations.append(
                        f"Service '{svc_name}' has {len(overlap)} env keys "
                        f"that overlap with x-runtime-env: {sorted(overlap)[:5]}..."
                    )

        # This is an advisory check — warn rather than fail
        # Full enforcement requires compose-config resolution which needs Docker running
        if violations:
            import warnings

            for v in violations:
                warnings.warn(
                    f"x-runtime-env bypass advisory: {v}",
                    stacklevel=1,
                )


class TestRuntimeServiceKafkaInstanceIds:
    """Production runtime services must not compete in the same Kafka groups."""

    @pytest.mark.unit
    def test_runtime_services_have_unique_kafka_instance_ids(self) -> None:
        data = _load_compose()
        services = data.get("services", {})

        expected = {
            "omninode-runtime": "runtime-main",
            "runtime-effects": "runtime-effects",
            "runtime-worker": "runtime-worker",
        }

        observed: dict[str, str] = {}
        for service_name, expected_instance_id in expected.items():
            service = services[service_name]
            assert isinstance(service, dict)
            environment = service["environment"]
            assert isinstance(environment, dict)
            observed[service_name] = environment["KAFKA_INSTANCE_ID"]
            assert observed[service_name] == expected_instance_id

        assert len(set(observed.values())) == len(observed)
