# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The literal ``SLACK_BOT_TOKEN`` ref resolves with zero warnings [OMN-16921].

Residual from OMN-16778 (epic OMN-16776): every successful Slack delivery
logged

    [WARNING] secret_resolver: Secret resolution failed (configuration issue): SLACK_BOT_TOKEN

right before succeeding.

``node_slack_publish_effect._resolve_transport()``
(``omnimarket/src/omnimarket/nodes/node_slack_publish_effect/handlers/
handler_slack_publish_effect.py:457``) does::

    slack_ref = contract_secret_ref(_CONTRACT_PATH, "SLACK_BOT_TOKEN")
    secret = await resolve_api_key_async(slack_ref, env_var_fallback=slack_ref)

``slack_ref`` is the LITERAL string ``"SLACK_BOT_TOKEN"`` -- the dict key
under the node's own ``contract.yaml`` ``secrets:`` block, not a dotted
logical ref. On a deployed lane that first-choice lookup goes through this
repo's ``SecretResolver``, built from ``secret_resolver_mappings`` in
``contracts/services/runtime_policy.contract.yaml`` with
``enable_convention_fallback: false`` (``scripts/render_runtime_policy_env.py``
hardcodes this -- the strict posture is deliberate, OMN-14951 gap 1).

The dev/stability-test/judge profiles mapped only the DOTTED logical name
``slack.bot_token`` -> env ``SLACK_BOT_TOKEN``. ``"SLACK_BOT_TOKEN"`` (what
the call site actually passes) was never a key in ``_mappings``, so
``_get_source_spec`` fell through required_secrets (not required) and
convention fallback (disabled) to ``None``, and ``_record_resolution_failure``
logged the ``no_mapping`` warning. Delivery still succeeded only because
``resolve_api_key_async``'s SECOND lookup -- ``env_var_fallback=slack_ref`` --
did a raw ``os.environ.get("SLACK_BOT_TOKEN")``, which the container does
export. Any future Slack-publish caller that does not thread
``env_var_fallback=ref`` fails closed with the token sitting right there.

Fix: add a mapping entry for the LITERAL ref name ``SLACK_BOT_TOKEN`` (not a
rename of the existing dotted mapping -- additive) to each of the three
profiles, following the identical pattern OMN-16891 (omnibase_infra#2993,
``c5c827e6``) used for ``llm.openrouter.api_key`` / ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_POLICY = _REPO_ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
_RUNTIME_POLICY_ENV = _REPO_ROOT / "docker" / "runtime-policy.env"

# The literal ref name the slack-publish call site actually passes --
# contract_secret_ref(_CONTRACT_PATH, "SLACK_BOT_TOKEN") returns this dict key
# verbatim. NOT the dotted convention-style name.
_SLACK_REF = "SLACK_BOT_TOKEN"
_SLACK_ENV_VAR = "SLACK_BOT_TOKEN"

_DEPLOYED_PROFILES = ("dev", "stability-test", "judge")


def _load_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def _mappings_for_profile(profile_name: str) -> list[dict[str, Any]]:
    lanes: dict[str, Any] = _load_yaml(_RUNTIME_POLICY)["profiles"]
    profile = lanes[profile_name]
    mappings: list[dict[str, Any]] = profile.get("secret_resolver_mappings") or []
    return mappings


def _resolver_for_profile(profile_name: str) -> SecretResolver:
    """Build the exact resolver posture the deployed lane uses.

    Mirrors ``scripts/render_runtime_policy_env.py``'s
    ``ModelSecretResolverConfig(mappings=..., enable_convention_fallback=False)``
    construction -- the strict, deployed-lane shape, not a relaxed test double.
    """
    raw_mappings = _mappings_for_profile(profile_name)
    config = ModelSecretResolverConfig.model_validate(
        {"mappings": raw_mappings, "enable_convention_fallback": False}
    )
    return SecretResolver(config=config)


@pytest.mark.unit
class TestSlackBotTokenLiteralRefMapping:
    """The literal ``SLACK_BOT_TOKEN`` ref must resolve directly, no fallback."""

    @pytest.mark.parametrize("profile_name", _DEPLOYED_PROFILES)
    def test_contract_declares_a_mapping_for_the_literal_ref(
        self, profile_name: str
    ) -> None:
        """RED before the fix: no mapping names the literal call-site ref.

        This is the data-level assertion the fix must satisfy -- an explicit
        ``secret_resolver_mappings`` entry keyed by the LITERAL string
        ``"SLACK_BOT_TOKEN"`` (not the pre-existing dotted ``slack.bot_token``
        entry, which is a different dict key and does not help this lookup).
        """
        mappings = _mappings_for_profile(profile_name)
        matches = [m for m in mappings if m.get("logical_name") == _SLACK_REF]

        assert matches, (
            f"profile {profile_name!r} declares no secret_resolver_mappings "
            f"entry for the literal ref {_SLACK_REF!r} -- the exact string "
            "node_slack_publish_effect._resolve_transport() passes to "
            "resolve_api_key_async(). The pre-existing 'slack.bot_token' "
            "dotted mapping is a DIFFERENT dict key and does not resolve "
            "this lookup."
        )
        assert matches[0]["source"]["source_type"] == "env"
        assert matches[0]["source"]["source_path"] == _SLACK_ENV_VAR

    @pytest.mark.parametrize("profile_name", _DEPLOYED_PROFILES)
    def test_resolution_succeeds_with_zero_warnings_no_env_var_fallback_needed(
        self, profile_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falsifiable AC: resolving the literal ref must not touch no_mapping.

        Builds the resolver from the profile's OWN declared mappings under the
        real deployed posture (``enable_convention_fallback=False``) and
        resolves ``SLACK_BOT_TOKEN`` directly through the mapping -- with NO
        ``env_var_fallback`` rescue, unlike the omnimarket#2197 call site.
        A ``no_mapping`` failure count above zero reproduces the defect.
        """
        monkeypatch.setenv(_SLACK_ENV_VAR, "xoxb-test-token-value")
        resolver = _resolver_for_profile(profile_name)

        secret = resolver.get_secret(_SLACK_REF)

        assert secret is not None, (
            f"profile {profile_name!r}: SecretResolver.get_secret({_SLACK_REF!r}) "
            "returned None -- the literal ref is unmapped under the strict "
            "(enable_convention_fallback=False) deployed posture."
        )
        assert secret.get_secret_value() == "xoxb-test-token-value"

        metrics = resolver.get_resolution_metrics()
        no_mapping_failures = metrics.failure_counts.get("unknown", 0)
        assert no_mapping_failures == 0, (
            f"profile {profile_name!r}: resolving {_SLACK_REF!r} recorded "
            f"{no_mapping_failures} 'unknown'-source (no_mapping) failure(s) -- "
            "this is exactly the resolution-failure warning this ticket "
            "eliminates, even though env_var_fallback later rescues delivery."
        )

    def test_rendered_runtime_policy_env_carries_the_mapping(self) -> None:
        """``docker/runtime-policy.env`` must stay in sync with the contract.

        ``tests/ci/test_runtime_policy_contract.py`` enforces byte-for-byte
        sync via the renderer; this asserts the SEMANTIC content landed --
        catches a forgotten ``scripts/render_runtime_policy_env.py`` re-run.
        """
        text = _RUNTIME_POLICY_ENV.read_text(encoding="utf-8")
        missing: list[str] = []

        prefix_by_profile = {
            "dev": "DEV",
            "stability-test": "STABILITY_TEST",
            "judge": "JUDGE",
        }

        for profile_name, prefix in prefix_by_profile.items():
            found = False
            for line in text.splitlines():
                var_name, sep, raw = line.partition("=")
                if not sep or "_SECRET_RESOLVER_CONFIG_JSON" not in var_name:
                    continue
                if not var_name.startswith(prefix):
                    continue
                raw = raw.strip()
                if not (raw.startswith("'") and raw.endswith("'")):
                    continue
                blob = json.loads(raw[1:-1])
                for mapping in blob.get("mappings", []):
                    if (
                        mapping.get("logical_name") == _SLACK_REF
                        and mapping.get("source", {}).get("source_path")
                        == _SLACK_ENV_VAR
                    ):
                        found = True
            if not found:
                missing.append(profile_name)

        assert not missing, (
            f"profile(s) {missing} have no rendered "
            f"*_SECRET_RESOLVER_CONFIG_JSON blob mapping {_SLACK_REF!r} -- "
            "docker/runtime-policy.env is stale; re-run "
            "scripts/render_runtime_policy_env.py"
        )
