# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The modules that call a third-party service hold no URL literal (OMN-21623).

Runs core's contracts-and-DI validator over each module that used to spell a
service URL, and over a planted URL as the positive control, then reads the
constants the modules hold back from the declared endpoints.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)
from omnibase_infra.models.third_party_endpoints import load_third_party_endpoints

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra"

_CONSUMERS: tuple[str, ...] = (
    "adapters/project_tracker/linear_graphql_project_tracker_adapter.py",
    "docker/catalog/validator_healthcheck_semantic_probe.py",
    "event_bus/topic_violation_alerter.py",
    "handlers/handler_gmail_api.py",
    "handlers/handler_slack_webhook.py",
    "nodes/node_artifact_reconciliation_orchestrator/handlers/handler_plan_to_pr_comment.py",
    "nodes/node_evidence_autoclose_sweep_effect/handlers/handler_evidence_autoclose_sweep.py",
    "nodes/node_evidence_autoclose_sweep_effect/offer_companion_sources.py",
    "nodes/node_runtime_source_attestor_effect/handlers/handler_source_attestation.py",
    "nodes/node_setup_validate_effect/handlers/handler_service_validate.py",
    "services/post_merge/checks.py",
    "models/third_party_endpoints/model_third_party_endpoints.py",
)


def _rules_reported(path: Path) -> list[str]:
    result = ValidatorContractsAndDi().validate([path])
    return [issue.rule_name or "" for issue in result.issues]


def test_the_check_reports_a_planted_url(tmp_path: Path) -> None:
    planted = tmp_path / "planted_module.py"
    planted.write_text(
        'SERVICE_URL = "https://service.example.invalid/api"\n', encoding="utf-8"
    )
    assert "literal_url" in _rules_reported(planted)


@pytest.mark.parametrize("module", _CONSUMERS)
def test_consumer_holds_no_url_literal(module: str) -> None:
    assert "literal_url" not in _rules_reported(_SRC / module)


def test_consumer_defaults_read_the_declared_endpoints() -> None:
    from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
        AdapterLinearGraphQLProjectTracker,
    )
    from omnibase_infra.handlers import handler_gmail_api, handler_slack_webhook

    endpoints = load_third_party_endpoints()
    default_endpoint = (
        inspect.signature(AdapterLinearGraphQLProjectTracker.__init__)
        .parameters["endpoint"]
        .default
    )
    assert default_endpoint == endpoints.linear_graphql_url
    assert endpoints.slack_post_message_url == handler_slack_webhook._SLACK_WEB_API_URL
    assert endpoints.gmail_api_base_url == handler_gmail_api._GMAIL_API_BASE


def test_declared_values_are_the_expected_endpoints() -> None:
    """Pin the declared file to independent literals, so a changed value is seen."""
    endpoints = load_third_party_endpoints()
    assert endpoints.linear_graphql_url == "https://api.linear.app/graphql"
    assert endpoints.linear_oauth_token_url == "https://api.linear.app/oauth/token"
    assert endpoints.slack_post_message_url == "https://slack.com/api/chat.postMessage"
    assert endpoints.gmail_oauth_token_url == "https://oauth2.googleapis.com/token"
    assert (
        endpoints.gmail_api_base_url == "https://gmail.googleapis.com/gmail/v1/users/me"
    )
    assert endpoints.github_api_base_url == "https://api.github.com"
    assert (
        endpoints.runtime_source_repo_url
        == "https://github.com/OmniNode-ai/omnibase_infra.git"
    )


def test_module_constants_equal_the_declared_values() -> None:
    from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.handlers import (
        handler_plan_to_pr_comment,
    )
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
        handler_evidence_autoclose_sweep,
    )
    from omnibase_infra.nodes.node_runtime_source_attestor_effect.handlers import (
        handler_source_attestation,
    )

    assert handler_plan_to_pr_comment._GITHUB_API_BASE == "https://api.github.com"
    assert (
        handler_evidence_autoclose_sweep._LINEAR_OAUTH_TOKEN_URL
        == "https://api.linear.app/oauth/token"
    )
    assert (
        handler_source_attestation._DEFAULT_REPO_URL
        == "https://github.com/OmniNode-ai/omnibase_infra.git"
    )


def test_endpoints_file_is_packaged_beside_the_model() -> None:
    declared = _SRC / "models" / "third_party_endpoints" / "third_party_endpoints.yaml"
    assert declared.is_file()
    assert "linear_graphql_url" in declared.read_text(encoding="utf-8")
