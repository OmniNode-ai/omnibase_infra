# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The third-party endpoints slice is held by its own ticket contract (OMN-21708).

omnibase_infra#5505 declared the GitHub, Linear, Slack, Gmail, Claude Code schema
and loopback-probe URLs in ``third_party_endpoints.yaml`` and moved eleven modules
onto ``load_third_party_endpoints()``. Its acceptance items were written into the
shared OMN-21623 contract, which five other slices also edit. This test holds the
slice under ``contracts/OMN-21708.yaml``:

* AC1: the modules #5505 moved read the declared file through the model, and the
  contracts_and_di check, re-run over those paths in one call, reports no
  ``literal_url`` finding (16 at #5505's merge base) while a planted URL in the
  same call is reported.
* AC2: the OMN-21708 contract binds both criteria to checks in this file and does
  not point at the shared OMN-21623 contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)
from omnibase_infra.models.third_party_endpoints import (
    ModelThirdPartyEndpoints,
    load_third_party_endpoints,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[4]
_SRC = _ROOT / "src" / "omnibase_infra"
_CONTRACT = _ROOT / "contracts" / "OMN-21708.yaml"
_THIS_TEST = Path(__file__).resolve().relative_to(_ROOT).as_posix()
_DECLARED_FILE = (
    _SRC / "models" / "third_party_endpoints" / "third_party_endpoints.yaml"
)

# The eleven modules omnibase_infra#5505 moved onto the declared endpoints.
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
)


def _imports_loader(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "omnibase_infra.models.third_party_endpoints"
        and any(alias.name == "load_third_party_endpoints" for alias in node.names)
        for node in ast.walk(tree)
    )


def test_ac1_consumers_read_the_declared_file_through_the_model() -> None:
    document = yaml.safe_load(_DECLARED_FILE.read_text(encoding="utf-8"))
    assert load_third_party_endpoints() == ModelThirdPartyEndpoints.model_validate(
        document["endpoints"]
    )
    missing = [m for m in _CONSUMERS if not _imports_loader(_SRC / m)]
    assert missing == [], missing


def test_ac1_check_reports_no_literal_url_over_the_slice_paths(
    tmp_path: Path,
) -> None:
    planted = tmp_path / "planted_module.py"
    planted.write_text(
        'SERVICE_URL = "https://service.example.invalid/api"\n', encoding="utf-8"
    )
    paths = [_SRC / m for m in _CONSUMERS]
    paths.append(_SRC / "models" / "third_party_endpoints")
    paths.append(planted)

    result = ValidatorContractsAndDi().validate(paths)
    literal_url_files = sorted(
        Path(str(issue.file_path)).name
        for issue in result.issues
        if issue.rule_name == "literal_url"
    )
    # Positive control: the planted module was scanned in the same call.
    assert literal_url_files == ["planted_module.py"], literal_url_files


def _contract() -> dict[str, object]:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_ac2_ticket_contract_owns_the_slice_acceptance() -> None:
    contract = _contract()
    assert contract["ticket_id"] == "OMN-21708"

    requirements = contract["requirements"]
    assert isinstance(requirements, list)
    declared = {
        ac["id"] for requirement in requirements for ac in requirement["acceptance"]
    }
    assert declared == {"AC1", "AC2"}

    evidence = contract["dod_evidence"]
    assert isinstance(evidence, list)
    bound = {ac for item in evidence for ac in item["binds_ac"]}
    assert bound == declared

    check_values = [
        check["check_value"] for item in evidence for check in item["checks"]
    ]
    assert check_values
    assert all(_THIS_TEST in value for value in check_values), check_values
    assert not any("OMN-21623" in value for value in check_values), check_values
