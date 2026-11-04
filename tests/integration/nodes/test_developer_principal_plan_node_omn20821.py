# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20821: the developer principal plan node runs from its contract on disk.

The handler and models are resolved from ``contract.yaml``'s ``handler_routing``
and ``input_model``, not imported by hand, and run over the committed lab and local
declarations. The plan matches the lab issuer's own name, digest and grant list.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from omnibase_infra.models.developer_principal import (
    ModelDeveloperPrincipalPlan,
)
from scripts import lab_principal_issuer as lpi

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_developer_principal_plan_compute"
    / "contract.yaml"
)
LAB = REPO_ROOT / "deploy" / "lab"
LOGIN = "dev.person@omninode.ai"


def _load(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    "declaration",
    [
        LAB / "developer-principal-grants.yaml",
        REPO_ROOT / "deploy/local/developer-principal-grants.yaml",
    ],
    ids=["lab", "local"],
)
def test_contract_routed_handler_plans_what_the_lab_issuer_would_issue(
    declaration: Path,
    tmp_path: Path,
) -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    (route,) = contract["handler_routing"]["handlers"]
    assert route["operation"] == "developer_principal.plan"
    handler_class = _load(route["handler"]["module"], route["handler"]["name"])
    request_model = _load(
        contract["input_model"]["module"], contract["input_model"]["name"]
    )
    output_model = _load(
        contract["output_model"]["module"], contract["output_model"]["name"]
    )
    assert isinstance(request_model, type) and issubclass(request_model, BaseModel)
    assert isinstance(handler_class, type)
    request = request_model.model_validate(
        {
            "declaration_yaml": declaration.read_text(),
            "derived_documents": {
                p.name: p.read_text()
                for p in declaration.parent.glob("*-derived-grants.json")
            },
            "login": LOGIN,
            "device": "mac",
        }
    )
    plan = handler_class().handle(request)
    assert output_model is ModelDeveloperPrincipalPlan
    assert isinstance(plan, ModelDeveloperPrincipalPlan)

    base_declaration = declaration
    base_plan = plan
    if declaration.parent.name == "local":
        # The lab issuer is intentionally dev-only for machine additions. Keep
        # its legacy base comparison while checking the shipped local union below.
        base_document = yaml.safe_load(declaration.read_text())
        base_document.pop("machine_grants_from", None)
        base_declaration = tmp_path / "developer-principal-grants.yaml"
        base_declaration.write_text(yaml.safe_dump(base_document))
        base_plan = handler_class().handle(
            request_model.model_validate(
                {
                    "declaration_yaml": base_declaration.read_text(),
                    "derived_documents": {},
                    "login": LOGIN,
                    "device": "mac",
                }
            )
        )
    issued = lpi.load_declaration(base_declaration)
    principal = lpi.principal_name(issued, LOGIN, "mac")
    assert (base_plan.decision, base_plan.principal, base_plan.declaration_digest) == (
        "create",
        principal,
        issued.digest,
    )
    expected_base = {
        (g.resource, g.for_principal(principal).name, g.pattern, g.operations)
        for g in issued.grants_for(principal)
    }
    assert {
        (g.resource, g.name, g.pattern, g.operations) for g in base_plan.grants
    } == expected_base
    expected = expected_base
    if declaration.parent.name == "local":
        # Independent approved oracle: the lab bindings, never generated local JSON.
        bindings = yaml.safe_load((LAB / "machine-principal-bindings.yaml").read_text())
        expected = expected_base | {
            ("topic", binding["physical_topic"], "literal", ("describe", "write"))
            for binding in bindings["bindings"]
        }
    assert (plan.decision, plan.principal) == ("create", principal)
    assert {
        (g.resource, g.name, g.pattern, g.operations) for g in plan.grants
    } == expected
