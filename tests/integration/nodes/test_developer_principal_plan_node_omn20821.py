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
        REPO_ROOT / "deploy" / "local" / "developer-principal-grants.yaml",
    ],
    ids=["lab", "local"],
)
def test_contract_routed_handler_plans_what_the_lab_issuer_would_issue(
    declaration: Path,
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
                p.name: p.read_text() for p in LAB.glob("*-derived-grants.json")
            },
            "login": LOGIN,
            "device": "mac",
        }
    )
    plan = handler_class().handle(request)

    assert output_model is ModelDeveloperPrincipalPlan
    assert isinstance(plan, ModelDeveloperPrincipalPlan)
    issued = lpi.load_declaration(declaration)
    principal = lpi.principal_name(issued, LOGIN, "mac")
    assert (plan.decision, plan.principal, plan.declaration_digest) == (
        "create",
        principal,
        issued.digest,
    )
    assert {(g.resource, g.name, g.pattern, g.operations) for g in plan.grants} == {
        (g.resource, g.for_principal(principal).name, g.pattern, g.operations)
        for g in issued.grants_for(principal)
    }
