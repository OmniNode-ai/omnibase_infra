# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The INSTALLED omnimarket contract resolves a budget rather than refusing (OMN-18924).

WHY THIS IS AN INTEGRATION TEST AND THE UNIT MODULE IS NOT ENOUGH. The unit
module next door proves the rule against contracts it writes itself, so it can
only ever prove what this repo believes the shipped contract looks like. The
outage was precisely a disagreement between that belief and the artifact: this
repo's fixtures had gained an ``execution_budgets`` map while the omnimarket
package on the host had not, so every fixture-backed assertion passed while
every real delegation was refused before dispatch.

This module reads the contract out of the ACTUALLY INSTALLED omnimarket
package, the same file the CLI resolves at runtime, and asserts a budget comes
back for every class the contract gateway-exposes. It is the only assertion in
either module that could have gone red on 2026-09-20.

SKIPS rather than passes when omnimarket is not importable. omnibase_infra's
own unit environment deliberately has no omnimarket, and a skip that announces
itself is honest where a vacuous pass is not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from omnibase_infra.cli.task_class_selection import (
    DEFAULT_EXECUTION_BUDGET,
    load_selectable_task_classes,
    resolve_task_class_execution_budget,
)

pytestmark = pytest.mark.integration


def _installed_contract_or_skip() -> Path:
    spec = importlib.util.find_spec("omnimarket")
    if spec is None or not spec.origin:
        pytest.skip(
            "omnimarket is not importable here, so the INSTALLED contract cannot "
            "be read; this assertion did NOT run and is not evidence"
        )
    path = (
        Path(spec.origin).resolve().parent / "configs" / "task_class_contracts.v1.yaml"
    )
    if not path.is_file():
        pytest.skip(f"the installed omnimarket ships no contract at {path}")
    return path


@pytest.mark.integration
def test_every_exposed_class_resolves_a_budget_on_the_installed_contract() -> None:
    """The reproduction, against the artifact rather than against a fixture.

    Before this change, this raised for the first class tried, which is what
    refused every delegation on the host.
    """
    contract = _installed_contract_or_skip()
    exposed = [entry.name for entry in load_selectable_task_classes(contract)]
    assert exposed, "the installed contract exposes no public class"
    for task_type in exposed:
        budget = resolve_task_class_execution_budget(contract, task_type=task_type)
        assert budget.task_class_timeout_ceiling_seconds > 0, task_type
        assert budget.terminal_delivery_margin_seconds > 0, task_type


@pytest.mark.integration
def test_the_resolution_agrees_with_what_the_installed_contract_declares() -> None:
    """Positive control on the assertion above, so it cannot pass vacuously.

    A budget coming back proves nothing on its own: it would also come back if
    the default silently shadowed a declared map. This pins the two cases apart
    against whichever contract happens to be installed, so the module stays
    honest after the producer lands and the map appears.
    """
    contract = _installed_contract_or_skip()
    declared = yaml.safe_load(contract.read_text(encoding="utf-8")).get(
        "execution_budgets"
    )
    exposed = [entry.name for entry in load_selectable_task_classes(contract)]
    for task_type in exposed:
        budget = resolve_task_class_execution_budget(contract, task_type=task_type)
        if isinstance(declared, dict) and isinstance(declared.get(task_type), dict):
            entry = declared[task_type]
            assert (
                budget.task_class_timeout_ceiling_seconds
                == entry["task_class_timeout_ceiling_seconds"]
            ), task_type
        else:
            assert budget == DEFAULT_EXECUTION_BUDGET, task_type
