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
import os
from pathlib import Path

import pytest
import yaml

from omnibase_infra.cli.task_class_selection import (
    DEFAULT_EXECUTION_BUDGET,
    load_selectable_task_classes,
    resolve_task_class_execution_budget,
)

pytestmark = pytest.mark.integration


#: Where the contract sits relative to the omnimarket package root.
_CONTRACT_RELATIVE = Path("configs") / "task_class_contracts.v1.yaml"


def _shipped_contract_or_skip() -> Path:
    """Resolve the contract omnimarket actually ships, however it is present.

    Three sources, in the order that makes this test RUN rather than skip.
    The installed package is the truest, but this repo's venv-purity gate
    refuses to run with omnimarket installed, so it is never available in CI.
    The pinned checkout at `.proof-dependencies/omnimarket` is what the
    Application Database Domain Enforcement job provides, and is the same
    source the OMN-18863 modules use. The sibling SOURCE checkout is resolved
    by the same order node-migration-sync and the skill-catalog check use. Either way
    the bytes are omnimarket's, not a fixture this repo wrote about itself,
    which is the whole point of the module.
    """
    spec = importlib.util.find_spec("omnimarket")
    if spec is not None and spec.origin:
        installed = Path(spec.origin).resolve().parent / _CONTRACT_RELATIVE
        if installed.is_file():
            return installed

    repo_root = Path(__file__).resolve().parents[3]
    pinned = repo_root / ".proof-dependencies" / "omnimarket" / "src" / "omnimarket"
    if (pinned / _CONTRACT_RELATIVE).is_file():
        return pinned / _CONTRACT_RELATIVE

    explicit = os.environ.get("OMNIMARKET_SRC")
    if explicit and (Path(explicit) / "pyproject.toml").is_file():
        candidate = Path(explicit) / "src" / "omnimarket" / _CONTRACT_RELATIVE
        if candidate.is_file():
            return candidate

    omni_home = os.environ.get("OMNI_HOME")
    if omni_home:
        candidate = (
            Path(omni_home) / "omnimarket" / "src" / "omnimarket" / _CONTRACT_RELATIVE
        )
        if candidate.is_file():
            return candidate

    pytest.skip(
        "omnimarket is neither installed nor resolvable as a source tree "
        "(set OMNIMARKET_SRC or OMNI_HOME); CI wires the sibling checkout, so "
        "this assertion did NOT run and is not evidence"
    )


@pytest.mark.integration
def test_every_exposed_class_resolves_a_budget_on_the_installed_contract() -> None:
    """The reproduction, against the artifact rather than against a fixture.

    Before this change, this raised for the first class tried, which is what
    refused every delegation on the host.
    """
    contract = _shipped_contract_or_skip()
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
    contract = _shipped_contract_or_skip()
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
