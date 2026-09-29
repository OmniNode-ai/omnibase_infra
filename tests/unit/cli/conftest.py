# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for ``tests/unit/cli/``.

Restores the complete process environment after every CLI test because CLI tests
exercise process-wide runtime configuration in-process (OMN-15572).

Neutralizes the ``onex node`` omnimarket pre-flight drift guard
(OMN-14560, mirroring OMN-14531's ``onex skill`` fix) so CLI-wiring tests
across this directory stay hermetic regardless of the ambient developer
shell's ``$OMNI_HOME`` and whether this test venv happens to have omnimarket
co-installed. Two files invoke ``run_node_by_name``: ``test_cli_node.py`` and
``test_cli_node_receipt.py`` -- a per-file fixture would have to be duplicated
twice, so it lives here instead.

The guard's own behavior (fail-open vs. raise) is tested directly in
``test_omnimarket_drift_guard.py``, which imports ``check_omnimarket_drift``
from ``omnimarket_drift_guard`` and is unaffected by this fixture. CLI-wiring
proofs that need the REAL guard restore it explicitly within their own test
scope (see ``test_drift_guard_fires_before_unknown_node_lookup`` in
``test_cli_node.py``).
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

from omnibase_infra.cli import cli_delegate, cli_node
from tests.helpers.cli_registry_stand_in import (
    install_stand_in_registry,
    wiring_authority,
)
from tests.helpers.cli_registry_stand_in.node_delegate_stand_in.model_stand_in_delegate_request import (
    ModelStandInDelegateRequest,
)


@pytest.fixture(autouse=True)
def _restore_process_environment_after_cli_test() -> Generator[None, None, None]:
    """Keep process-wide CLI configuration changes inside each CLI test."""
    environment_before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(environment_before)


@pytest.fixture(autouse=True)
def _no_omnimarket_drift_guard_cli_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_node, "check_omnimarket_drift", lambda **_: None)


@pytest.fixture(autouse=True)
def stand_in_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give ``onex delegate`` stand-in registry entries (OMN-19407).

    The CLI reads its task-class vocabulary, selection and budgets from the
    ``onex.contracts:task_class_authority`` registry entry, and its request
    vocabularies from the delegate contract's input model. omnibase_infra's
    test environment has no omnimarket, so both are stand-ins: the authority
    is ``wiring_authority()``, and the input model is the stand-in node's.
    Neither is compared to a production list anywhere; the registry tests in
    ``test_cli_registry_vocabulary.py`` assert the CLI offers whatever the
    stand-in declares.
    """
    install_stand_in_registry(monkeypatch, wiring_authority())
    monkeypatch.setattr(
        cli_delegate, "_delegate_request_model", lambda: ModelStandInDelegateRequest
    )
