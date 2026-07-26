# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for ``tests/unit/cli/``.

Neutralizes the ``onex node``/``onex run`` omnimarket pre-flight drift guard
(OMN-14560, mirroring OMN-14531's ``onex skill`` fix) so CLI-wiring tests
across this directory stay hermetic regardless of the ambient developer
shell's ``$OMNI_HOME`` and whether this test venv happens to have omnimarket
co-installed. Three files invoke ``run_node_by_name`` (directly or via the
``onex run`` alias): ``test_cli_node.py``, ``test_cli_node_receipt.py``, and
``test_onex_run.py`` -- a per-file fixture would have to be duplicated three
times, so it lives here instead.

The guard's own behavior (fail-open vs. raise) is tested directly in
``test_omnimarket_drift_guard.py``, which imports ``check_omnimarket_drift``
from ``omnimarket_drift_guard`` and is unaffected by this fixture. CLI-wiring
proofs that need the REAL guard restore it explicitly within their own test
scope (see ``test_drift_guard_fires_before_unknown_node_lookup`` in
``test_cli_node.py``).
"""

from __future__ import annotations

import pytest

from omnibase_infra.cli import cli_node


@pytest.fixture(autouse=True)
def _no_omnimarket_drift_guard_cli_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_node, "check_omnimarket_drift", lambda **_: None)
