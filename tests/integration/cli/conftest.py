# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Stand-in registry entries for every ``onex`` CLI integration test (OMN-19407).

``onex delegate`` reads its task-class vocabulary, selection, fallback and
budgets from the ``onex.contracts:task_class_authority`` registry entry, and
its request vocabularies from the delegate contract's input model, whenever it
renders help or runs. This repo's test environment has no omnimarket, so every
test here gets the stand-ins from ``tests.helpers.cli_registry_stand_in``. A
test that needs a different authority installs its own on top.
"""

from __future__ import annotations

import pytest

from tests.helpers.cli_registry_stand_in import use_stand_in_registry


@pytest.fixture(autouse=True)
def stand_in_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    use_stand_in_registry(monkeypatch)
