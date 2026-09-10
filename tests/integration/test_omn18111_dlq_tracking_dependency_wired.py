# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18111 -- a contract-declared dependency the runtime never supplies is a
silent no-op, not a wiring failure.

``node_dlq_replay_effect``'s ``contract.yaml`` declares four dependencies:
``consumer``, ``producer``, ``quarantine_producer`` and ``tracking``
("ServiceDlqTracking for dlq_replay_history persistence"). The runtime's
``_build_runtime_handler_dependencies`` supplied the first three and never the
fourth. Because ``tracking`` is ``required: false`` and ``HandlerDlqReplay``
defaults it to ``None``, nothing failed: the handler wired, drained, quarantined
~62,000 records in an 18-minute window on the .201 dev lane, and
``_record()`` returned at its first line every single time.

That is the whole failure mode. There was no error to find, no exception to
read, and no red check anywhere -- only a table that had been empty since the
node shipped, and three separate places in the node's own prose asserting it
could not be.

The second test below is the mechanical form of the rule, and it is
DISCOVERED from the contract rather than written against a list: every
dependency the node declares must appear in the map the resolver reads. A
future dependency added to ``contract.yaml`` and forgotten in the kernel is red
here instead of silent for three months.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import omnibase_infra.nodes.node_dlq_replay_effect as _dlq_replay_pkg
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

pytestmark = pytest.mark.integration

_HANDLER_NAME = "HandlerDlqReplay"
_BOOTSTRAP = "localhost:9092"


class _FakeTrackingService:
    """Shape-compatible stand-in for ``ServiceDlqTracking``."""

    @property
    def is_tracking_enabled(self) -> bool:
        return True

    async def record_replay_attempt(self, record: object) -> None:  # pragma: no cover
        return None


def _contract_dependency_names() -> set[str]:
    contract_path = Path(str(_dlq_replay_pkg.__file__)).parent / "contract.yaml"
    contract: dict[str, Any] = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    return {dependency["name"] for dependency in contract["dependencies"]}


def test_runtime_supplies_the_declared_tracking_dependency() -> None:
    """RED on origin/dev: ``_build_runtime_handler_dependencies`` has no
    ``dlq_tracking`` parameter at all, so this call is a ``TypeError``."""
    dependencies = _build_runtime_handler_dependencies(
        None,
        _BOOTSTRAP,
        dlq_tracking=_FakeTrackingService(),
    )

    assert dependencies is not None
    assert _HANDLER_NAME in dependencies, sorted(dependencies)
    tracking = dependencies[_HANDLER_NAME].get("tracking")
    assert isinstance(tracking, _FakeTrackingService), dependencies[_HANDLER_NAME]


def test_every_contract_declared_dependency_is_wired() -> None:
    """RED on origin/dev: ``tracking`` is declared and never wired.

    Discovered from ``contract.yaml``, so it covers the class rather than the
    one dependency that was missed.
    """
    declared = _contract_dependency_names()
    assert "tracking" in declared, (
        "guard on the guard: this test is vacuous if the contract stops "
        "declaring the dependency it is about"
    )

    dependencies = _build_runtime_handler_dependencies(
        None,
        _BOOTSTRAP,
        dlq_tracking=_FakeTrackingService(),
    )
    assert dependencies is not None
    wired = set(dependencies[_HANDLER_NAME])

    missing = sorted(declared - wired)
    assert not missing, (
        f"{_HANDLER_NAME} declares {sorted(declared)} in contract.yaml but the "
        f"runtime supplies {sorted(wired)}; {missing} would resolve to None and "
        "no-op silently"
    )


def test_no_tracking_service_leaves_the_key_absent_rather_than_none() -> None:
    """A runtime with no database must not plant a ``tracking: None`` entry.

    The resolver reads this map; an explicit ``None`` and an absent key behave
    the same for ``HandlerDlqReplay`` today, but only the absent key states
    honestly that the runtime had nothing to give.
    """
    dependencies = _build_runtime_handler_dependencies(
        None,
        _BOOTSTRAP,
        dlq_tracking=None,
    )
    assert dependencies is not None
    assert "tracking" not in dependencies[_HANDLER_NAME], dependencies[_HANDLER_NAME]
