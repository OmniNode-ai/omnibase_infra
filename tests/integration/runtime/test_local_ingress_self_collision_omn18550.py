# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The shipped contract corpus must not self-collide on a local ingress alias (OMN-18550).

On 2026-09-17 the ``.201`` dev-lane ``omninode-runtime`` container exited 1 every
forty seconds for hours. The killer was ``discover_runtime_local_ingress_routes``
raising on a contract in an INSTALLED package that declared one handler operation
twice. Route discovery runs on every boot, before the runtime can serve anything,
so a single malformed contract anywhere in the active package set stops the whole
runtime host.

That defect was caught by nothing until the container was already looping. These
tests run the real discovery path over the real on-disk contract corpus, so the
same class of defect in THIS repo's contracts fails a pull request instead of a
boot.

The first test alone would be a vacuous pass if discovery silently returned
nothing, so the second is its positive control: the same real entry point, over a
real on-disk package tree carrying an injected duplicate, must still raise -- and
must raise the attributable message rather than naming one contract path twice.

Related Tickets:
    - OMN-18550: the runtime cannot finish starting and says nothing useful.
    - OMN-17888: the omnimarket contract that carried the live duplicate.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_infra.runtime.runtime_local_ingress import (
    discover_runtime_local_ingress_routes,
)

pytestmark = pytest.mark.integration


def test_shipped_contracts_discover_without_a_self_collision() -> None:
    """The real corpus, through the real entry point, must not raise.

    No fixture, no monkeypatch: this imports ``omnibase_infra`` and walks the
    contracts actually shipped in it, which is exactly what the kernel does at
    boot. A contract that declares one operation twice fails here.
    """
    routes = discover_runtime_local_ingress_routes(("omnibase_infra",))

    # Non-vacuous: the corpus is real and non-trivial, so an empty result would
    # mean discovery silently found nothing rather than that nothing collides.
    assert len(routes) > 50, (
        "local ingress discovery returned almost nothing over the real package; "
        "the no-collision assertion above would be vacuous"
    )


def test_a_duplicated_operation_in_a_real_package_tree_still_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control for the test above, and the OMN-18550 message fix.

    Copies a real shipped contract into a package tree on disk, appends a second
    handler entry reusing the first entry's operation with a different input
    model -- the exact shape measured in the lane image -- and drives the same
    public entry point the kernel calls.
    """
    package_root = tmp_path / "fakepkg"
    node_dir = package_root / "nodes" / "node_dupe"
    node_dir.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    contract_path = node_dir / "contract.yaml"
    contract_path.write_text(
        """
name: node_dupe
event_bus:
  subscribe_topics:
    - onex.cmd.dupe.deploy-requested.v1
terminal_event: onex.evt.dupe.deploy-completed.v1
handler_routing:
  handlers:
    - operation: dupe.deploy.publish_monitor
      input_model:
        module: fakepkg.models.model_deploy_publish_command
        name: ModelDeployPublishCommand
    - operation: dupe.deploy.publish_monitor
      input_model:
        module: fakepkg.events.runtime_deployment
        name: ModelDeployRebuildCompleted
""".strip(),
        encoding="utf-8",
    )
    # Prove the tree is real on disk and readable, not an in-memory stand-in.
    assert contract_path.is_file()

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )

    with pytest.raises(ValueError) as excinfo:
        discover_runtime_local_ingress_routes(("fakepkg",))

    message = str(excinfo.value)
    assert "dupe.deploy.publish_monitor" in message
    # The regression this ticket fixes: the path was printed on both sides of
    # "and", describing a collision between two contracts that does not exist.
    assert message.count(str(contract_path)) == 1
    assert "same contract" in message


def test_discovery_is_unaffected_by_an_unreadable_sibling_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed neighbour is skipped; a self-collision is still fatal.

    Guards the boundary between the two dispositions, so a future change cannot
    quietly downgrade the collision raise into the same "skip and continue" path
    that unreadable contracts take.
    """
    package_root = tmp_path / "fakepkg"
    good_dir = package_root / "nodes" / "node_good"
    good_dir.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (good_dir / "contract.yaml").write_text(
        """
name: node_good
event_bus:
  subscribe_topics:
    - onex.cmd.good.start.v1
handler_routing:
  handlers:
    - operation: good.run
""".strip(),
        encoding="utf-8",
    )

    bad_dir = package_root / "nodes" / "node_unreadable"
    bad_dir.mkdir(parents=True)
    (bad_dir / "contract.yaml").write_text("name: [unterminated", encoding="utf-8")

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )

    routes = discover_runtime_local_ingress_routes(("fakepkg",))

    assert "good.run" in routes
    assert routes["node_good"].command_topic == "onex.cmd.good.start.v1"

    # Now make the good node self-collide and confirm the same corpus raises.
    shutil.rmtree(bad_dir)
    (good_dir / "contract.yaml").write_text(
        """
name: node_good
event_bus:
  subscribe_topics:
    - onex.cmd.good.start.v1
handler_routing:
  handlers:
    - operation: good.run
      input_model:
        module: fakepkg.models.model_a
        name: ModelA
    - operation: good.run
      input_model:
        module: fakepkg.models.model_b
        name: ModelB
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        discover_runtime_local_ingress_routes(("fakepkg",))
