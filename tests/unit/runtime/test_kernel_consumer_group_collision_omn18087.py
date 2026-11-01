# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A node-bound process must not join the kernel consumer groups of the main runtime.

``projection-claude-hook-events-writer`` runs the same kernel as
``omninode-runtime`` under ``KAFKA_INSTANCE_ID=runtime-main`` (so the hook-events
group keeps its committed offsets). The kernel's own consumers (contract
registry, plugins, savings, runtime-error triage) derive their group from
``config.name``, which both processes read from the same ``runtime_config.yaml``.
Both therefore joined ``<env>.<name>.<node>.consume.<ver>.__i.runtime-main.__t.<topic>``:
two members on a one-partition topic, one of them with an empty assignment, and
``/ready`` 503 whenever the writer won the join (OMN-18087, 18 of 192 groups
measured on the dev lane on 2026-11-01).

The kernel's consumer service name now comes from the node selection, so a process
bound to a contract by ``ONEX_RUNTIME_NODE_SELECTION`` owns its own kernel groups,
while the selected contract's own group (``package_name`` / contract name) is not
built from it and keeps ``runtime-main``.
"""

import ast
from pathlib import Path

import pytest

from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ENV_RUNTIME_NODE_SELECTION,
    ModelRuntimeNodeSelection,
)
from omnibase_infra.utils import apply_instance_discriminator, compute_consumer_group_id

pytestmark = [pytest.mark.unit]

KERNEL = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "service_kernel.py"
)

CONFIG_NAME = "runtime_config"
INSTANCE_ID = "runtime-main"
WRITER_CONTRACT = "projection_claude_hook_events"
# The node names the kernel builds its own consumer identities for.
KERNEL_NODES = (
    "contract-registry",
    "node_savings_estimation_compute",
    "runtime-error-triage",
    CONFIG_NAME,
)
TOPIC_SUFFIX = ".__t.onex.evt.platform.contract-registered.v1"


def _kernel_group(selection: ModelRuntimeNodeSelection, node_name: str) -> str:
    """The effective group a kernel consumer of ``node_name`` joins on one topic."""
    identity = selection.kernel_identity("local", CONFIG_NAME, node_name, "1.0.0")
    base = compute_consumer_group_id(identity, EnumConsumerGroupPurpose.CONSUME)
    return apply_instance_discriminator(base, INSTANCE_ID) + TOPIC_SUFFIX


def _selection(raw: str) -> ModelRuntimeNodeSelection:
    return ModelRuntimeNodeSelection.from_env({ENV_RUNTIME_NODE_SELECTION: raw})


def test_writer_and_main_share_no_kernel_group() -> None:
    main = _selection("")
    writer = _selection(WRITER_CONTRACT)
    for node_name in KERNEL_NODES:
        assert _kernel_group(main, node_name) != _kernel_group(writer, node_name), (
            f"{node_name}: main runtime and writer join the same group, so one "
            "member of the one-partition topic holds an empty assignment"
        )


def test_unselected_process_keeps_its_existing_kernel_groups() -> None:
    """No selection changes nothing: the main runtime's 169 groups keep their offsets."""
    assert _selection("").consumer_service(CONFIG_NAME) == CONFIG_NAME
    assert _selection("").excluded == ()
    excluding = ModelRuntimeNodeSelection.from_env(
        {"ONEX_RUNTIME_NODE_EXCLUSION": WRITER_CONTRACT}
    )
    assert excluding.consumer_service(CONFIG_NAME) == CONFIG_NAME


def test_two_writers_of_different_contracts_do_not_collide() -> None:
    first = _selection("projection_claude_hook_events")
    second = _selection("projection_other")
    assert first.consumer_service(CONFIG_NAME) != second.consumer_service(CONFIG_NAME)


def test_kernel_identity_is_the_old_identity_when_nothing_is_selected() -> None:
    identity = _selection("").kernel_identity(
        "local", CONFIG_NAME, "contract-registry", None
    )
    assert identity == ModelNodeIdentity(
        env="local", service=CONFIG_NAME, node_name="contract-registry", version="v1"
    )
    assert _selection("").kernel_identity("local", None, "x", "1.0.0").service == (
        "onex-kernel"
    )


def test_kernel_builds_every_consumer_identity_through_the_selection() -> None:
    """No kernel consumer identity is built straight from ``config.name``."""
    tree = ast.parse(KERNEL.read_text(encoding="utf-8"))
    direct = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelNodeIdentity"
    ]
    via_selection = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "kernel_identity"
    ]
    assert direct == [], (
        "kernel consumer identities built outside the node selection collide with "
        f"the main runtime's groups when the instance id is shared: {direct}"
    )
    assert len(via_selection) == 4, "contract registry, plugins, savings, triage"


def test_kernel_hands_the_host_process_the_selection_scoped_service_name() -> None:
    """RuntimeHostProcess derives its subscription groups from ``service_name``."""
    source = KERNEL.read_text(encoding="utf-8")
    assert 'runtime_config_dict["service_name"] = config.name' not in source
    assert (
        'runtime_config_dict["service_name"] = node_selection.consumer_service('
        in source
    )
