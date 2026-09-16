# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The routing decision is a CONTRACT, and these tests bind to it (OMN-18412).

``test_runner_route_decision.py`` covers what the handler DECIDES. This module
covers the half that behaviour cannot defend on its own: that the node is
shaped the way the canonical architecture requires, that the thresholds are
declared in the contract rather than anywhere else, and that the values the
contract declares are ones the handler can actually act on.

THE FAILURE MODE THIS EXISTS FOR IS SILENT. A reason string misspelled in the
contract's reversal list does not raise -- it just stops reversing, so a private
repository quietly lands on a hosted runner it may not use, and every test about
placement still passes because they pass the list they expect. Only a test that
reads the contract against the enum catches it.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

NODE_DIR = REPO_ROOT / "src/omnibase_infra/nodes/node_ci_runner_route_compute"
CONTRACT_PATH = NODE_DIR / "contract.yaml"

from omnibase_infra.nodes.node_ci_runner_route_compute.handlers.handler_ci_runner_route import (
    HandlerCIRunnerRoute,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_reason import (
    EnumCIRunnerRouteReason,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_decision import (
    ModelCIRunnerRouteDecision,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_policy import (
    ModelCIRunnerRoutePolicy,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_request import (
    ModelCIRunnerRouteRequest,
)


def _contract() -> dict[str, Any]:
    loaded = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


# --- the node is the archetype the architecture names ----------------------


def test_the_node_is_a_compute_node_with_one_declared_capability() -> None:
    contract = _contract()
    assert contract["node_type"] == "COMPUTE_GENERIC"
    assert [c["name"] for c in contract["capabilities"]] == ["ci.runner.route"]


def test_the_node_class_is_declarative() -> None:
    """A node coordinates; the handler computes.

    Anything beyond ``__init__`` on the node class is logic that escaped the
    handler, which is the one shape the repo invariants forbid outright.
    """
    module = importlib.import_module(
        "omnibase_infra.nodes.node_ci_runner_route_compute.node"
    )
    node_cls = module.NodeCIRunnerRouteCompute
    own = [
        name
        for name, value in vars(node_cls).items()
        if callable(value) and not name.startswith("__")
    ]
    assert own == [], f"node.py carries logic: {own}"


def test_the_contract_routes_the_capability_to_the_handler_that_exists() -> None:
    """A contract may not name a handler nobody can import."""
    routing = _contract()["handler_routing"]
    assert routing["routing_strategy"] == "operation_match"
    (entry,) = routing["handlers"]
    assert entry["operation"] == "ci.runner.route"
    module = importlib.import_module(entry["handler"]["module"])
    assert getattr(module, entry["handler"]["name"]) is HandlerCIRunnerRoute


@pytest.mark.parametrize("key", ["input_model", "output_model"])
def test_the_declared_models_are_importable_and_are_the_ones_used(key: str) -> None:
    declared = _contract()[key]
    module = importlib.import_module(declared["module"])
    model = getattr(module, declared["name"])
    expected = (
        ModelCIRunnerRouteRequest
        if key == "input_model"
        else ModelCIRunnerRouteDecision
    )
    assert model is expected


def test_the_handler_is_the_canonical_definition_b_shape() -> None:
    """``handle(request: ModelX) -> ModelY``, typed payload in and out.

    The pre-def-B envelope signature is not canonical and is rejected by the
    canon-shape ratchet, so this is pinned here rather than discovered there.
    """
    signature = inspect.signature(HandlerCIRunnerRoute.handle)
    params = [p for p in signature.parameters if p != "self"]
    assert params == ["request"]
    hints = HandlerCIRunnerRoute.handle.__annotations__
    assert hints["request"] is ModelCIRunnerRouteRequest or hints["request"] == (
        "ModelCIRunnerRouteRequest"
    )
    assert hints["return"] is ModelCIRunnerRouteDecision or hints["return"] == (
        "ModelCIRunnerRouteDecision"
    )


def _imported_names(path: Path) -> set[str]:
    """Every name this module imports, from the parse rather than the text.

    TEXT MATCHING WOULD FIRE ON ITS OWN PROSE. A handler whose docstring says
    it uses no plugin base contains the word; a test that greps for it fails on
    the sentence asserting the rule -- the exact failure CLAUDE.md rule 15
    records for gates that substring-match their own subject. The parse sees
    imports, not sentences.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


HANDLER_PATH = NODE_DIR / "handlers/handler_ci_runner_route.py"


def test_the_handler_imports_no_envelope_and_subclasses_nothing() -> None:
    """No envelope in the core, and no plugin base anywhere near it.

    A core that references the event envelope is rejected as a non-canonical
    handler shape; a plugin base class is not part of the architecture at all.
    """
    imported = _imported_names(HANDLER_PATH)
    assert not {name for name in imported if "Envelope" in name}
    assert not {name for name in imported if "HandlerOutput" in name}
    assert not {name for name in imported if name.startswith("Plugin")}
    assert HandlerCIRunnerRoute.__bases__ == (object,)


def test_the_handler_performs_no_io() -> None:
    """A compute handler is stateless and deterministic.

    Checked from the parse rather than promised in a docstring: the decision is
    replayed from recorded evidence, which is only true while nothing in it
    reaches the network, the filesystem or a subprocess.
    """
    imported = _imported_names(HANDLER_PATH)
    for forbidden in ("urllib", "requests", "subprocess", "socket", "pathlib", "os"):
        assert forbidden not in imported, f"handler performs I/O via {forbidden}"


# --- the thresholds are DECLARED, and declared only here -------------------


def test_the_contract_config_parses_into_the_typed_policy() -> None:
    policy = ModelCIRunnerRoutePolicy.model_validate(_contract()["config"])
    assert policy.policy_version > 0
    assert policy.private_repo_hosted_placement == "refuse"


@pytest.mark.parametrize("field", sorted(ModelCIRunnerRoutePolicy.model_fields))
def test_every_declared_threshold_is_required(field: str) -> None:
    """No silent defaults (CLAUDE.md rule 8).

    A defaulted threshold is how a routing gate quietly stops gating: the run
    still succeeds, the decision still looks reasonable, and the value nobody
    set is the one deciding where the jobs went.
    """
    config = dict(_contract()["config"])
    del config[field]
    with pytest.raises(ValidationError) as excinfo:
        ModelCIRunnerRoutePolicy.model_validate(config)
    assert field in str(excinfo.value)


def test_the_contract_declares_no_threshold_the_model_ignores() -> None:
    """The opposite direction: a key the model does not know is a typo.

    ``extra="forbid"`` makes this fail rather than silently ignoring a
    misspelled threshold that the author believes is in force.
    """
    ModelCIRunnerRoutePolicy.model_validate(_contract()["config"])


def test_every_reversible_reason_is_a_real_reason() -> None:
    """THE SILENT ONE.

    ``capacity_downgrade_reasons`` decides which hosted verdicts may be
    reversed for a repository that must never run hosted. A misspelling here
    does not raise: it just stops reversing, and a private repository lands on
    a runner class the operator ruling forbids. Nothing else in the suite
    catches it, because every other test passes the list it expects.
    """
    declared = set(_contract()["config"]["capacity_downgrade_reasons"])
    known = {reason.value for reason in EnumCIRunnerRouteReason}
    assert declared <= known, f"not real reasons: {sorted(declared - known)}"


def test_the_reversible_set_excludes_every_trust_reason() -> None:
    """The split IS the safety property, so it is asserted, not described.

    Reversing fork isolation would put untrusted code on the fleet; reversing
    the hosted workflow list would put a registry push on the fleet it is
    isolated from. Neither may ever be bought with a capacity argument.
    """
    declared = set(_contract()["config"]["capacity_downgrade_reasons"])
    for trust_reason in (
        EnumCIRunnerRouteReason.FORK_ISOLATION,
        EnumCIRunnerRouteReason.POLICY_ALLOWLIST,
        EnumCIRunnerRouteReason.SEAM_CEILING_HOSTED,
        EnumCIRunnerRouteReason.NEVER_WIDEN_VIOLATION,
    ):
        assert trust_reason.value not in declared


def test_the_thresholds_live_in_the_contract_and_nowhere_else() -> None:
    """One home. A second copy is a second policy.

    The operator direction is that configuration is contract-driven; this is
    the mechanical form of it. If a threshold reappears in the repository's
    routing config file, this fails and names it.
    """
    section = yaml.safe_load(
        (REPO_ROOT / "config" / "runner_routing_policy.yaml").read_text(
            encoding="utf-8"
        )
    )["route"]
    for field in ModelCIRunnerRoutePolicy.model_fields:
        assert field not in section, (
            f"{field} is declared in the node contract; a second copy in the "
            f"routing config file is a policy that drifts silently"
        )


def test_no_threshold_is_read_from_an_environment_variable() -> None:
    """Configuration is contract-driven, and the environment is the classic
    escape from that.

    A threshold read from the environment is invisible to every audit, cannot
    be reviewed in a pull request, and differs per runner. Read from the parse,
    for the same reason as the import checks above: this file's own prose says
    the word.
    """
    for path in (HANDLER_PATH, NODE_DIR / "models/model_ci_runner_route_policy.py"):
        imported = _imported_names(path)
        assert "os" not in imported
        assert "getenv" not in imported
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"environ", "getenv"}


def test_the_fleet_size_is_read_from_the_inventory_not_the_contract() -> None:
    """The floor tracks the fleet because the fleet is declared once.

    ``expected_count`` lives in the fleet inventory, which is what the deploy
    and monitor paths already read. Restating it in the contract would be the
    second copy this whole rule exists to prevent -- and it is the copy that
    went stale when the fleet was capped.
    """
    config = _contract()["config"]
    assert "expected_count" not in config
    assert "fleet_expected_count" not in config
    assert "min_online_runners" not in config
    assert "min_online_fraction" in config
    inventory = yaml.safe_load(
        (REPO_ROOT / "config" / "runner_fleet.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(inventory["expected_count"], int)
    assert config["runner_group"] == inventory["runner_group"]
