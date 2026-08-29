# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract-conformance guard for the secret-seeding node — OMN-16897.

The seeding node is invoked by hand, occasionally, by an operator who needs
it to work on the day they reach for it. Nothing else exercises it, so
without this file an unrelated edit could silently unwire it and the failure
would surface only at the moment someone was trying to land a production
credential. These tests are the enforcement (CLAUDE.md Rule 5) that keeps
the contract, the skill mapping, and the entry point in lock-step.

Two of them are not wiring checks at all but VALUE-FLOW checks, asserted
against the declared surface rather than against handler behaviour:

* the result model must never grow a value-carrying field — it is the thing
  that gets printed, pasted into tickets, and potentially published;
* the skill mapping must never grow a CLI arg that carries a value — those
  args become the node payload, which is serialised onto the bus and into
  the event log.

Both are the kind of change that looks harmless in review ("just surface the
value so the receipt is complete") and is not.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.nodes.node_secret_seed_effect.models.model_secret_seed_result import (
    ModelSecretSeedResult,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NODE_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "nodes" / "node_secret_seed_effect"
_CONTRACT = _NODE_DIR / "contract.yaml"
_SKILL_MAPPING = _REPO_ROOT / "src" / "omnibase_infra" / "cli" / "skill_mapping.yaml"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_NODE_NAME = "node_secret_seed_effect"
_SKILL_NAME = "seed_secrets"
_HANDLER_MODULE = (
    "omnibase_infra.nodes.node_secret_seed_effect.handlers.handler_secret_seed"
)

# Substrings that would indicate a field or flag able to carry a secret
# VALUE. Every list field on the result model is a NAME list and must stay
# that way.
_VALUE_SHAPED_TOKENS = (
    "value",
    "secret",
    "api_key",
    "password",
    "token",
    "credential",
)

# Names that contain a value-shaped token but are ADDRESSES, not values.
# Enumerated rather than pattern-excluded so that adding a genuinely
# value-carrying field still trips the guard — an exclusion has to be
# written down deliberately, which is the point.
_ADDRESS_FIELD_ALLOWLIST = frozenset({"secret_path"})


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    loaded = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def skill_row() -> dict[str, Any]:
    loaded = yaml.safe_load(_SKILL_MAPPING.read_text(encoding="utf-8"))
    rows = [s for s in loaded["skills"] if s["skill_name"] == _SKILL_NAME]
    assert len(rows) == 1, f"expected exactly one '{_SKILL_NAME}' mapping row"
    return dict(rows[0])


# --- value-flow surface guards ---------------------------------------------


def test_result_model_declares_no_value_carrying_field() -> None:
    """The receipt is printed, pasted, and possibly published.

    ``created_names`` / ``verified_names`` and friends are deliberately the
    only list fields. A field whose name suggests it holds a value is a
    leak waiting to be serialised.
    """
    offenders = [
        name
        for name in ModelSecretSeedResult.model_fields
        if name not in _ADDRESS_FIELD_ALLOWLIST
        and any(token in name.lower() for token in _VALUE_SHAPED_TOKENS)
    ]
    assert offenders == [], (
        f"ModelSecretSeedResult grew value-shaped field(s) {offenders}. The "
        "receipt must carry names, counts, addresses, and redacted messages "
        "only — never key material."
    )


def test_skill_mapping_exposes_no_value_carrying_cli_arg(
    skill_row: dict[str, Any],
) -> None:
    """A --value flag would put key material into the node payload.

    ``onex skill`` builds the backing node's payload from these args, and
    that payload is serialised onto the bus and into the event log. The
    supported shape is ``--source-path``, which names a local file.
    """
    offenders = [
        arg["name"]
        for arg in skill_row["args"]
        if arg["payload_field"] not in _ADDRESS_FIELD_ALLOWLIST
        and any(token in arg["name"].lower() for token in _VALUE_SHAPED_TOKENS)
    ]
    assert offenders == [], (
        f"skill '{_SKILL_NAME}' grew value-shaped CLI arg(s) {offenders}; "
        "values must arrive via --source-path, never on the payload."
    )
    assert any(arg["name"] == "source-path" for arg in skill_row["args"])


def test_contract_declares_no_bus_topics(contract: dict[str, Any]) -> None:
    """A secret-writing node must not be remotely triggerable.

    Seeding is an operator-invoked action against one named instance. A
    declared ``subscribe_topics`` entry is how the runtime auto-wires a live
    consumer, which would turn this into a node anyone who can publish to a
    topic can make write secrets.
    """
    event_bus = contract.get("event_bus") or {}
    assert not event_bus.get("subscribe_topics")
    assert not event_bus.get("publish_topics")


# --- wiring lock-step ------------------------------------------------------


def test_contract_identity_and_archetype(contract: dict[str, Any]) -> None:
    assert contract["name"] == _NODE_NAME
    assert contract["node_type"] == "EFFECT_GENERIC"


def test_contract_declares_the_toplevel_handler_block(
    contract: dict[str, Any],
) -> None:
    """RuntimeLocal single-shot dispatch resolves the handler from HERE.

    Without this block ``onex skill seed_secrets`` fails closed with
    "Workflow contract missing 'terminal_event' topic and no handler spec
    found" — the trap node_chain_canary_effect documents in its own
    contract. handler_routing alone is not enough for the local path.
    """
    handler = contract["handler"]
    assert handler["module"] == _HANDLER_MODULE
    assert handler["class"] == "HandlerSecretSeed"
    assert handler["input_model"].endswith("ModelSecretSeedRequest")


def test_handler_is_registered_in_handler_routing(
    contract: dict[str, Any],
) -> None:
    """Required by arch-handler-contract-compliance.

    A handler module absent from ``handler_routing.handlers[]`` is reported
    as MISSING_HANDLER_ROUTING and classified DEAD by the CI scanner.
    """
    modules = {
        entry["handler"]["module"] for entry in contract["handler_routing"]["handlers"]
    }
    assert _HANDLER_MODULE in modules


def test_contract_declares_the_machine_identity_dependencies(
    contract: dict[str, Any],
) -> None:
    """Fail-fast auth is only honest if the contract says what it needs."""
    env_vars = {
        dep.get("env_var")
        for dep in contract["dependencies"]
        if dep.get("type") == "environment"
    }
    assert {"INFISICAL_CLIENT_ID", "INFISICAL_CLIENT_SECRET"} <= env_vars


def test_skill_mapping_points_at_this_node(skill_row: dict[str, Any]) -> None:
    assert skill_row["node_name"] == _NODE_NAME
    assert skill_row["result_model"].endswith("ModelSecretSeedResult")


def test_skill_mapping_requires_every_addressing_arg(
    skill_row: dict[str, Any],
) -> None:
    """A defaulted target would seed a real key into the wrong instance.

    This estate runs three separate Infisical servers. CLAUDE.md Rule 8:
    fail fast on missing config rather than silently pick one.
    """
    required = {arg["name"] for arg in skill_row["args"] if arg.get("required")}
    assert required == {
        "source-path",
        "infisical-host",
        "project-id",
        "environment-slug",
        "secret-path",
    }


def test_writing_is_a_positive_opt_in_flag(skill_row: dict[str, Any]) -> None:
    """``--execute``, never ``--dry-run``.

    ``onex skill`` boolean args are PRESENCE flags — ``cli_skill`` sets True
    on sight and has no path that sets False. A ``--dry-run`` defaulting to
    true could therefore never be turned off from the command line, so the
    flag is inverted: absent means plan-only.
    """
    by_name = {arg["name"]: arg for arg in skill_row["args"]}
    assert "dry-run" not in by_name
    execute = by_name["execute"]
    assert execute["arg_type"] == "boolean"
    assert execute["payload_field"] == "execute"
    assert execute["default"] is False


def test_node_is_registered_as_an_onex_entry_point() -> None:
    """Unregistered, the skill mapping resolves to nothing at dispatch."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    nodes = data["project"]["entry-points"]["onex.nodes"]
    assert nodes[_NODE_NAME] == f"omnibase_infra.nodes.{_NODE_NAME}"


def test_node_py_is_declarative() -> None:
    """``node.py`` must hold no logic — handlers own behaviour.

    The compliance scanner reports any method beyond ``__init__`` as
    LOGIC_IN_NODE, and the repo invariant is stricter than the scanner:
    nodes coordinate, handlers compute.
    """
    source = (_NODE_DIR / "node.py").read_text(encoding="utf-8")
    assert source.count("    def ") == 1
    assert "def __init__" in source
    assert "super().__init__(container)" in source
