# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26324: dev's main profile wires only the contracts dev runs, declared as data.

``contracts/services/runtime_lane_node_exclusion.contract.yaml`` names the
contracts the dev profile does not attach and the contracts it needs; a runtime
whose declared lane belongs to that profile reads them. The ownership filter reads it through
``ModelRuntimeNodeSelection.from_env``. Every name asserted here is read from
that file or from ``runtime_policy.contract.yaml``, never spelled in the test.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path

import pytest
import yaml

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.runtime.model_runtime_lane_node_exclusion import (
    LANE_PROFILE_PREFIX,
    RUNTIME_LANE_NODE_EXCLUSION_CONTRACT,
    ModelRuntimeLaneNodeExclusion,
)
from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ENV_RUNTIME_NODE_EXCLUSION,
    ModelRuntimeNodeSelection,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    runtime_profile_owns_contract,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACTS_DIR = REPO_ROOT / "contracts"
DEV_PROFILE = "dev"
DEV_LANE = f"{LANE_PROFILE_PREFIX}{DEV_PROFILE}"
assert DEV_LANE in REGISTERED_RUNTIME_LANES
UNRELATED = "node_unrelated_to_the_dev_profile_omn26324"


def _dev_environ(contracts_dir: Path = CONTRACTS_DIR) -> dict[str, str]:
    return {"ONEX_RUNTIME_LANE": DEV_LANE, "ONEX_CONTRACTS_DIR": str(contracts_dir)}


def _declaration(contracts_dir: Path = CONTRACTS_DIR) -> ModelRuntimeLaneNodeExclusion:
    return ModelRuntimeLaneNodeExclusion.load(
        ModelRuntimeLaneNodeExclusion.declaration_path(contracts_dir)
    )


def _excluded(contracts_dir: Path = CONTRACTS_DIR) -> tuple[str, ...]:
    return _declaration(contracts_dir).entry_for(DEV_LANE).excluded_contracts


def _required(contracts_dir: Path = CONTRACTS_DIR) -> tuple[str, ...]:
    """What dev needs: the declaration's list and dev main's boot priority list."""
    policy = yaml.safe_load(
        (CONTRACTS_DIR / "services" / "runtime_policy.contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    priority = policy["profiles"]["dev"]["services"]["main"]["boot"][
        "priority_contracts"
    ]
    declared = _declaration(contracts_dir).entry_for(DEV_LANE).required_contracts
    return tuple(dict.fromkeys((*declared, *priority)))


def _contract(name: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="COMPUTE_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
    )


def _manifest(*name_groups: tuple[str, ...]) -> ModelAutoWiringManifest:
    names = dict.fromkeys(name for group in name_groups for name in group)
    return ModelAutoWiringManifest(contracts=tuple(_contract(n) for n in names))


def _wired(manifest: ModelAutoWiringManifest, environ: dict[str, str]) -> set[str]:
    result = filter_manifest_for_runtime_profile(manifest, "main", environ=environ)
    return {contract.name for contract in result.manifest.contracts}


def _dropped_required(contracts_dir: Path) -> set[str]:
    """Required contracts the dev lane would not wire from this declaration."""
    required = _required(contracts_dir)
    manifest = _manifest(required, _excluded(contracts_dir), (UNRELATED,))
    return set(required) - _wired(manifest, _dev_environ(contracts_dir))


def _mutated_contracts_dir(tmp_path: Path, extra_exclusion: str) -> Path:
    """A copy of the declaration whose dev exclusion also names one more contract."""
    declared = REPO_ROOT / "contracts" / RUNTIME_LANE_NODE_EXCLUSION_CONTRACT
    target = tmp_path / RUNTIME_LANE_NODE_EXCLUSION_CONTRACT
    target.parent.mkdir(parents=True)
    shutil.copy(declared, target)
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    data["profiles"][DEV_PROFILE]["excluded_contracts"].append(extra_exclusion)
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


def test_dev_profile_contract_set_declares_a_trim() -> None:
    assert _excluded(), "the dev lane declares no excluded contract"


def test_dev_profile_contract_set_leaves_declared_contracts_unattached() -> None:
    excluded = _excluded()
    manifest = _manifest(excluded, (UNRELATED,))

    result = filter_manifest_for_runtime_profile(
        manifest, "main", environ=_dev_environ()
    )

    assert {c.name for c in result.manifest.contracts} == {UNRELATED}
    assert result.node_selection_excluded_contracts == excluded
    assert result.manifest.errors == ()


def test_dev_profile_contract_set_reaches_the_raw_ownership_path() -> None:
    """The legacy subscription loops read the same list as the manifest filter."""
    name = _excluded()[0]

    assert not runtime_profile_owns_contract(
        {"name": name}, "main", environ=_dev_environ()
    )
    assert runtime_profile_owns_contract(
        {"name": UNRELATED}, "main", environ=_dev_environ()
    )


@pytest.mark.parametrize("lane", sorted(REGISTERED_RUNTIME_LANES - {DEV_LANE}) + [""])
def test_dev_profile_contract_set_leaves_every_other_lane_unchanged(lane: str) -> None:
    excluded = _excluded()
    environ = {"ONEX_RUNTIME_LANE": lane, "ONEX_CONTRACTS_DIR": str(CONTRACTS_DIR)}

    assert _wired(_manifest(excluded, (UNRELATED,)), environ) == {*excluded, UNRELATED}
    assert ModelRuntimeNodeSelection.from_env(environ).excluded == ()


def test_dev_profile_contract_set_absent_declaration_keeps_todays_behaviour(
    tmp_path: Path,
) -> None:
    excluded = _excluded()
    manifest = _manifest(excluded, (UNRELATED,))

    for environ in (
        {"ONEX_RUNTIME_LANE": DEV_LANE},
        {"ONEX_RUNTIME_LANE": DEV_LANE, "ONEX_CONTRACTS_DIR": ""},
        {"ONEX_RUNTIME_LANE": DEV_LANE, "ONEX_CONTRACTS_DIR": str(tmp_path)},
    ):
        assert _wired(manifest, environ) == {*excluded, UNRELATED}


def test_dev_profile_contract_set_malformed_declaration_refuses(
    tmp_path: Path,
) -> None:
    target = tmp_path / RUNTIME_LANE_NODE_EXCLUSION_CONTRACT
    target.parent.mkdir(parents=True)
    target.write_text("name: runtime_lane_node_exclusion\nversion: 0\n", "utf-8")

    with pytest.raises(ValueError, match="version"):
        ModelRuntimeNodeSelection.from_env(_dev_environ(tmp_path))


def test_dev_profile_contract_set_composes_with_the_environment_exclusion() -> None:
    declared = _excluded()
    environ = {
        **_dev_environ(),
        ENV_RUNTIME_NODE_EXCLUSION: f"{declared[0]},{UNRELATED}",
    }

    selection = ModelRuntimeNodeSelection.from_env(environ)

    assert selection.excluded == (declared[0], UNRELATED)
    assert selection.declared_excluded == declared[1:]
    assert not selection.admits(UNRELATED)
    assert not selection.admits(declared[-1])


def test_dev_profile_contract_set_stale_declared_name_is_a_discovery_error() -> None:
    """A declared name no discovered contract has is reported, and never stops startup."""
    declared = _excluded()
    manifest = _manifest(declared[1:], (UNRELATED,))

    result = filter_manifest_for_runtime_profile(
        manifest, "main", environ=_dev_environ()
    )

    assert {c.name for c in result.manifest.contracts} == {UNRELATED}
    assert [e.entry_point_name for e in result.manifest.errors] == [declared[0]]


def test_dev_profile_contract_set_unknown_environment_name_still_refuses() -> None:
    """The environment's own exclusion keeps its refusal (OMN-19513)."""
    environ = {**_dev_environ(), ENV_RUNTIME_NODE_EXCLUSION: UNRELATED}

    with pytest.raises(RuntimeHostError, match=UNRELATED):
        filter_manifest_for_runtime_profile(
            _manifest(_excluded()), "main", environ=environ
        )


def test_dev_profile_contract_set_adds_no_contract_name_list_under_src() -> None:
    """The declared names live in the YAML; no .py file under src spells a list of them.

    A lone name can be an unrelated literal (an enum value that shares a contract's
    name); two or more declared names in one file is a name list.
    """
    declared = _excluded()
    offenders: dict[str, list[str]] = {}
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        spelled = [
            name
            for name in declared
            if re.search(rf"""["']{re.escape(name)}["']""", text)
        ]
        if len(spelled) >= 2:
            offenders[str(path.relative_to(REPO_ROOT))] = spelled

    assert offenders == {}


def test_dev_required_contracts_wired() -> None:
    required = _required()
    assert required, "no required contract is declared for the dev lane"
    assert not set(required) & set(_excluded())

    assert _dropped_required(CONTRACTS_DIR) == set()


def test_dev_required_contracts_wired_fails_when_one_is_excluded(
    tmp_path: Path,
) -> None:
    """Control: adding one required contract to the exclusion is caught here, not on dev."""
    victim = _required()[0]

    dropped = _dropped_required(_mutated_contracts_dir(tmp_path, victim))

    assert dropped == {victim}


# --- OMN-26357: no contract a declared delegation-check path uses is excluded -------------------

_TOPIC = re.compile(r"^onex\.(cmd|evt|snapshot)\.")
_IGNORED_KEYS = (
    "broker_acl",
    "dlq",
    "widget_topics",
    "seams",
    "probed_topics",
    "chain_scenarios",
    "config/",
    "projection_api",
    "snapshot_reads",
)
_SUBSCRIBE_KEYS = (
    "subscribe",
    "command_topic",
    "input_subscriptions",
    "consumed_events",
    "subscriptions",
    "externally_consumed",
    "handler_routing",
    "response_topic",
    "reply_topic",
)
_PUBLISH_KEYS = (
    "publish",
    "terminal_event",
    "externally_produced",
    "request_topic",
    "emit",
)


def _topics_of(node: object, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """Every topic string in a parsed contract with its direction: sub, pub or both."""
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found += _topics_of(value, (*path, str(key)))
    elif isinstance(node, list):
        for value in node:
            found += _topics_of(value, path)
    elif isinstance(node, str) and _TOPIC.match(node):
        where = "/".join(path).lower()
        if any(key in where for key in _IGNORED_KEYS):
            return found
        if any(key in where for key in _SUBSCRIBE_KEYS):
            found.append((node, "sub"))
        elif any(key in where for key in _PUBLISH_KEYS):
            found.append((node, "pub"))
        else:
            found.append((node, "both"))
    return found


def _visible_contract_files() -> list[Path]:
    """The contract files this environment ships: this repo's nodes and omnibase_core's."""
    roots = [REPO_ROOT / "src"]
    spec = importlib.util.find_spec("omnibase_core")
    if spec is not None and spec.submodule_search_locations:
        roots += [Path(location) for location in spec.submodule_search_locations]
    return sorted(path for root in roots for path in root.rglob("contract.yaml"))


def _topic_edges() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(subscribers by topic, publishers by topic) read from the visible contract files."""
    subscribers: dict[str, set[str]] = defaultdict(set)
    publishers: dict[str, set[str]] = defaultdict(set)
    for path in _visible_contract_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("name"):
            continue
        for topic, direction in _topics_of(data):
            if direction in ("sub", "both"):
                subscribers[topic].add(data["name"])
            if direction in ("pub", "both"):
                publishers[topic].add(data["name"])
    return subscribers, publishers


def _computed_use_set(contracts_dir: Path) -> set[str]:
    """Contracts the declared paths use, by topic chains over the visible contract files.

    The seeds (contract names and topics per path) are read from the declaration;
    a used contract's published topics bring in the contracts that subscribe to them,
    to a fixed point.
    """
    entry = _declaration(contracts_dir).entry_for(DEV_LANE)
    subscribers, publishers = _topic_edges()
    published: dict[str, set[str]] = defaultdict(set)
    for topic, names in publishers.items():
        for name in names:
            published[name].add(topic)
    used: set[str] = set()
    queue: deque[str] = deque()

    def add(name: str) -> None:
        if name not in used:
            used.add(name)
            queue.append(name)

    for use_path in entry.delegation_check_paths:
        for name in use_path.nodes:
            add(name)
        for topic in use_path.topics:
            for name in subscribers.get(topic, ()):
                add(name)
    while queue:
        for topic in published.get(queue.popleft(), ()):
            for name in subscribers.get(topic, ()):
                add(name)
    return used


def _excluded_but_used(contracts_dir: Path) -> set[str]:
    """Excluded contracts that a declared path uses (seeds, the declared reached set, topic chains)."""
    entry = _declaration(contracts_dir).entry_for(DEV_LANE)
    used = _computed_use_set(contracts_dir) | set(entry.reached_contracts)
    return used & set(entry.excluded_contracts)


def test_dev_delegation_path_not_excluded() -> None:
    """No contract a declared delegation-check path uses is in the dev exclusion."""
    entry = _declaration().entry_for(DEV_LANE)
    assert entry.delegation_check_paths, (
        "the dev lane declares no delegation-check path"
    )

    assert _excluded_but_used(CONTRACTS_DIR) == set()


def test_dev_delegation_path_not_excluded_seeds_are_declared_data() -> None:
    entry = _declaration().entry_for(DEV_LANE)

    assert all(path.nodes or path.topics for path in entry.delegation_check_paths)
    assert len({path.id for path in entry.delegation_check_paths}) == len(
        entry.delegation_check_paths
    )
    seeded = {name for path in entry.delegation_check_paths for name in path.nodes}
    assert seeded <= set(entry.reached_contracts)


def test_dev_delegation_path_not_excluded_reached_set_covers_visible_chains() -> None:
    """A contract the visible files reach from the seeds is in the declared reached set."""
    entry = _declaration().entry_for(DEV_LANE)

    assert _computed_use_set(CONTRACTS_DIR) - set(entry.reached_contracts) == set()


def test_dev_delegation_path_not_excluded_fails_when_a_seed_is_excluded(
    tmp_path: Path,
) -> None:
    """Control: excluding one contract a path calls by name is caught here, not on dev."""
    entry = _declaration().entry_for(DEV_LANE)
    victim = next(name for path in entry.delegation_check_paths for name in path.nodes)

    caught = _excluded_but_used(_mutated_contracts_dir(tmp_path, victim))

    assert caught == {victim}


def test_dev_delegation_path_not_excluded_fails_when_a_chained_contract_is_excluded(
    tmp_path: Path,
) -> None:
    """Control: excluding a contract only a topic chain reaches is caught too."""
    entry = _declaration().entry_for(DEV_LANE)
    seeded = {name for path in entry.delegation_check_paths for name in path.nodes}
    chained = sorted(_computed_use_set(CONTRACTS_DIR) - seeded)
    assert chained, "no contract is reached through a topic chain"

    caught = _excluded_but_used(_mutated_contracts_dir(tmp_path, chained[0]))

    assert caught == {chained[0]}


def test_dev_lane_attaches_stall_responder_omn26550() -> None:
    """OMN-26550: the stall responder declares a runtime-tick schedule, so dev wires it."""
    name = "node_stall_responder_compute"

    assert name not in _excluded()
    assert name in _required()
    assert name in _wired(_manifest((name,), (UNRELATED,)), _dev_environ())
