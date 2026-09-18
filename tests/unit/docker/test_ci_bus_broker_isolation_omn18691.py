# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18691: the dedicated CI-bus broker must not share a lane's fate.

WHAT THIS PROTECTS. The fleet's CI publishers used to publish to the DEV LANE's
Redpanda. The dev lane is rebuilt on every runtime-affecting merge, so every
rebuild was a window in which fleet CI could not publish; on 2026-09-18 a
watchdog kill mid-recreate left that broker REMOVED and every Receipt Gate on
the fleet was red for ~35 minutes on pull requests that had nothing to do with
the lane. `docker/docker-compose.ci-bus.yml` moves the CI bus onto its own
compose project so a lane teardown cannot reach it.

WHY THESE ASSERTIONS AND NOT A REVIEW. The isolation is a property of FOUR
separate global namespaces on the Docker daemon -- project, container name,
network name, volume name -- and renaming only some of them isolates nothing.
That is the precise lesson of OMN-15565, where the e2e stack shared the lab
lane's project name as a *default* and its nightly `down -v` deleted the live
lane's data volumes for at least eight consecutive nights. The collision was
invisible at review time. It is not invisible to a test.

NOTHING ABOUT LANE IDENTITY IS HARDCODED HERE. Every protected name is derived
from `deploy/lane-census/lane-manifest.yaml` and the lane compose files it
names, so a lane added later is covered without editing this file -- the same
construction `test_e2e_compose_lane_isolation.py` uses, and for the same reason.

THE PORT ASSERTION IS THE ONE THAT WOULD HAVE BEEN CAUGHT LATE. A port collision
does not fail at `docker compose config`; it fails at `up`, on the host, against
whichever project claimed the port first -- which on a host running six projects
is a coin flip resolved at 3am.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

#: `${VAR:-default}` -> `default`, and a bare `${VAR}` -> nothing (it binds an
#: address this test cannot resolve, so it cannot assert a collision on it).
_DEFAULTED_VAR = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]*)\}")
_BARE_VAR = re.compile(r"\$\{[^}]*\}")

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_BUS_COMPOSE = REPO_ROOT / "docker" / "docker-compose.ci-bus.yml"
LANE_MANIFEST = REPO_ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"

#: The manifest entry this file is about. It is a lane entry for census
#: purposes only -- it runs no runtime -- so it must be excluded from the set of
#: LANES whose namespaces it is checked against, or it would collide with itself.
CI_BUS_LANE_KEY = "ci-bus"


class _ComposeLoader(yaml.SafeLoader):
    """A SafeLoader that tolerates compose's own YAML tags.

    The lane overlays legitimately use `!override` (compose's "replace this
    sequence instead of appending to it") and `!!merge`. `yaml.safe_load`
    refuses both, and this module has to read those files to derive the names
    and ports it protects against. Dropping the tag and keeping the value is
    correct here: this test reads identifiers, never merge semantics.
    """


def _drop_tag(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    # Unreachable for any YAML this repo contains: a node is one of the three
    # kinds above. Refuse rather than return None, so an unparsed value can
    # never read downstream as "this file declares no ports".
    raise AssertionError(f"unhandled YAML node kind: {type(node).__name__}")


_ComposeLoader.add_multi_constructor("!", _drop_tag)  # type: ignore[no-untyped-call]
_ComposeLoader.add_multi_constructor(  # type: ignore[no-untyped-call]
    "tag:yaml.org,2002:", _drop_tag
)


def _parse(text: str) -> Any:
    return yaml.load(text, Loader=_ComposeLoader)  # noqa: S506 - local tolerant loader


def _load(path: Path) -> dict[str, Any]:
    """Load a YAML document, refusing an absent or non-mapping file.

    A missing compose file must fail the test rather than skip it. A skip here
    would report green on exactly the change that deleted the thing under test.
    """
    if not path.is_file():
        raise AssertionError(
            f"{path} is missing. This test compared nothing; it did not pass."
        )
    loaded = _parse(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError(f"{path} did not parse as a mapping.")
    return loaded


@pytest.fixture(scope="module")
def ci_bus() -> dict[str, Any]:
    return _load(CI_BUS_COMPOSE)


@pytest.fixture(scope="module")
def lane_manifest() -> dict[str, Any]:
    return _load(LANE_MANIFEST)


def _lane_specs(lane_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every manifest lane EXCEPT the CI bus entry itself."""
    lanes = lane_manifest.get("lanes")
    assert isinstance(lanes, dict) and lanes, "lane manifest declares no lanes"
    return {
        name: spec
        for name, spec in lanes.items()
        if name != CI_BUS_LANE_KEY and isinstance(spec, dict)
    }


def _published_ports(compose: dict[str, Any]) -> set[str]:
    """Host-side published ports of a compose document, as literal strings.

    Read from the LEFT of the `host:container` mapping. A `${VAR:-default}`
    form contributes its default, which is the value the host actually binds
    when the variable is unset -- which is how these files are rendered in
    practice.
    """
    published: set[str] = set()
    for service in (compose.get("services") or {}).values():
        if not isinstance(service, dict):
            continue
        for entry in service.get("ports") or []:
            if not isinstance(entry, str):
                continue
            # Resolve `${VAR:-default}` FIRST. Splitting on ':' before this
            # would cut the mapping inside the interpolation, which is a real
            # parsing bug this comment exists to stop being reintroduced.
            resolved = _DEFAULTED_VAR.sub(lambda m: m.group(1), entry)
            resolved = _BARE_VAR.sub("", resolved)
            parts = [p for p in resolved.strip().strip('"').split(":") if p]
            if len(parts) < 2:
                # A bare `9092` (container port only, ephemeral host port) binds
                # nothing predictable and cannot collide.
                continue
            host_side = parts[-2] if parts[-1].isdigit() else parts[-1]
            host_side = host_side.split("/")[0]
            if host_side.isdigit():
                published.add(host_side)
    return published


def test_ci_bus_declares_its_own_compose_project(ci_bus: dict[str, Any]) -> None:
    """The project name is a LITERAL, never a default derived from a directory.

    OMN-15565's whole failure was a project name that was a default. A literal
    `name:` is what makes a teardown of the lane project — volumes included —
    provably unable to reach this one, because a compose teardown is scoped to
    the project it names.
    """
    assert ci_bus.get("name") == "omninode-ci-bus", (
        "docker-compose.ci-bus.yml must declare a literal `name:` of "
        "'omninode-ci-bus'. Without it the project name defaults to the "
        "directory, which is how the e2e stack once resolved into the live lane."
    )


def test_ci_bus_project_is_not_any_lane_project(
    ci_bus: dict[str, Any], lane_manifest: dict[str, Any]
) -> None:
    """The CI bus must not share a compose project with any declared lane."""
    lane_projects = {
        spec.get("compose_project") for spec in _lane_specs(lane_manifest).values()
    }
    assert ci_bus.get("name") not in lane_projects, (
        f"CI-bus compose project {ci_bus.get('name')!r} collides with a lane "
        "project. A shared project means a lane teardown removes the CI bus, "
        "which is the entire defect OMN-18691 exists to close."
    )


def test_ci_bus_network_is_not_any_lane_network(
    ci_bus: dict[str, Any], lane_manifest: dict[str, Any]
) -> None:
    """Network names are a separate global namespace from project names."""
    lane_networks = {
        str(spec["network"])
        for spec in _lane_specs(lane_manifest).values()
        if spec.get("network")
    }
    declared = {
        str(net["name"])
        for net in (ci_bus.get("networks") or {}).values()
        if isinstance(net, dict) and net.get("name")
    }
    assert declared, "the CI-bus project must name its network explicitly"
    overlap = declared & lane_networks
    assert not overlap, (
        f"CI-bus network(s) {sorted(overlap)} collide with a lane network. "
        "Attaching to a lane's network re-couples the broker's fate to that "
        "lane's `down`."
    )


def test_ci_bus_volume_is_named_and_distinct(
    ci_bus: dict[str, Any], lane_manifest: dict[str, Any]
) -> None:
    """The data volume is the durability the whole design rests on.

    It is what makes a command published during a lane rebuild still be there
    afterwards. An unnamed volume would be project-prefixed and therefore still
    safe, but a volume NAMED like a lane's would be deleted by that lane's
    `down -v` -- the OMN-15565 shape exactly.
    """
    volumes = ci_bus.get("volumes") or {}
    names = {vol.get("name") for vol in volumes.values() if isinstance(vol, dict)}
    assert names, "the CI-bus project must name its data volume explicitly"

    lane_volume_names: set[str] = set()
    for spec in _lane_specs(lane_manifest).values():
        compose_file = spec.get("compose_file")
        if not compose_file:
            continue
        path = REPO_ROOT / str(compose_file)
        if not path.is_file():
            continue
        lane_doc = _parse(path.read_text(encoding="utf-8")) or {}
        for vol in (lane_doc.get("volumes") or {}).values():
            if isinstance(vol, dict) and vol.get("name"):
                lane_volume_names.add(str(vol["name"]))

    overlap = {n for n in names if n in lane_volume_names}
    assert not overlap, (
        f"CI-bus volume(s) {sorted(overlap)} collide with a lane volume name. "
        "A lane `down -v` would delete the CI bus's durable queue."
    )


def test_ci_bus_container_names_are_distinct_from_every_lane(
    ci_bus: dict[str, Any], lane_manifest: dict[str, Any]
) -> None:
    """Container names are a THIRD global namespace and must not collide.

    An explicit `container_name` bypasses compose's project prefixing entirely,
    so two projects can claim the same name and the second `up` fails on the
    host rather than at render time.
    """
    ci_names = {
        service["container_name"]
        for service in (ci_bus.get("services") or {}).values()
        if isinstance(service, dict) and service.get("container_name")
    }
    assert ci_names, "every CI-bus service must set an explicit container_name"

    lane_names: set[str] = set()
    for spec in _lane_specs(lane_manifest).values():
        for svc in spec.get("services") or []:
            if isinstance(svc, dict) and svc.get("name"):
                lane_names.add(str(svc["name"]))

    overlap = ci_names & lane_names
    assert not overlap, (
        f"CI-bus container name(s) {sorted(overlap)} are already declared by a "
        "lane in the census. Two projects cannot hold one container name."
    )


def test_ci_bus_publishes_no_port_any_lane_publishes(ci_bus: dict[str, Any]) -> None:
    """Host ports are the FOURTH namespace, and the one that fails latest.

    A collision renders fine and fails at `up` on the host, against whichever
    project bound the port first. The lane ports are read out of the lane compose
    files rather than listed here, so a lane that claims a new port later moves
    this assertion with it.
    """
    ci_ports = _published_ports(ci_bus)
    assert ci_ports, "the CI-bus broker must publish at least one host port"

    lane_ports: dict[str, set[str]] = {}
    for compose_path in sorted((REPO_ROOT / "docker").glob("docker-compose.*.yml")):
        if compose_path == CI_BUS_COMPOSE:
            continue
        doc = _parse(compose_path.read_text(encoding="utf-8")) or {}
        if not isinstance(doc, dict):
            continue
        found = _published_ports(doc)
        if found:
            lane_ports[compose_path.name] = found

    collisions = {
        name: sorted(ports & ci_ports)
        for name, ports in lane_ports.items()
        if ports & ci_ports
    }
    assert not collisions, (
        f"CI-bus host port(s) collide with another compose file: {collisions}. "
        "This would not fail at render; it would fail at `up` on .201, against "
        "whichever project bound the port first."
    )


def test_ci_bus_is_declared_in_the_lane_census(lane_manifest: dict[str, Any]) -> None:
    """A running-but-undeclared container is the dangerous direction.

    The `judge` lane ran for months while absent from the generated lane table
    (retro B-6 / OMN-13034). The CI bus is declared before it is built, which is
    the loud direction and is intended.
    """
    lanes = lane_manifest.get("lanes") or {}
    assert CI_BUS_LANE_KEY in lanes, (
        "the CI-bus project must be declared in deploy/lane-census/"
        "lane-manifest.yaml, or the census cannot see it and a container that "
        "is running is invisible."
    )
    spec = lanes[CI_BUS_LANE_KEY]
    assert spec.get("compose_file") == "docker/docker-compose.ci-bus.yml"
    assert spec.get("compose_project") == "omninode-ci-bus"
    declared = {svc["name"] for svc in spec["services"] if isinstance(svc, dict)}
    compose = _load(CI_BUS_COMPOSE)
    actual = {
        service["container_name"]
        for service in (compose.get("services") or {}).values()
        if isinstance(service, dict) and service.get("container_name")
    }
    assert declared == actual, (
        "the census entry and the compose file disagree about which containers "
        f"this project runs: census-only={sorted(declared - actual)}, "
        f"compose-only={sorted(actual - declared)}. A census that does not match "
        "the file it names reports drift it invented, or misses drift that is real."
    )


def test_ci_bus_requires_its_advertise_host_and_credentials(
    ci_bus: dict[str, Any],
) -> None:
    """Every operator-supplied value fails CLOSED at render, never defaults.

    A `${VAR:-localhost}` advertise address renders a broker address that no
    off-host client can resolve -- and every client this broker has is an
    off-host GitHub Actions runner, so the symptom would surface far from the
    file (OMN-15173). A `:-` fallback on the SASL password would create a
    principal with a guessable secret on a tailnet-reachable broker.
    """
    raw = CI_BUS_COMPOSE.read_text(encoding="utf-8")
    for required in (
        "CI_BUS_REDPANDA_ADVERTISE_HOST:?",
        "CI_BUS_KAFKA_SASL_USERNAME:?",
        "CI_BUS_KAFKA_SASL_PASSWORD:?",
    ):
        assert required in raw, (
            f"{required[:-2]} must use the `:?` required form. A `:-` default "
            "here renders a broker nobody can reach or a credential anybody can "
            "guess, and both read as a successful deploy."
        )
    for forbidden in (
        "CI_BUS_REDPANDA_ADVERTISE_HOST:-",
        "CI_BUS_KAFKA_SASL_USERNAME:-",
        "CI_BUS_KAFKA_SASL_PASSWORD:-",
    ):
        assert forbidden not in raw, (
            f"{forbidden[:-2]} must never carry a `:-` default."
        )


def test_ci_bus_topics_are_exactly_the_ci_bus_set(ci_bus: dict[str, Any]) -> None:
    """The topic list in the bring-up one-shot is pinned here, not by review.

    THIS TEST IS WHY `docker/docker-compose.ci-bus.yml` IS EXCLUDED FROM THE
    `no-hardcoded-topics` PRE-COMMIT GATE. That gate stops a topic name in source
    drifting from the contract that declares it. Every other lane satisfies it by
    writing no topic names at all -- the runtime's TopicProvisioner creates each
    contract-declared topic at boot by reading the contracts. That mechanism is
    unavailable on this project by design: a broker that must survive a dev-lane
    rebuild cannot be hosted inside anything a lane rebuild recreates, so it runs
    no contract-reading process, so its topics are nobody's job but the one-shot's.

    The gate is therefore replaced rather than excused: an edit to that list is a
    red test here. The exclusion without this assertion would be a hole.

    THE FIVE NAMES. Four are the topics the thin CI publishers write to; the fifth
    is consumed by the rebuild trigger when it waits for a rebuild, so it rides the
    same broker and is provisioned with them.
    """
    expected = {
        "onex.cmd.omnimarket.occ-autobind.v1",  # onex-topic-allow: OMN-18691 CI-bus provisioning set
        "onex.cmd.omnimarket.occ-companion-effect-requested.v1",  # onex-topic-allow: OMN-18691
        "onex.evt.github.pr-merged.v1",  # onex-topic-allow: OMN-18691
        "onex.cmd.omnimarket.redeploy-start.v1",  # onex-topic-allow: OMN-18691
        "onex.evt.deploy.rebuild-completed.v1",  # onex-topic-allow: OMN-18691
    }

    raw = CI_BUS_COMPOSE.read_text(encoding="utf-8")
    found = set(re.findall(r"onex\.[a-z]+\.[a-z0-9.-]+\.v\d+", raw))
    assert found == expected, (
        "the CI-bus bring-up one-shot provisions a different topic set than this "
        f"test pins: compose-only={sorted(found - expected)}, "
        f"test-only={sorted(expected - found)}. A topic the publishers write to "
        "but the one-shot does not create is a publish that fails on a broker "
        "with no runtime to create it; a topic created but never written is dead "
        "retention. Change both together, deliberately."
    )

    # Cross-check the one name this repo owns a canonical constant for. The other
    # four are declared in omnimarket's topic registry, which this repo does not
    # import -- stated rather than left as an apparent omission.
    from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_GITHUB_PR_MERGED

    assert SUFFIX_GITHUB_PR_MERGED in expected, (
        "the canonical pr-merged topic constant no longer matches the name the "
        "CI-bus one-shot creates. The registry moved and this file did not."
    )


def test_ci_bus_topics_are_created_explicitly_not_auto(
    ci_bus: dict[str, Any],
) -> None:
    """Topics are created by a one-shot, never left to broker auto-create.

    None of the four publishers carries an AdminClient; they produce and flush.
    On a brand-new broker, auto-create would silently yield a 1-partition topic
    with default retention on first publish, and partitions can only ever be
    INCREASED -- so the first consumer to attach would find drift it can repair
    in one direction only.
    """
    services = ci_bus.get("services") or {}
    assert "ci-bus-topics" in services, (
        "the CI-bus project must carry an explicit topic-creation one-shot."
    )
    command = str(services["ci-bus-topics"].get("command"))
    assert "--partitions 6" in command, (
        "topics must be created at 6 partitions -- the platform default for a "
        "topic whose contract declares no override, which is exactly what the "
        "runtime provisioner would create. Matching it means a consumer "
        "attaching later finds nothing to reconcile."
    )
    assert "--replicas 1" in command, (
        "this is a single-node cluster; any higher replication factor leaves "
        "every topic permanently under-replicated."
    )
    assert "retention.ms=604800000" in command, (
        "CI commands are not a ledger. Retention is 7 days so this broker cannot "
        "grow into the disk problem that produced the outage it exists to stop."
    )
