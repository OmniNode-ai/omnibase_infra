# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Derive the stability-lane health gate's declared consumer groups [OMN-15837].

WHY THIS MODULE EXISTS
----------------------
``consumer_groups_stability.yaml`` used to carry FULL, VERSION-PINNED Kafka
consumer-group names, hand-copied from ``rpk group list``. Consumer group ids
are minted from the *contract* identity --
``{env}.{package}.{node}.consume.{contract_version}`` plus a
``.__i.{instance}`` discriminator and a ``.__t.{topic}`` per-topic suffix (see
``omnibase_infra.utils.util_consumer_group`` and
``EventBusKafka._resolve_effective_group_id``) -- so the pinned name goes stale
the moment a contract version is bumped or a consumer is re-homed. A stale name
does not read as "unknown": ``rpk group describe`` on a name that is not a real
group answers ``STATE Dead / MEMBERS 0 / TOTAL-LAG 0``, which the gate scored as
a hard failure and rolled a HEALTHY refresh back on.

That has now happened twice on the same file: ``fd4a84b1c`` repinned a stale
``.consume.<version>.`` segment by hand (OMN-15838), and on 2026-09-08 the
refresh in OMN-16753 was rolled back on two more stale names -- one a
``0.3.0 -> 0.4.0`` version bump, one a projection re-homed out of the runtime
into a standalone writer container (OMN-17562). Both retired identities read
``Dead / 0 / 0``; every other gate criterion passed.

WHAT REPLACES IT
----------------
The declared set is DERIVED at gate time from the auto-wiring manifest the
refreshed image itself serves on ``/v1/introspection/manifest``, using the same
identity components the runtime mints group ids from. A version bump or a
re-home changes the manifest and therefore changes the derived set in the same
step -- there is nothing left to keep in sync by hand.

The YAML file keeps only what NO contract declares: consumer groups minted
outside contract auto-wiring (the standalone projection writers), each entry
naming its source. Those are matched by exact name and carry no contract
version.

FAIL-CLOSED
-----------
There is no fallback to a static list. If the manifests cannot be fetched or
parsed, if the derivation yields nothing, or if the derived env prefix matches
no live group at all, the caller raises ``DerivationError`` and the gate reports
INFRA_ERROR naming the cause. A gate that cannot derive its own expectation has
not passed; it has not run.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

# ─── Group-id minting (mirrors omnibase_infra.utils.util_consumer_group) ─────
#
# Deliberately re-implemented here rather than imported. This verifier is a
# standalone ops script executed as ``python scripts/runtime_build/verify_*.py``
# against a lane that may be running a DIFFERENT image than the clone the script
# lives in; importing the clone's runtime package would make the gate's answer
# depend on the clone's code path (and on the package being importable at all in
# the deploy-runner container). The three rules below are a stable, published
# wire format -- ``normalize_kafka_identifier`` has not changed since OMN-1602 --
# and ``test_declared_consumer_groups_omn15837.py`` pins them against the real
# group names read live off the lane broker.

KAFKA_CONSUMER_GROUP_MAX_LENGTH = 255
_INVALID_CHAR_PATTERN = re.compile(r"[^a-z0-9._-]")
_CONSECUTIVE_SEPARATOR_PATTERN = re.compile(r"[._-]{2,}")
_EDGE_SEPARATOR_PATTERN = re.compile(r"^[._-]+|[._-]+$")

INSTANCE_INFIX = ".__i."
TOPIC_INFIX = ".__t."

CONSUME_PURPOSE = "consume"


class DerivationError(RuntimeError):
    """The declared set could not be derived. Always fail closed, never fall back."""


def normalize_kafka_identifier(value: str) -> str:
    """Lowercase, replace invalid chars, collapse separators, strip edges."""
    if not value:
        raise DerivationError("consumer-group identifier component is empty")
    result = value.lower()
    result = _INVALID_CHAR_PATTERN.sub("_", result)
    result = _CONSECUTIVE_SEPARATOR_PATTERN.sub(lambda m: m.group(0)[0], result)
    result = _EDGE_SEPARATOR_PATTERN.sub("", result)
    if not result:
        raise DerivationError(
            f"identifier component {value!r} normalizes to an empty string"
        )
    if len(result) > KAFKA_CONSUMER_GROUP_MAX_LENGTH:
        hash_suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
        result = f"{result[: KAFKA_CONSUMER_GROUP_MAX_LENGTH - 9]}_{hash_suffix}"
    return result


def compute_base_group_id(
    *, env: str, service: str, node_name: str, version: str
) -> str:
    """``{env}.{service}.{node_name}.consume.{version}``, each part normalized."""
    parts = [
        normalize_kafka_identifier(env),
        normalize_kafka_identifier(service),
        normalize_kafka_identifier(node_name),
        normalize_kafka_identifier(CONSUME_PURPOSE),
        normalize_kafka_identifier(version),
    ]
    group_id = ".".join(parts)
    if len(group_id) > KAFKA_CONSUMER_GROUP_MAX_LENGTH:
        hash_input = f"{env}|{service}|{node_name}|{CONSUME_PURPOSE}|{version}"
        hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        group_id = f"{group_id[: KAFKA_CONSUMER_GROUP_MAX_LENGTH - 9]}_{hash_suffix}"
    return group_id


def split_group_name(group: str) -> tuple[str, str | None, str | None]:
    """Split a live group id into ``(base, instance, topic)``.

    ``base.__i.<instance>.__t.<topic>`` -> ``(base, instance, topic)``
    ``base.__t.<topic>``                -> ``(base, None, topic)``
    ``base``                            -> ``(base, None, None)``

    The topic is taken from the LAST ``.__t.`` so a topic name that itself
    contained the infix could not shift the split (mirrors
    ``probe_subscription._topic_from_scoped_group``).
    """
    head, _, topic = group.rpartition(TOPIC_INFIX)
    if not head:
        head, topic_value = group, None
    else:
        topic_value = topic
    base, _, instance = head.rpartition(INSTANCE_INFIX)
    if not base:
        return head, None, topic_value
    return base, instance, topic_value


# ─── Declared-set inputs ────────────────────────────────────────────────────


@dataclass(frozen=True)
class NonContractGroup:
    """A consumer group no contract declares, matched by exact name."""

    name: str
    source: str


@dataclass(frozen=True)
class DerivedGroupKey:
    """One contract-declared subscription, as a group identity + topic."""

    base_group_id: str
    topic: str
    contract_name: str
    package: str
    contract_version: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.base_group_id, self.topic)

    @property
    def probe_name(self) -> str:
        """Instance-agnostic name used only for reporting an absent identity."""
        return f"{self.base_group_id}{TOPIC_INFIX}{self.topic}"


def load_non_contract_groups(path: Path) -> tuple[NonContractGroup, ...]:
    """Read the allowlist of groups no contract declares.

    Raises ``DerivationError`` rather than returning an empty allowlist on a
    malformed file: an unreadable declaration must not silently narrow the
    gate's surface.
    """
    try:
        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise DerivationError(
            f"cannot read declared-groups file {path}: {exc}"
        ) from exc
    if not isinstance(data, Mapping):
        raise DerivationError(f"declared-groups file {path} is not a mapping")
    if "consumer_groups" in data:
        raise DerivationError(
            f"{path} still carries the retired `consumer_groups:` key (a "
            "hand-pinned, version-pinned group list). OMN-15837 replaced it "
            "with contract derivation plus `non_contract_groups:`; re-pinning "
            "names by hand is the defect, not the fix."
        )
    raw = data.get("non_contract_groups")
    if raw is None:
        raise DerivationError(
            f"{path} declares no `non_contract_groups:` key (use an empty list "
            "to declare that every checked group is contract-derived)"
        )
    if not isinstance(raw, list):
        raise DerivationError(f"{path}: `non_contract_groups` must be a list")
    groups: list[NonContractGroup] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise DerivationError(
                f"{path}: non_contract_groups[{index}] is not a mapping"
            )
        name = entry.get("name")
        source = entry.get("source")
        if not isinstance(name, str) or not name.strip():
            raise DerivationError(f"{path}: non_contract_groups[{index}] has no `name`")
        if not isinstance(source, str) or not source.strip():
            raise DerivationError(
                f"{path}: non_contract_groups[{index}] ({name}) has no `source`. "
                "Every allowlisted group names where it is minted -- an "
                "unsourced entry is indistinguishable from a stale one."
            )
        groups.append(NonContractGroup(name=name.strip(), source=source.strip()))
    return tuple(groups)


def derive_expected_keys(
    manifests: Sequence[Mapping[str, object]], *, env: str
) -> tuple[DerivedGroupKey, ...]:
    """Derive every contract-declared subscription identity from the manifests.

    One key per ``(contract, subscribe_topic)``. Skipped:

    * contracts with no ``event_bus`` or no ``subscribe_topics`` -- nothing to
      consume, so no group is ever minted;
    * ``event_bus.plugin_managed`` contracts -- auto-wiring deliberately does
      NOT open a Kafka subscription for these; a domain plugin owns it with its
      own config and mints its own group id (``ModelEventBusWiring``
      docstring). Measured on the live lane 2026-09-08: including them produced
      13 phantom expectations that no live group could ever satisfy.

    The instance discriminator is deliberately NOT derived. It is a per-CONTAINER
    value (``stability-test-main`` / ``-effects`` / ``-worker``), not a contract
    property, and only two of the three runtimes expose an HTTP manifest. Keys
    are matched instance-agnostically against the live group list instead, so a
    subscription satisfied by any runtime instance counts.
    """
    keys: dict[tuple[str, str], DerivedGroupKey] = {}
    for manifest in manifests:
        contracts = manifest.get("contracts")
        if not isinstance(contracts, list):
            raise DerivationError(
                "introspection manifest has no `contracts` list -- refusing to "
                "derive a declared set from an unrecognised payload shape"
            )
        for contract in contracts:
            if not isinstance(contract, Mapping):
                continue
            event_bus = contract.get("event_bus")
            if not isinstance(event_bus, Mapping):
                continue
            if event_bus.get("plugin_managed"):
                continue
            topics = event_bus.get("subscribe_topics")
            if not isinstance(topics, list) or not topics:
                continue
            name = contract.get("name")
            package = contract.get("package_name")
            version = _format_contract_version(contract.get("contract_version"))
            if not isinstance(name, str) or not isinstance(package, str):
                raise DerivationError(
                    "introspection manifest carries a contract with no name/"
                    f"package_name: {contract!r:.200}"
                )
            base = compute_base_group_id(
                env=env, service=package, node_name=name, version=version
            )
            for topic in topics:
                if not isinstance(topic, str) or not topic:
                    continue
                derived = DerivedGroupKey(
                    base_group_id=base,
                    topic=topic,
                    contract_name=name,
                    package=package,
                    contract_version=version,
                )
                keys[derived.key] = derived
    if not keys:
        raise DerivationError(
            "derived zero consumer groups from the introspection manifest(s) -- "
            "the refreshed image declares no subscribing contract, which is not "
            "a state this lane can be healthy in"
        )
    return tuple(sorted(keys.values(), key=lambda k: k.key))


def _format_contract_version(value: object) -> str:
    """``{major, minor, patch}`` -> ``"1.2.0"``; a plain string passes through."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        try:
            return f"{value['major']}.{value['minor']}.{value['patch']}"
        except KeyError as exc:
            raise DerivationError(
                f"contract_version {value!r} is missing {exc}"
            ) from exc
    raise DerivationError(f"unrecognised contract_version payload: {value!r}")


# ─── Live-broker reconciliation ─────────────────────────────────────────────


def parse_group_list(stdout: str) -> dict[str, str]:
    """Parse ``rpk group list`` (``BROKER  GROUP  STATE``) into ``{group: state}``.

    One call answers "which groups exist, and in what state" for the whole
    broker, so the gate never has to ``describe`` a name to find out whether it
    is real. That distinction is the entire defect this module removes:
    ``describe`` on a name that is not a group answers ``Dead``, which is
    indistinguishable from a group that genuinely died.
    """
    groups: dict[str, str] = {}
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0] == "BROKER":
            continue
        groups[fields[1]] = fields[2]
    return groups


@dataclass(frozen=True)
class GroupDescription:
    """``STATE`` / ``MEMBERS`` / ``TOTAL-LAG`` read off ``rpk group describe``."""

    state: str | None
    members: int | None
    total_lag: int | None
    error: str | None = None


def parse_group_describe(stdout: str) -> GroupDescription:
    state: str | None = None
    members: int | None = None
    total_lag: int | None = None
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        if fields[0] == "STATE" and state is None:
            state = fields[1]
        elif fields[0] == "MEMBERS" and members is None:
            members = _safe_int(fields[1])
        elif fields[0] == "TOTAL-LAG" and total_lag is None:
            total_lag = _safe_int(fields[1])
    return GroupDescription(state=state, members=members, total_lag=total_lag)


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


class DescribeFn(Protocol):
    def __call__(self, group: str) -> GroupDescription: ...


# States a group may hold and still be healthy. ``Empty`` -- registered with the
# coordinator, subscription known, zero connected members -- is the NORMAL idle
# state of a demand-driven consumer and was already treated as healthy before
# OMN-15837 (the OMN-14873 canary proved the alternative produced false
# rollbacks). Deliberately unchanged here: 55 of the 611 contract-derived groups
# on the live lane are ``Empty`` and 21 of those carry non-zero lag, all of them
# in-runtime projections superseded by the OMN-17562 standalone writers. Scoring
# idle-with-backlog as a failure would simply have replaced one false-failure
# generator with another.
HEALTHY_STATES = frozenset({"Stable", "Empty"})

# A group the broker does not know at all. ``rpk group describe`` answers this
# for a nonexistent name too -- which is why existence is decided from
# ``rpk group list``, never from a describe.
DEAD_STATE = "Dead"


@dataclass
class GroupFinding:
    """One evaluated group: what it is, what the broker said, how it scored."""

    group: str
    origin: str  # "contract" | "non_contract"
    state: str | None
    classification: str
    members: int | None = None
    total_lag: int | None = None
    detail: str = ""
    failed: bool = False
    contract: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "origin": self.origin,
            "state": self.state,
            "classification": self.classification,
            "members": self.members,
            "total_lag": self.total_lag,
            "detail": self.detail,
            "failed": self.failed,
            "contract": self.contract,
        }


@dataclass
class ConsumerGroupAudit:
    """Reconciliation of the derived declared set against the live broker."""

    env: str
    derived_total: int = 0
    derived_live: int = 0
    min_coverage: float = 0.0
    findings: list[GroupFinding] = field(default_factory=list)
    absent_identities: list[str] = field(default_factory=list)
    retired_identities: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if self.derived_total == 0:
            return 0.0
        return self.derived_live / self.derived_total

    @property
    def coverage_ok(self) -> bool:
        return self.derived_total > 0 and self.coverage >= self.min_coverage

    @property
    def failures(self) -> list[GroupFinding]:
        return [f for f in self.findings if f.failed]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.failures and self.coverage_ok

    def to_dict(self) -> dict[str, object]:
        return {
            "env": self.env,
            "derived_total": self.derived_total,
            "derived_live": self.derived_live,
            "coverage": round(self.coverage, 4),
            "min_coverage": self.min_coverage,
            "coverage_ok": self.coverage_ok,
            "absent_identities": self.absent_identities,
            "retired_identities": self.retired_identities,
            "failures": [f.to_dict() for f in self.failures],
            "findings_count": len(self.findings),
            "errors": self.errors,
            "ok": self.ok,
        }


def reconcile(
    *,
    env: str,
    derived: Sequence[DerivedGroupKey],
    non_contract: Iterable[NonContractGroup],
    live: Mapping[str, str],
    describe: DescribeFn,
    min_coverage: float,
) -> ConsumerGroupAudit:
    """Score the derived declared set + the allowlist against the live broker.

    Scoring rules, and why each is what it is:

    * A derived identity with at least one live group (any instance) whose state
      is ``Stable``/``Empty`` -> healthy.
    * A derived identity whose live group reads ``Dead`` -> the identity is
      described. ``MEMBERS 0 / TOTAL-LAG 0`` is a RETIRED identity: the consumer
      that owned it is gone and it left nothing behind. That is what a version
      bump and a re-home both look like, it is not a fault, and it is logged
      rather than failed. Non-zero lag with zero members is the opposite claim --
      the group lost its members while still holding a backlog -- and FAILS.
    * A derived identity with NO live group is reported as absent and counted
      against coverage, not failed individually. It cannot be described
      meaningfully (without an instance discriminator the probe name is not a
      real group and always answers ``Dead / 0 / 0``), and a handful are
      expected: contracts wired through the core-runtime single-owner path or
      whose topics are owned elsewhere. The COVERAGE FLOOR is what turns a real
      wiring collapse into a failure.
    * An allowlisted non-contract group MUST be live and healthy. Those entries
      exist precisely because nothing derives them, so their absence is the
      silent-wiring-death signal for the standalone projection writers.
    """
    audit = ConsumerGroupAudit(env=env, min_coverage=min_coverage)

    if not any(group.startswith(f"{env}.") for group in live):
        audit.errors.append(
            f"no live consumer group carries the derived env prefix {env!r} "
            f"({len(live)} groups on the broker) -- the lane name and the "
            "runtime's minted group prefix disagree, so the derived set could "
            "not be matched against anything"
        )
        return audit

    live_index: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for group, state in live.items():
        base, _instance, topic = split_group_name(group)
        if topic is None:
            continue
        live_index.setdefault((base, topic), []).append((group, state))

    audit.derived_total = len(derived)
    for key in derived:
        matches = live_index.get(key.key)
        if not matches:
            audit.absent_identities.append(key.probe_name)
            continue
        audit.derived_live += 1
        for group, state in matches:
            audit.findings.append(
                _score(
                    group=group,
                    state=state,
                    origin="contract",
                    contract=f"{key.package}/{key.contract_name}@{key.contract_version}",
                    describe=describe,
                    audit=audit,
                    absent_is_failure=False,
                )
            )

    for declared in non_contract:
        declared_state = live.get(declared.name)
        if declared_state is None:
            audit.findings.append(
                GroupFinding(
                    group=declared.name,
                    origin="non_contract",
                    state=None,
                    classification="absent",
                    detail=(
                        "declared non-contract group is not registered with the "
                        f"broker at all (source: {declared.source})"
                    ),
                    failed=True,
                )
            )
            continue
        audit.findings.append(
            _score(
                group=declared.name,
                state=declared_state,
                origin="non_contract",
                contract=None,
                describe=describe,
                audit=audit,
                absent_is_failure=True,
            )
        )

    return audit


def _score(
    *,
    group: str,
    state: str,
    origin: str,
    contract: str | None,
    describe: DescribeFn,
    audit: ConsumerGroupAudit,
    absent_is_failure: bool,
) -> GroupFinding:
    if state in HEALTHY_STATES:
        return GroupFinding(
            group=group,
            origin=origin,
            state=state,
            classification="healthy",
            detail=f"state={state}",
            contract=contract,
        )

    described = describe(group)
    if described.error is not None:
        return GroupFinding(
            group=group,
            origin=origin,
            state=state,
            classification="describe_error",
            detail=described.error,
            failed=True,
            contract=contract,
        )
    members = described.members
    total_lag = described.total_lag
    if members is None or total_lag is None:
        return GroupFinding(
            group=group,
            origin=origin,
            state=state,
            classification="describe_unparsed",
            members=members,
            total_lag=total_lag,
            detail=(
                "rpk group describe returned no readable MEMBERS/TOTAL-LAG -- "
                "refusing to score an unreadable group as healthy"
            ),
            failed=True,
            contract=contract,
        )
    if members == 0 and total_lag == 0:
        if absent_is_failure:
            return GroupFinding(
                group=group,
                origin=origin,
                state=state,
                classification="declared_group_dead",
                members=members,
                total_lag=total_lag,
                detail=(
                    "declared non-contract group is registered but Dead with no "
                    "members and no lag -- its consumer never joined"
                ),
                failed=True,
                contract=contract,
            )
        audit.retired_identities.append(group)
        return GroupFinding(
            group=group,
            origin=origin,
            state=state,
            classification="retired",
            members=members,
            total_lag=total_lag,
            detail=(
                "retired identity: no members and no lag. A version bump or a "
                "re-home leaves exactly this trace; it is not a refresh fault."
            ),
            contract=contract,
        )
    return GroupFinding(
        group=group,
        origin=origin,
        state=state,
        classification="lost_members_with_lag",
        members=members,
        total_lag=total_lag,
        detail=(
            f"group is {state} with members={members} but TOTAL-LAG={total_lag} "
            "-- it lost its consumers while still holding a backlog, which is a "
            "stall, not a retirement"
        ),
        failed=True,
        contract=contract,
    )


# ─── Second derivation source: lane compose KAFKA_CONSUMER_GROUP ────────────


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates Compose's merge tags (``!override``, ``!reset``).

    ``docker-compose.stability-test.yml`` uses them, and ``yaml.safe_load``
    raises ``ConstructorError`` on an unregistered tag -- which the gate would
    correctly, and uselessly, report as a fail-closed derivation error on every
    single refresh. Unknown tags are resolved to their underlying node value; no
    Python object is ever constructed, so this stays as safe as SafeLoader.
    """


def _construct_undefined_as_value(
    loader: yaml.SafeLoader, _tag: str, node: yaml.Node
) -> object:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    return None


_ComposeLoader.add_multi_constructor("", _construct_undefined_as_value)  # type: ignore[no-untyped-call]


def derive_compose_declared_groups(
    compose_path: Path, *, env: str
) -> tuple[NonContractGroup, ...]:
    """Derive the lane's non-contract consumer groups from its compose file.

    The standalone projection writers (OMN-17562) do not go through contract
    auto-wiring: each container is handed its group id literally, as the
    ``KAFKA_CONSUMER_GROUP`` environment value in
    ``docker/docker-compose.<lane>.yml``. That file is the minting authority, so
    it is read as a derivation source rather than copied into an allowlist --
    a writer that is re-homed or renamed changes the compose file in the same
    commit, and the gate follows it without a second edit.

    Only groups carrying the lane's own ``{env}.`` prefix are returned, so a
    shared compose fragment cannot pull another lane's group into this gate.
    """
    try:
        with compose_path.open() as fh:
            data = yaml.load(fh, Loader=_ComposeLoader) or {}  # noqa: S506
    except (OSError, yaml.YAMLError) as exc:
        raise DerivationError(
            f"cannot read lane compose file {compose_path}: {exc}"
        ) from exc
    services = data.get("services") if isinstance(data, Mapping) else None
    if not isinstance(services, Mapping):
        raise DerivationError(
            f"lane compose file {compose_path} has no `services` mapping"
        )
    derived: dict[str, NonContractGroup] = {}
    for service_name, service in services.items():
        if not isinstance(service, Mapping):
            continue
        value = _compose_env_value(service.get("environment"), "KAFKA_CONSUMER_GROUP")
        if value is None:
            continue
        if "${" in value:
            raise DerivationError(
                f"{compose_path}: service {service_name!r} declares a "
                "KAFKA_CONSUMER_GROUP that is not a literal "
                f"({value!r}); the gate will not guess an interpolated group id"
            )
        if not value.startswith(f"{env}."):
            continue
        derived[value] = NonContractGroup(
            name=value,
            source=(
                f"{compose_path.name} service {service_name} "
                "KAFKA_CONSUMER_GROUP (standalone projection writer, minted "
                "outside contract auto-wiring)"
            ),
        )
    return tuple(sorted(derived.values(), key=lambda g: g.name))


def _compose_env_value(environment: object, key: str) -> str | None:
    """Read one env value from either compose ``environment`` spelling."""
    if isinstance(environment, Mapping):
        value = environment.get(key)
        return str(value) if value is not None else None
    if isinstance(environment, list):
        prefix = f"{key}="
        for entry in environment:
            if isinstance(entry, str) and entry.startswith(prefix):
                return entry[len(prefix) :].strip().strip('"').strip("'")
    return None
