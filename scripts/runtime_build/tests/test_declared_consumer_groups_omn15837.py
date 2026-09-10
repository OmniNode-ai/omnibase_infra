# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the derived declared consumer-group set [OMN-15837].

The defect these pin: ``consumer_groups_stability.yaml`` used to carry FULL,
version-pinned Kafka consumer-group names. A contract version bump or a re-home
invalidated an entry, ``rpk group describe`` answered ``Dead / 0 / 0`` for the
now-nonexistent name, and the stability-lane health gate rolled a HEALTHY
refresh back on it -- twice (fd4a84b1c, then OMN-16753 on 2026-09-08).

No Docker daemon, no broker, no live lane: the manifest payloads and the ``rpk``
outputs are fixtures shaped exactly like the real ones read off the
stability-test lane on 2026-09-08.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "declared_consumer_groups", _SCRIPT_DIR / "declared_consumer_groups.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["declared_consumer_groups"] = _MOD
_SPEC.loader.exec_module(_MOD)

DerivationError = _MOD.DerivationError
GroupDescription = _MOD.GroupDescription
NonContractGroup = _MOD.NonContractGroup
compute_base_group_id = _MOD.compute_base_group_id
derive_compose_declared_groups = _MOD.derive_compose_declared_groups
derive_expected_keys = _MOD.derive_expected_keys
load_non_contract_groups = _MOD.load_non_contract_groups
parse_group_describe = _MOD.parse_group_describe
parse_group_list = _MOD.parse_group_list
reconcile = _MOD.reconcile
split_group_name = _MOD.split_group_name

LANE = "stability-test"

# The two identities the OMN-16753 refresh was rolled back on, at the versions
# that were pinned in the retired YAML and at the versions that were actually
# live. Copied verbatim from the lane, not reconstructed.
ROUTING_REDUCER_STALE = (
    "stability-test.omnimarket.node_delegation_routing_reducer.consume.0.3.0"
    ".__i.stability-test-main.__t.onex.cmd.omnibase-infra.delegation-routing-request.v1"
)
ROUTING_REDUCER_LIVE = (
    "stability-test.omnimarket.node_delegation_routing_reducer.consume.0.4.0"
    ".__i.stability-test-main.__t.onex.cmd.omnibase-infra.delegation-routing-request.v1"
)
PROJECTION_DELEGATION_STALE = (
    "stability-test.omnimarket.projection_delegation.consume.1.0.0"
    ".__i.stability-test-main.__t.onex.evt.omnibase-infra.delegation-completed.v1"
)
DELEGATION_WRITER = "stability-test.omnimarket-projections.delegation-writer.consume.v1"


def _contract(
    name: str,
    package: str,
    version: tuple[int, int, int],
    topics: list[str],
    *,
    plugin_managed: bool = False,
) -> dict[str, object]:
    major, minor, patch = version
    return {
        "name": name,
        "package_name": package,
        "contract_version": {"major": major, "minor": minor, "patch": patch},
        "event_bus": {
            "subscribe_topics": topics,
            "publish_topics": [],
            "plugin_managed": plugin_managed,
        },
    }


def _manifest(contracts: list[dict[str, object]], profile: str) -> dict[str, object]:
    return {"contracts": contracts, "errors": [], "runtime_profile": profile}


def _live(*groups: tuple[str, str]) -> dict[str, str]:
    return dict(groups)


def _describe_never_called(group: str) -> GroupDescription:  # pragma: no cover
    raise AssertionError(f"describe should not have been called for {group}")


# ─── Derivation from a fixture manifest ─────────────────────────────────────


def test_derives_group_identity_from_contract_identity():
    """The derived base id is exactly what the runtime mints from the contract."""
    manifest = _manifest(
        [
            _contract(
                "node_registration_orchestrator",
                "omnibase_infra",
                (1, 1, 1),
                ["onex.evt.platform.node-heartbeat.v1"],
            )
        ],
        "main",
    )
    keys = derive_expected_keys([manifest], env=LANE)
    assert len(keys) == 1
    assert keys[0].base_group_id == (
        "stability-test.omnibase_infra.node_registration_orchestrator.consume.1.1.1"
    )
    assert keys[0].topic == "onex.evt.platform.node-heartbeat.v1"


def test_derivation_reproduces_a_real_live_group_name():
    """Golden check against a name read off the live broker on 2026-09-08."""
    live_name = (
        "stability-test.omnibase_infra.node_ledger_projection_compute.consume.1.2.0"
        ".__i.stability-test-main.__t.onex.evt.platform.node-heartbeat.v1"
    )
    base, instance, topic = split_group_name(live_name)
    assert instance == "stability-test-main"
    assert topic == "onex.evt.platform.node-heartbeat.v1"
    assert base == compute_base_group_id(
        env=LANE,
        service="omnibase_infra",
        node_name="node_ledger_projection_compute",
        version="1.2.0",
    )


def test_one_key_per_subscribe_topic_across_both_profile_manifests():
    main = _manifest(
        [_contract("node_a", "omnibase_infra", (1, 0, 0), ["t.one", "t.two"])], "main"
    )
    effects = _manifest(
        [_contract("node_b", "omnimarket", (2, 1, 0), ["t.three"])], "effects"
    )
    keys = derive_expected_keys([main, effects], env=LANE)
    assert {(k.contract_name, k.topic) for k in keys} == {
        ("node_a", "t.one"),
        ("node_a", "t.two"),
        ("node_b", "t.three"),
    }


def test_plugin_managed_contracts_are_not_expected_to_mint_a_group():
    """Auto-wiring opens no Kafka subscription for these; expecting one is phantom."""
    manifest = _manifest(
        [
            _contract("node_a", "omnimarket", (1, 0, 0), ["t.one"]),
            _contract(
                "node_delegation_orchestrator",
                "omnimarket",
                (0, 6, 0),
                ["t.two", "t.three"],
                plugin_managed=True,
            ),
        ],
        "main",
    )
    keys = derive_expected_keys([manifest], env=LANE)
    assert [k.contract_name for k in keys] == ["node_a"]


def test_instance_discriminator_is_not_derived_and_any_instance_satisfies():
    """The `.__i.` value is a per-container property, never a contract property."""
    manifest = _manifest(
        [_contract("node_a", "omnimarket", (1, 0, 0), ["t.one"])], "main"
    )
    keys = derive_expected_keys([manifest], env=LANE)
    group = f"{keys[0].base_group_id}.__i.stability-test-worker.__t.t.one"
    audit = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(),
        live=_live((group, "Stable")),
        describe=_describe_never_called,
        min_coverage=1.0,
    )
    assert audit.ok
    assert audit.derived_live == 1


# ─── Version bump ───────────────────────────────────────────────────────────


def test_version_bump_is_followed_without_a_repin():
    """0.3.0 -> 0.4.0: the derived set moves with the image, the gate stays green.

    This is the OMN-16753 rollback, replayed. The retired static list pinned
    0.3.0; only 0.4.0 was ever live.
    """
    manifest = _manifest(
        [
            _contract(
                "node_delegation_routing_reducer",
                "omnimarket",
                (0, 4, 0),
                ["onex.cmd.omnibase-infra.delegation-routing-request.v1"],
            )
        ],
        "main",
    )
    keys = derive_expected_keys([manifest], env=LANE)
    audit = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(),
        live=_live((ROUTING_REDUCER_LIVE, "Stable")),
        describe=_describe_never_called,
        min_coverage=1.0,
    )
    assert audit.ok, audit.to_dict()
    assert ROUTING_REDUCER_STALE not in {f.group for f in audit.findings}


def test_the_stale_pinned_version_is_never_probed_at_all():
    """The old pin cannot fail the gate because it is no longer part of the set."""
    manifest = _manifest(
        [
            _contract(
                "node_delegation_routing_reducer",
                "omnimarket",
                (0, 4, 0),
                ["onex.cmd.omnibase-infra.delegation-routing-request.v1"],
            )
        ],
        "main",
    )
    keys = derive_expected_keys([manifest], env=LANE)
    assert all("consume.0.3.0" not in k.base_group_id for k in keys)


# ─── Re-home ────────────────────────────────────────────────────────────────


def test_rehome_out_of_the_runtime_does_not_fail_the_gate():
    """projection_delegation left the runtime for a standalone writer (OMN-17562).

    The contract no longer declares the subscription, so nothing derives the old
    identity; the writer's group is derived from the lane compose file instead.
    """
    manifest = _manifest(
        [_contract("node_other", "omnimarket", (1, 0, 0), ["t.one"])], "main"
    )
    keys = derive_expected_keys([manifest], env=LANE)
    writer = NonContractGroup(name=DELEGATION_WRITER, source="compose fixture")
    audit = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(writer,),
        live=_live(
            (f"{keys[0].base_group_id}.__i.stability-test-main.__t.t.one", "Stable"),
            (DELEGATION_WRITER, "Stable"),
        ),
        describe=_describe_never_called,
        min_coverage=1.0,
    )
    assert audit.ok, audit.to_dict()
    assert PROJECTION_DELEGATION_STALE not in {f.group for f in audit.findings}


def test_rehomed_writer_group_is_derived_from_the_lane_compose_file(tmp_path: Path):
    compose = tmp_path / "docker-compose.stability-test.yml"
    compose.write_text(
        "services:\n"
        "  projection-delegation-writer:\n"
        "    environment:\n"
        f'      KAFKA_CONSUMER_GROUP: "{DELEGATION_WRITER}"\n'
        "  projection-live-events-writer:\n"
        "    environment:\n"
        "      - KAFKA_CONSUMER_GROUP=stability-test.omnimarket-projections"
        ".live-events-writer.consume.v1\n"
        "  some-other-lane-service:\n"
        "    environment:\n"
        '      KAFKA_CONSUMER_GROUP: "local.omnimarket-projections.x.consume.v1"\n'
        "  no-consumer-group-service:\n"
        "    image: busybox\n"
    )
    groups = derive_compose_declared_groups(compose, env=LANE)
    assert [g.name for g in groups] == [
        DELEGATION_WRITER,
        "stability-test.omnimarket-projections.live-events-writer.consume.v1",
    ]
    assert all("docker-compose.stability-test.yml" in g.source for g in groups)


def test_a_missing_writer_group_is_a_hard_failure():
    """Nothing derives a writer from a contract, so its absence IS wiring death."""
    manifest = _manifest(
        [_contract("node_other", "omnimarket", (1, 0, 0), ["t.one"])], "main"
    )
    keys = derive_expected_keys([manifest], env=LANE)
    audit = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(
            NonContractGroup(name=DELEGATION_WRITER, source="compose fixture"),
        ),
        live=_live(
            (f"{keys[0].base_group_id}.__i.stability-test-main.__t.t.one", "Stable")
        ),
        describe=_describe_never_called,
        min_coverage=1.0,
    )
    assert not audit.ok
    assert [f.classification for f in audit.failures] == ["absent"]


# ─── Retired vs. lost-with-lag ──────────────────────────────────────────────


def _fixed_describe(members: int, lag: int):
    def _describe(group: str) -> GroupDescription:
        return GroupDescription(state="Dead", members=members, total_lag=lag)

    return _describe


def test_dead_with_no_members_and_no_lag_is_a_retired_identity_not_a_failure():
    manifest = _manifest(
        [_contract("node_a", "omnimarket", (1, 0, 0), ["t.one"])], "main"
    )
    keys = derive_expected_keys([manifest], env=LANE)
    group = f"{keys[0].base_group_id}.__i.stability-test-main.__t.t.one"
    audit = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(),
        live=_live((group, "Dead")),
        describe=_fixed_describe(members=0, lag=0),
        min_coverage=1.0,
    )
    assert audit.ok, audit.to_dict()
    assert audit.retired_identities == [group]
    assert [f.classification for f in audit.findings] == ["retired"]


def test_members_lost_while_lag_retained_is_a_failure():
    manifest = _manifest(
        [_contract("node_a", "omnimarket", (1, 0, 0), ["t.one"])], "main"
    )
    keys = derive_expected_keys([manifest], env=LANE)
    group = f"{keys[0].base_group_id}.__i.stability-test-main.__t.t.one"
    audit = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(),
        live=_live((group, "Dead")),
        describe=_fixed_describe(members=0, lag=1508),
        min_coverage=1.0,
    )
    assert not audit.ok
    assert [f.classification for f in audit.failures] == ["lost_members_with_lag"]
    assert audit.failures[0].total_lag == 1508


def test_empty_with_lag_stays_healthy_idle_not_a_new_false_failure():
    """55 of 611 live derived groups are Empty and 21 of those carry lag.

    They are in-runtime projections superseded by the OMN-17562 writers.
    Scoring idle-with-backlog as a failure would replace one false-failure
    generator with another, so ``Empty`` stays healthy and is never described.
    """
    manifest = _manifest(
        [_contract("node_a", "omnimarket", (1, 0, 0), ["t.one"])], "main"
    )
    keys = derive_expected_keys([manifest], env=LANE)
    group = f"{keys[0].base_group_id}.__i.stability-test-main.__t.t.one"
    audit = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(),
        live=_live((group, "Empty")),
        describe=_describe_never_called,
        min_coverage=1.0,
    )
    assert audit.ok
    assert [f.classification for f in audit.findings] == ["healthy"]


def test_unreadable_describe_is_never_scored_healthy():
    manifest = _manifest(
        [_contract("node_a", "omnimarket", (1, 0, 0), ["t.one"])], "main"
    )
    keys = derive_expected_keys([manifest], env=LANE)
    group = f"{keys[0].base_group_id}.__i.stability-test-main.__t.t.one"
    audit = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(),
        live=_live((group, "Dead")),
        describe=lambda g: GroupDescription(
            state=None, members=None, total_lag=None, error="rpk group describe failed"
        ),
        min_coverage=1.0,
    )
    assert not audit.ok
    assert [f.classification for f in audit.failures] == ["describe_error"]


# ─── Coverage floor ─────────────────────────────────────────────────────────


def test_absent_identities_are_counted_against_the_coverage_floor():
    contracts = [
        _contract(f"node_{i}", "omnimarket", (1, 0, 0), ["t.one"]) for i in range(10)
    ]
    keys = derive_expected_keys([_manifest(contracts, "main")], env=LANE)
    live = _live(
        *(
            (f"{k.base_group_id}.__i.stability-test-main.__t.t.one", "Stable")
            for k in keys[:8]
        )
    )
    healthy = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(),
        live=live,
        describe=_describe_never_called,
        min_coverage=0.75,
    )
    assert healthy.ok
    assert healthy.coverage == pytest.approx(0.8)
    strict = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(),
        live=live,
        describe=_describe_never_called,
        min_coverage=0.9,
    )
    assert not strict.ok
    assert len(strict.absent_identities) == 2


# ─── Fail closed ────────────────────────────────────────────────────────────


def test_no_manifest_contracts_key_fails_closed():
    with pytest.raises(DerivationError, match="contracts"):
        derive_expected_keys([{"runtime_profile": "main"}], env=LANE)


def test_zero_derived_groups_fails_closed_rather_than_passing_vacuously():
    with pytest.raises(DerivationError, match="derived zero consumer groups"):
        derive_expected_keys([_manifest([], "main")], env=LANE)


def test_env_prefix_mismatch_fails_closed_instead_of_reporting_all_absent():
    manifest = _manifest(
        [_contract("node_a", "omnimarket", (1, 0, 0), ["t.one"])], "main"
    )
    keys = derive_expected_keys([manifest], env=LANE)
    audit = reconcile(
        env=LANE,
        derived=keys,
        non_contract=(),
        live=_live(("dev.omnimarket.node_a.consume.1.0.0.__t.t.one", "Stable")),
        describe=_describe_never_called,
        min_coverage=0.0,
    )
    assert not audit.ok
    assert audit.errors and "env prefix" in audit.errors[0]


def test_retired_hand_pinned_key_is_rejected_rather_than_silently_ignored(
    tmp_path: Path,
):
    path = tmp_path / "consumer_groups_stability.yaml"
    path.write_text("consumer_groups:\n  - name: stability-test.x.y.consume.1.0.0\n")
    with pytest.raises(DerivationError, match="retired `consumer_groups:` key"):
        load_non_contract_groups(path)


def test_non_contract_entry_without_a_source_is_rejected(tmp_path: Path):
    path = tmp_path / "declared.yaml"
    path.write_text("non_contract_groups:\n  - name: stability-test.some.group\n")
    with pytest.raises(DerivationError, match="has no `source`"):
        load_non_contract_groups(path)


def test_missing_non_contract_key_fails_closed(tmp_path: Path):
    path = tmp_path / "declared.yaml"
    path.write_text("something_else: []\n")
    with pytest.raises(DerivationError, match="non_contract_groups"):
        load_non_contract_groups(path)


def test_empty_non_contract_list_is_a_valid_explicit_declaration(tmp_path: Path):
    path = tmp_path / "declared.yaml"
    path.write_text("non_contract_groups: []\n")
    assert load_non_contract_groups(path) == ()


def test_interpolated_compose_group_fails_closed(tmp_path: Path):
    compose = tmp_path / "docker-compose.stability-test.yml"
    compose.write_text(
        "services:\n"
        "  writer:\n"
        "    environment:\n"
        '      KAFKA_CONSUMER_GROUP: "${LANE}.omnimarket-projections.x.consume.v1"\n'
    )
    with pytest.raises(DerivationError, match="not a literal"):
        derive_compose_declared_groups(compose, env=LANE)


def test_unreadable_compose_file_fails_closed(tmp_path: Path):
    with pytest.raises(DerivationError, match="cannot read lane compose file"):
        derive_compose_declared_groups(tmp_path / "absent.yml", env=LANE)


# ─── rpk output parsing ─────────────────────────────────────────────────────


def test_parse_group_list_reads_state_for_every_group_in_one_call():
    stdout = (
        "BROKER  GROUP                      STATE\n"
        "0       stability-test.a.b.consume.1.0.0.__t.t.one   Stable\n"
        "0       stability-test.c.d.consume.1.0.0.__t.t.two   Empty\n"
    )
    assert parse_group_list(stdout) == {
        "stability-test.a.b.consume.1.0.0.__t.t.one": "Stable",
        "stability-test.c.d.consume.1.0.0.__t.t.two": "Empty",
    }


def test_parse_group_describe_reads_state_members_and_lag():
    stdout = (
        "GROUP        stability-test.x\n"
        "COORDINATOR  0\n"
        "STATE        Empty\n"
        "BALANCER     \n"
        "MEMBERS      0\n"
        "TOTAL-LAG    1508\n"
        "\n"
        "TOPIC   PARTITION  CURRENT-OFFSET  LOG-START-OFFSET  LOG-END-OFFSET  LAG\n"
        "t.one   0          37              40                277             240\n"
    )
    described = parse_group_describe(stdout)
    assert described.state == "Empty"
    assert described.members == 0
    assert described.total_lag == 1508


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("base.__i.inst.__t.topic.v1", ("base", "inst", "topic.v1")),
        ("base.__t.topic.v1", ("base", None, "topic.v1")),
        ("base-only", ("base-only", None, None)),
    ],
)
def test_split_group_name(group: str, expected: tuple[str, str | None, str | None]):
    assert split_group_name(group) == expected
