# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Two deploy-agent instances, one control topic, one acceptor per command (OMN-19506).

Task B3 of the second-deploy-slot plan (epic OMN-19500). The ticket's
acceptance criteria, by test name:

* AC1 ``routed_elsewhere``: a dev command requested by omnimarket is accepted by
  the dev-202 instance and skipped by the .201 instance, with no rejection
  published.
* AC2 ``routing_default``: a requester no route names goes to the declared
  default, and a table with no default refuses agent start.
* AC3 ``consumer_group_per_instance``: each instance subscribes with the group
  its table entry declares, and no two instances share one.

AC4 is the TLA+ model and its TLC logs, attached to the ticket before the
first commit. Its three counterexamples are why the route is read at the
command's own ref (``test_routed_elsewhere_follows_the_table_at_the_command_ref``),
why each instance has its own group, and why the skip publishes nothing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from deploy_agent import consumer as consumer_mod
from deploy_agent.consumer import (
    LEGACY_CONSUMER_GROUP,
    DeployConsumer,
    consumer_group_for,
)
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested
from deploy_agent.routing import (
    ENV_INSTANCE,
    PINNED_DEFAULT_INSTANCE,
    ROUTING_TABLE_RELPATH,
    DeployRouter,
    GitTableAtRef,
    ModelRoutingTable,
    RoutingTableError,
    load_routing_table,
    parse_routing_table,
    resolve_instance,
)

pytestmark = pytest.mark.unit

SHA_A = "a" * 40
SHA_B = "b" * 40

TABLE_WITH_OMNIMARKET_ROUTE = """
default_instance: dev-201
instances:
  dev-201:
    hostnames: [omninode-pc]
    consumer_group: onex-deploy-agent
  dev-202:
    hostnames: [omnipc2]
    consumer_group: onex-deploy-agent-dev-202
routes:
  - runtime_lane: dev
    requester_repository: omnimarket
    instance: dev-202
"""

TABLE_WITHOUT_ROUTES = (
    TABLE_WITH_OMNIMARKET_ROUTE.split("routes:", maxsplit=1)[0] + "routes: []\n"
)


def _command(
    requested_by: str, git_ref: str = SHA_A, lane: str = "dev"
) -> ModelRebuildRequested:
    return ModelRebuildRequested.model_validate(
        {
            "correlation_id": str(uuid4()),
            "requested_by": requested_by,
            "scope": "full",
            "runtime_lane": lane,
            "build_source": "workspace",
            "git_ref": git_ref,
        }
    )


def _router(instance: str, tables_at_ref: dict[str, str | None]) -> DeployRouter:
    table = parse_routing_table(TABLE_WITH_OMNIMARKET_ROUTE)

    def at_ref(sha: str) -> ModelRoutingTable | None:
        if sha not in tables_at_ref:
            raise RoutingTableError(f"commit {sha} is not in the clone")
        text = tables_at_ref[sha]
        return None if text is None else parse_routing_table(text)

    return DeployRouter(table, table.instances[instance], at_ref)


def _message(cmd: ModelRebuildRequested, offset: int = 100) -> SimpleNamespace:
    payload = cmd.model_dump(mode="json") | {"_signature": "a" * 64}
    return SimpleNamespace(
        value=payload, topic="t", partition=0, offset=offset, key=None, timestamp=None
    )


def _consumer(router: DeployRouter | None) -> DeployConsumer:
    consumer = DeployConsumer.__new__(DeployConsumer)
    consumer.consumer = Mock()
    consumer.job_store = Mock()
    consumer.job_store.has_active_job.return_value = False
    consumer.job_store.is_duplicate.return_value = False
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.DEV})
    consumer.self_update_hook = lambda rewind: None
    consumer.ancestry_resolver = None
    consumer.running_build = None
    consumer.ref_resolver = None
    consumer.tracking_ref = None
    consumer.notices = []
    consumer.on_rejected = consumer.notices.append
    consumer.router = router
    return consumer


def _process(
    consumer: DeployConsumer, cmd: ModelRebuildRequested
) -> tuple[ModelRebuildRequested | None, str | None]:
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        return consumer._process_message(_message(cmd))


def _committed(consumer: DeployConsumer) -> list[int]:
    return [
        meta.offset
        for call in consumer.consumer.commit.call_args_list
        for meta in call.args[0].values()
    ]


# --- AC1 ----------------------------------------------------------------------


def test_routed_elsewhere_the_201_instance_skips_an_omnimarket_command() -> None:
    consumer = _consumer(_router("dev-201", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}))
    accepted, reason = _process(consumer, _command("gha/omnimarket/pr-2851"))

    assert (accepted, reason) == (None, None)
    assert consumer.notices == [], "a skip publishes nothing"
    consumer.job_store.accept.assert_not_called()
    assert _committed(consumer) == [101], "the skip commits past the record"


def test_routed_elsewhere_the_202_instance_accepts_the_same_command() -> None:
    consumer = _consumer(_router("dev-202", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}))
    cmd = _command("gha/omnimarket/pr-2851")
    accepted, reason = _process(consumer, cmd)

    assert reason is None
    assert accepted == cmd
    assert consumer.notices == []


def test_routed_elsewhere_the_skip_runs_before_busy_and_duplicate() -> None:
    """A busy or duplicate refusal publishes; for another instance's command
    that would race its acceptance, so routing decides first."""
    consumer = _consumer(_router("dev-201", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}))
    consumer.job_store.has_active_job.return_value = True
    consumer.job_store.is_duplicate.return_value = True

    assert _process(consumer, _command("gha/omnimarket/pr-1")) == (None, None)
    assert consumer.notices == []


def test_routed_elsewhere_follows_the_table_at_the_command_ref() -> None:
    """The MC_loaded_table counterexample: each instance routes with the table
    at the command's ref, so the loaded table (which has the omnimarket route
    in both routers here) does not decide a command whose ref says otherwise."""
    tables = {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE, SHA_B: TABLE_WITHOUT_ROUTES}
    old_ref = _command("gha/omnimarket/pr-1", git_ref=SHA_B)
    acceptors = [
        name
        for name in ("dev-201", "dev-202")
        if _process(_consumer(_router(name, tables)), old_ref)[0] is not None
    ]
    assert acceptors == ["dev-201"]


def test_routed_elsewhere_exactly_one_instance_accepts_each_command() -> None:
    tables = {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}
    for requester in (
        "gha/omnimarket/pr-7",
        "gha/omnibase_infra/pr-7",
        "gha/omninode_infra/push-7",
        "lab-health-triage",
    ):
        cmd = _command(requester)
        acceptors = [
            name
            for name in ("dev-201", "dev-202")
            if _process(_consumer(_router(name, tables)), cmd)[0] is not None
        ]
        assert len(acceptors) == 1, (requester, acceptors)


def test_routed_elsewhere_lookahead_never_folds_another_instances_command() -> None:
    consumer = _consumer(_router("dev-201", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}))
    with patch("deploy_agent.consumer.verify_command", return_value=True):
        ours = consumer._decode_for_lookahead(
            _message(_command("gha/omnibase_infra/pr-1"))
        )
        theirs = consumer._decode_for_lookahead(
            _message(_command("gha/omnimarket/pr-1"))
        )
    assert ours is not None
    assert theirs is None


def test_routed_elsewhere_a_lane_the_table_does_not_route_is_untouched() -> None:
    consumer = _consumer(_router("dev-202", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}))
    consumer.allowed_lanes = frozenset({EnumRuntimeLane.STABILITY_TEST})
    cmd = _command("gha/omnibase_infra/pr-1", lane="stability-test")
    accepted, _reason = _process(consumer, cmd)
    assert accepted == cmd


# --- AC2 ----------------------------------------------------------------------


def test_routing_default_takes_a_requester_no_route_names() -> None:
    router = _router("dev-201", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE})
    decision = router.decide(_command("gha/omnibase_infra/pr-4084"))
    assert decision.instance == "dev-201"


@pytest.mark.parametrize(
    ("git_ref", "tables", "basis"),
    [
        ("origin/dev", {}, "not a sha"),
        (SHA_A, {SHA_A: None}, "no table at"),
        (SHA_A, {}, "unreadable"),
    ],
)
def test_routing_default_when_the_ref_names_no_table(
    git_ref: str, tables: dict[str, str | None], basis: str
) -> None:
    decision = _router("dev-202", tables).decide(
        _command("gha/omnimarket/pr-1", git_ref=git_ref)
    )
    assert decision.instance == PINNED_DEFAULT_INSTANCE
    assert basis in decision.basis


def test_routing_default_a_table_with_no_default_refuses() -> None:
    text = TABLE_WITH_OMNIMARKET_ROUTE.replace("default_instance: dev-201\n", "")
    with pytest.raises(RoutingTableError, match="no default_instance"):
        parse_routing_table(text)


def test_routing_default_refuses_agent_start_without_a_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The agent builds its router at construction; a table without a default
    raises there, before the health port binds or a record is polled."""
    from deploy_agent import agent as agent_mod
    from deploy_agent import routing as routing_mod

    table_path = tmp_path / ROUTING_TABLE_RELPATH
    table_path.parent.mkdir(parents=True)
    table_path.write_text(
        TABLE_WITH_OMNIMARKET_ROUTE.replace("default_instance: dev-201\n", ""),
        encoding="utf-8",
    )
    monkeypatch.setattr(routing_mod, "AGENT_CLONE_ROOT", tmp_path)
    with pytest.raises(RoutingTableError, match="no default_instance"):
        agent_mod.build_router_from_env("/nonexistent")


def test_routing_default_must_be_the_pinned_instance() -> None:
    text = TABLE_WITH_OMNIMARKET_ROUTE.replace(
        "default_instance: dev-201", "default_instance: dev-202"
    )
    table = parse_routing_table(text)
    with pytest.raises(RoutingTableError, match="must be 'dev-201'"):
        DeployRouter(table, table.instances["dev-201"], lambda sha: None)


def test_routing_default_the_committed_table_names_the_pinned_default() -> None:
    table = load_routing_table()
    assert table.default_instance == PINNED_DEFAULT_INSTANCE
    # OMN-19543 adds the .200 instance, routed nothing.
    assert set(table.instances) == {"dev-201", "dev-202", "dev-200"}


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("instance: dev-202", "instance: dev-303"), "not declared"),
        (
            (
                "routes:",
                "routes:\n  - {runtime_lane: dev, requester_repository: omnimarket, instance: dev-201}\n",
            ),
            "two routes",
        ),
        (("hostnames: [omnipc2]", "hostnames: [omninode-pc]"), "names two instances"),
    ],
)
def test_routing_default_an_inconsistent_table_refuses(
    mutation: tuple[str, str], match: str
) -> None:
    with pytest.raises(RoutingTableError, match=match):
        parse_routing_table(TABLE_WITH_OMNIMARKET_ROUTE.replace(*mutation, 1))


def test_routing_default_instance_resolves_from_env_then_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_INSTANCE, raising=False)
    table = parse_routing_table(TABLE_WITH_OMNIMARKET_ROUTE)
    assert resolve_instance(table, env={}, hostname="omnipc2").name == "dev-202"
    assert resolve_instance(table, env={}, hostname="OMNINODE-PC.local").name == (
        "dev-201"
    )
    assert (
        resolve_instance(table, env={ENV_INSTANCE: "dev-202"}, hostname="x").name
        == "dev-202"
    )
    with pytest.raises(RoutingTableError, match="matches no instance"):
        resolve_instance(table, env={}, hostname="stickybeatz")
    with pytest.raises(RoutingTableError, match="names no instance"):
        resolve_instance(table, env={ENV_INSTANCE: "dev-9"}, hostname="omnipc2")


# --- AC3 ----------------------------------------------------------------------


def test_consumer_group_per_instance_each_router_subscribes_with_its_group() -> None:
    tables = {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}
    assert consumer_group_for(_router("dev-202", tables)) == "onex-deploy-agent-dev-202"
    assert consumer_group_for(_router("dev-201", tables)) == LEGACY_CONSUMER_GROUP
    assert consumer_group_for(None) == LEGACY_CONSUMER_GROUP


def test_consumer_group_per_instance_reaches_the_kafka_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_kafka(*_args: Any, **kwargs: Any) -> Mock:
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(consumer_mod, "KafkaConsumer", fake_kafka)
    kafka_config = Mock()
    kafka_config.consumer_kwargs.return_value = {}
    store = Mock()
    store.state_dir = tmp_path / "state"
    DeployConsumer(
        kafka_config=kafka_config,
        job_store=store,
        allowed_lanes=frozenset({EnumRuntimeLane.DEV}),
        self_update_hook=lambda rewind: None,
        router=_router("dev-202", {}),
    )
    assert captured["group_id"] == "onex-deploy-agent-dev-202"


def test_consumer_group_per_instance_a_shared_group_refuses() -> None:
    text = TABLE_WITH_OMNIMARKET_ROUTE.replace(
        "consumer_group: onex-deploy-agent-dev-202", "consumer_group: onex-deploy-agent"
    )
    with pytest.raises(RoutingTableError, match="declared by two instances"):
        parse_routing_table(text)


def test_consumer_group_per_instance_the_committed_table_keeps_201s_group() -> None:
    """dev-201 keeps the group the running .201 agent commits under, so its
    offsets carry over on self-update rather than restarting at ``latest``."""
    table = load_routing_table()
    assert table.instances["dev-201"].consumer_group == LEGACY_CONSUMER_GROUP
    assert table.instances["dev-202"].consumer_group == "onex-deploy-agent-dev-202"


# --- the ref reader, on a real clone -------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(repo),
        },
    ).stdout.strip()


def test_git_table_at_ref_reads_the_committed_table_and_its_absence(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "clone"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "README").write_text("x", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "before the table")
    before = _git(repo, "rev-parse", "HEAD")
    table_path = repo / ROUTING_TABLE_RELPATH
    table_path.parent.mkdir(parents=True)
    table_path.write_text(TABLE_WITH_OMNIMARKET_ROUTE, encoding="utf-8")
    _git(repo, "add", ROUTING_TABLE_RELPATH)
    _git(repo, "commit", "-q", "-m", "the table")
    after = _git(repo, "rev-parse", "HEAD")

    reader = GitTableAtRef(str(repo))
    assert reader(before) is None
    table = reader(after)
    assert table is not None
    assert table.route(EnumRuntimeLane.DEV, "gha/omnimarket/pr-1") == "dev-202"
    with pytest.raises(RoutingTableError, match="is not in"):
        reader("c" * 40)
