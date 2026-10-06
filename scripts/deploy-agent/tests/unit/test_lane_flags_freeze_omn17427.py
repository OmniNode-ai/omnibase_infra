# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: a frozen lane accepts promotions and suppresses idle converge."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    EnumRejectionReason,
    EnumRuntimeLane,
    ModelRebuildRequested,
    ModelRejectionNotice,
)
from deploy_agent.idle_converge import (
    IDLE_AFTER,
    EnumIdleConvergeVerdict,
    ModelIdleConvergeInputs,
    decide,
)
from deploy_agent.job_state import JobStore
from deploy_agent.routing import (
    ROUTING_TABLE_RELPATH,
    DeployRouter,
    ModelLaneFlags,
    ModelLaneFreeze,
    ModelRoute,
    ModelRoutingTable,
    RoutingTableError,
    lane_flags_for_instance,
    load_routing_table,
    parse_lane_flags,
    parse_routing_table,
    resolve_instance,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SHA_A = "a" * 40
HEAD = "b" * 40
RUNNING = "a" * 40
QUIET = datetime(2026, 9, 25, 4, 53, tzinfo=UTC)
NOW = datetime(2026, 10, 10, tzinfo=UTC)
UNTIL = datetime(2026, 10, 19, 7, tzinfo=UTC)
FREEZE_REASON = "stable demo lane"
FROZEN_DETAIL = "FROZEN until 2026-10-19T07:00Z: stable demo lane"

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


def _flags(until: datetime = UNTIL) -> ModelLaneFlags:
    return ModelLaneFlags(freeze=ModelLaneFreeze(until=until, reason=FREEZE_REASON))


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
    consumer.lane_flags = _flags()
    return consumer


def _process(
    consumer: DeployConsumer, cmd: ModelRebuildRequested
) -> tuple[ModelRebuildRequested | None, str | None]:
    with (
        patch("deploy_agent.consumer.verify_command", return_value=True),
        patch("deploy_agent.consumer.datetime", wraps=datetime) as clock,
    ):
        clock.now.return_value = NOW
        return consumer._process_message(_message(cmd))


def _committed(consumer: DeployConsumer) -> list[int]:
    return [
        meta.offset
        for call in consumer.consumer.commit.call_args_list
        for meta in call.args[0].values()
    ]


def _inputs(**overrides: Any) -> ModelIdleConvergeInputs:
    values: dict[str, Any] = {
        "now": QUIET,
        "omnimarket_routed_elsewhere": True,
        "job_active": False,
        "last_activity": QUIET - IDLE_AFTER,
        "running_ref": RUNNING,
        "head_ref": HEAD,
        "probe_blocker": None,
        "windows_error": None,
        "attempted_heads": frozenset(),
    }
    values.update(overrides)
    return ModelIdleConvergeInputs(**values)


@pytest.mark.parametrize(
    "requested_by",
    [
        pytest.param("gha/omnibase_infra/pr-4570", id="runtime-rebuild-trigger"),
        pytest.param("gha/omninode_infra/push-12", id="onex-api-lab-delivery"),
        pytest.param("operator/deploy-agent-trigger", id="hand-trigger"),
    ],
)
def test_frozen_consumer_refuses_each_recreate_source(requested_by: str) -> None:
    consumer = _consumer(_router("dev-201", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}))
    cmd = _command(requested_by)

    assert _process(consumer, cmd) == (None, EnumRejectionReason.FROZEN.value)
    assert len(consumer.notices) == 1
    notice = consumer.notices[0]
    assert isinstance(notice, ModelRejectionNotice)
    assert notice.reason is EnumRejectionReason.FROZEN
    assert notice.correlation_id == cmd.correlation_id
    assert notice.scope is cmd.scope
    assert _committed(consumer) == [101]
    consumer.job_store.accept.assert_not_called()


def test_frozen_consumer_accepts_a_promotion() -> None:
    consumer = _consumer(_router("dev-201", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}))
    cmd = _command("promotion/operator")

    assert _process(consumer, cmd) == (cmd, None)
    assert consumer.notices == []
    consumer.job_store.accept.assert_called_once()


def test_expired_freeze_accepts_a_gha_command() -> None:
    consumer = _consumer(_router("dev-201", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}))
    consumer.lane_flags = _flags(until=NOW - timedelta(seconds=1))
    cmd = _command("gha/omnibase_infra/pr-4570")

    assert _process(consumer, cmd) == (cmd, None)
    assert consumer.notices == []
    consumer.job_store.accept.assert_called_once()


def test_frozen_consumer_silently_skips_a_command_routed_elsewhere() -> None:
    consumer = _consumer(_router("dev-201", {SHA_A: TABLE_WITH_OMNIMARKET_ROUTE}))

    assert _process(consumer, _command("gha/omnimarket/pr-2851")) == (None, None)
    assert consumer.notices == []
    assert _committed(consumer) == [101]
    consumer.job_store.accept.assert_not_called()


def test_idle_converge_names_frozen_and_preserves_its_detail() -> None:
    decision = decide(_inputs(frozen_reason=FROZEN_DETAIL))

    assert decision.verdict is EnumIdleConvergeVerdict.FROZEN
    assert decision.detail == FROZEN_DETAIL
    assert decide(_inputs()).verdict is EnumIdleConvergeVerdict.CONVERGE


@pytest.mark.parametrize(
    ("flags_block", "error"),
    [
        pytest.param("unknown: true", "Extra inputs", id="unknown-flag"),
        pytest.param(
            'freeze: {until: "2026-10-19T07:00:00", reason: stable demo lane}',
            "timezone",
            id="naive-until",
        ),
    ],
)
def test_parse_lane_flags_refuses_invalid_flags(flags_block: str, error: str) -> None:
    text = f"instances:\n  dev-201:\n    flags:\n      {flags_block}\n"

    with pytest.raises(RoutingTableError, match=error):
        parse_lane_flags(text)


def test_instance_without_flags_has_no_freeze(tmp_path: Path) -> None:
    assert parse_lane_flags(TABLE_WITH_OMNIMARKET_ROUTE) == {}
    path = tmp_path / ROUTING_TABLE_RELPATH
    path.parent.mkdir(parents=True)
    path.write_text(TABLE_WITH_OMNIMARKET_ROUTE, encoding="utf-8")

    flags = lane_flags_for_instance(tmp_path, "dev-201")

    assert flags == ModelLaneFlags()
    assert flags.frozen_reason(NOW) is None


def test_committed_dev_201_freeze_expires_at_the_declared_boundary() -> None:
    flags = lane_flags_for_instance(REPO_ROOT, "dev-201")

    assert flags.freeze is not None
    assert flags.freeze.until == UNTIL
    assert flags.frozen_reason(NOW) == (
        f"FROZEN until 2026-10-19T07:00Z: {flags.freeze.reason}"
    )
    assert flags.frozen_reason(UNTIL) is None
    for instance in ("dev-202", "dev-200"):
        other_flags = lane_flags_for_instance(REPO_ROOT, instance)
        assert other_flags.frozen_reason(NOW) is None
        assert other_flags.frozen_reason(UNTIL) is None


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: object, **kwargs: object) -> Any:
            self.calls.append(name)
            if name == "git_pull":
                return "d" * 40
            if name in ("rebuild_scope", "verify"):
                return []
            return None

        return record


def _agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeployAgent:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    agent = DeployAgent(skip_self_update=True)
    table = agent._router.table if agent._router else load_routing_table()
    routed = table.model_copy(
        update={
            "routes": (
                ModelRoute(
                    runtime_lane=EnumRuntimeLane.DEV,
                    requester_repository="omnimarket",
                    instance="dev-202",
                ),
            )
        }
    )
    agent._router = DeployRouter(routed, resolve_instance(routed), lambda sha: routed)
    agent.job_store = JobStore(tmp_path / "jobs")
    agent.executor = cast("Any", _FakeExecutor())
    agent._idle_converge_started_at = QUIET - timedelta(hours=1)
    agent._idle_converge_now = lambda: QUIET
    agent._idle_read_running_ref = lambda: RUNNING
    agent._idle_read_head_ref = lambda: HEAD
    agent._idle_probe_blocker = lambda now: None
    agent._lane_flags = ModelLaneFlags()
    return agent


def test_frozen_agent_never_starts_an_idle_converge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    agent._lane_flags = _flags()

    with patch.object(
        agent.job_store, "accept", wraps=agent.job_store.accept
    ) as accept:
        agent._maybe_idle_converge()

    accept.assert_not_called()
    assert list((tmp_path / "jobs").glob("*.json")) == []
    assert cast("Any", agent.executor).calls == []
    assert agent._idle_converge_last_verdict is EnumIdleConvergeVerdict.FROZEN


def test_freeze_substitute_instance_is_declared_data_omn20006() -> None:
    """OMN-20006: the freeze names the instance that proves in its place."""
    from deploy_agent.routing import parse_lane_flags

    text = (
        "instances:\n"
        "  dev-201:\n"
        "    flags:\n"
        "      freeze:\n"
        "        until: 2026-10-19T07:00:00Z\n"
        "        reason: demo\n"
        "        substitute_instance: dev-202\n"
    )
    freeze = parse_lane_flags(text)["dev-201"].freeze
    assert freeze is not None
    assert freeze.substitute_instance == "dev-202"
    bare = text.replace("        substitute_instance: dev-202\n", "")
    bare_freeze = parse_lane_flags(bare)["dev-201"].freeze
    assert bare_freeze is not None
    assert bare_freeze.substitute_instance == ""
