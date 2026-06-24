# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for subscription verification probe [OMN-7040]."""

from __future__ import annotations

import json

import pytest

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
)
from omnibase_infra.verification.probes.probe_subscription import (
    _discover_split_group_topics,
    _rpk_fallback,
    _topic_from_split_group_name,
    check_subscriptions,
)

CONTRACT_NAME = "node_registration_orchestrator"

SAMPLE_TOPICS = (
    "onex.evt.platform.node-introspection.v1",
    "onex.evt.platform.registry-request-introspection.v1",
    "onex.intent.platform.runtime-tick.v1",
)


def _make_contract(
    subscribe_topics: tuple[str, ...] = SAMPLE_TOPICS,
) -> ModelParsedContractForVerification:
    return ModelParsedContractForVerification(
        name=CONTRACT_NAME,
        node_type="ORCHESTRATOR_GENERIC",
        subscribe_topics=subscribe_topics,
    )


def _make_admin_fn(
    subscribed: set[str] | None = None,
    error: Exception | None = None,
) -> object:
    """Create a mock kafka_admin_fn.

    Args:
        subscribed: Set of topics to return. If None, returns all SAMPLE_TOPICS.
        error: If set, the function raises this exception.
    """
    if subscribed is None:
        subscribed = set(SAMPLE_TOPICS)

    def admin_fn(group_id: str) -> set[str]:
        if error is not None:
            raise error
        return subscribed

    return admin_fn


@pytest.mark.unit
class TestCheckSubscriptionsPass:
    """Happy-path subscription checks."""

    def test_all_topics_subscribed(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(contract, kafka_admin_fn=_make_admin_fn())
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.PASS for r in results)
        assert all(r.check_type == EnumContractCheckType.SUBSCRIPTION for r in results)

    def test_no_topics_declared(self) -> None:
        contract = _make_contract(subscribe_topics=())
        results = check_subscriptions(contract, kafka_admin_fn=_make_admin_fn())
        assert len(results) == 1
        assert results[0].verdict == EnumValidationVerdict.PASS
        assert "No subscribe_topics" in results[0].evidence

    def test_contract_name_preserved(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(contract, kafka_admin_fn=_make_admin_fn())
        assert all(r.contract_name == CONTRACT_NAME for r in results)


@pytest.mark.unit
class TestCheckSubscriptionsFail:
    """Failure-path subscription checks."""

    def test_grounded_consumer_group_empty_is_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.models import ModelNodeIdentity
        from omnibase_infra.verification.probes import probe_subscription

        monkeypatch.setattr(
            probe_subscription,
            "_discover_identity_via_rpk",
            lambda contract_name: ModelNodeIdentity(
                env="dev",
                service="omnibase_infra",
                node_name="registration-orchestrator",
                version="1.0.0",
            ),
        )

        contract = _make_contract()
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=set())
        )
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.FAIL for r in results)
        assert all("grounding=DISCOVERED" in r.evidence for r in results)

    def test_consumer_group_empty(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=set())
        )
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.QUARANTINE for r in results)
        assert all("no subscribed topics" in r.evidence.lower() for r in results)
        assert all("grounding=fabricated" in r.evidence.lower() for r in results)

    def test_partial_subscriptions(self) -> None:
        subscribed = {SAMPLE_TOPICS[0]}
        contract = _make_contract()
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=subscribed)
        )
        pass_count = sum(1 for r in results if r.verdict == EnumValidationVerdict.PASS)
        quarantine_count = sum(
            1 for r in results if r.verdict == EnumValidationVerdict.QUARANTINE
        )
        assert pass_count == 1
        assert quarantine_count == 2

    def test_missing_topic_evidence(self) -> None:
        subscribed = {SAMPLE_TOPICS[0]}
        contract = _make_contract()
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=subscribed)
        )
        degraded_results = [
            r for r in results if r.verdict == EnumValidationVerdict.QUARANTINE
        ]
        for r in degraded_results:
            assert "NOT subscribed" in r.evidence
            assert "grounding=FABRICATED" in r.evidence


@pytest.mark.unit
class TestCheckSubscriptionsQuarantine:
    """Degraded-state checks: admin API unavailable -> QUARANTINE."""

    def test_admin_api_unavailable(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(
            contract,
            kafka_admin_fn=_make_admin_fn(error=ConnectionError("Kafka unavailable")),
        )
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.QUARANTINE for r in results)

    def test_quarantine_evidence_includes_error(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(
            contract,
            kafka_admin_fn=_make_admin_fn(error=ConnectionError("Connection refused")),
        )
        assert all("Connection refused" in r.evidence for r in results)

    def test_quarantine_severity_is_required(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(
            contract,
            kafka_admin_fn=_make_admin_fn(error=RuntimeError("timeout")),
        )
        assert all(r.severity == EnumCheckSeverity.REQUIRED for r in results)

    def test_fabricated_grounding_marks_result_non_authoritative(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=set())
        )
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.QUARANTINE for r in results)
        assert all("grounding=FABRICATED" in r.evidence for r in results)
        assert all("quarantined" in r.message.lower() for r in results)


@pytest.mark.unit
class TestCheckSubscriptionsOnePerTopic:
    """One result per subscribe_topic."""

    def test_result_count_matches_topic_count(self) -> None:
        topics = (
            "topic.a",
            "topic.b",
            "topic.c",
            "topic.d",
        )
        contract = _make_contract(subscribe_topics=topics)
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=set(topics))
        )
        assert len(results) == 4

    def test_single_topic(self) -> None:
        contract = _make_contract(subscribe_topics=("only.topic",))
        results = check_subscriptions(
            contract,
            kafka_admin_fn=_make_admin_fn(subscribed={"only.topic"}),
        )
        assert len(results) == 1
        assert results[0].verdict == EnumValidationVerdict.PASS


@pytest.mark.unit
class TestRpkFallback:
    """rpk fallback should honor topic-scoped consumer groups."""

    def test_topic_scoped_groups_are_aggregated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base_group = "local.runtime_config.registration-orchestrator.consume.1.0.0"

        class CompletedProcessStub:
            def __init__(
                self, stdout: str, returncode: int = 0, stderr: str = ""
            ) -> None:
                self.stdout = stdout
                self.returncode = returncode
                self.stderr = stderr

        def fake_run(
            args: list[str],
            capture_output: bool,
            text: bool,
            timeout: int,
            check: bool,
            env: dict | None = None,
        ) -> CompletedProcessStub:
            assert capture_output is True
            assert text is True
            assert check is False
            if args == ["rpk", "group", "describe", base_group, "--format", "json"]:
                return CompletedProcessStub(stdout=json.dumps({"members": []}))
            if args == ["rpk", "group", "list", "--format", "json"]:
                return CompletedProcessStub(
                    stdout=json.dumps(
                        [
                            {
                                "name": f"{base_group}.__t.onex.evt.platform.node-heartbeat.v1"
                            },
                            {
                                "name": f"{base_group}.__t.onex.intent.platform.runtime-tick.v1"
                            },
                        ]
                    )
                )
            if (
                len(args) == 6
                and args[:3] == ["rpk", "group", "describe"]
                and args[-2:] == ["--format", "json"]
            ):
                scoped_group = args[3]
                topic = scoped_group.split(".__t.", 1)[1]
                return CompletedProcessStub(
                    stdout=json.dumps(
                        {
                            "members": [
                                {"assignments": [{"topic": topic}]},
                            ]
                        }
                    )
                )
            raise AssertionError(f"Unexpected subprocess invocation: {args}")

        monkeypatch.setattr(
            "omnibase_infra.verification.probes.probe_subscription.subprocess.run",
            fake_run,
        )

        assert _rpk_fallback(base_group) == {
            "onex.evt.platform.node-heartbeat.v1",
            "onex.intent.platform.runtime-tick.v1",
        }


# Live per-topic split-group names as emitted by the runtime (OMN-13554). The
# base group carries the .__i.{instance} infix and each topic gets its own
# .__t.{topic} split-group. These mirror the dev-lane broker output verbatim.
_LIVE_BASE = (
    "local.omnibase_infra.node_registration_orchestrator.consume.1.1.1.__i.runtime-main"
)
_LIVE_TOPICS = (
    "onex.cmd.platform.node-registration-acked.v1",
    "onex.cmd.platform.request-introspection.v1",
    "onex.cmd.platform.topic-catalog-query.v1",
    "onex.evt.platform.node-heartbeat.v1",
    "onex.evt.platform.node-introspection.v1",
    "onex.evt.platform.registry-request-introspection.v1",
    "onex.intent.platform.runtime-tick.v1",
)
_LIVE_SPLIT_GROUPS = tuple(f"{_LIVE_BASE}.__t.{t}" for t in _LIVE_TOPICS)

# The legacy monolithic group the stale derivation computes -- no longer exists
# on the broker. This is the name that produced the recurring false FAIL.
_LEGACY_DERIVED_GROUP = "local.runtime_config.registration-orchestrator.consume.1.0.0"


@pytest.mark.unit
class TestTopicFromSplitGroupName:
    """Parse the topic straight out of a per-topic split-group name."""

    def test_parses_topic_after_last_marker(self) -> None:
        group = _LIVE_SPLIT_GROUPS[0]
        assert (
            _topic_from_split_group_name(group)
            == "onex.cmd.platform.node-registration-acked.v1"
        )

    def test_uses_last_marker_when_topic_contains_no_marker(self) -> None:
        # The .__i. instance infix precedes the single .__t. topic marker; the
        # parser must split on the LAST .__t. so the topic is returned intact.
        for group, topic in zip(_LIVE_SPLIT_GROUPS, _LIVE_TOPICS, strict=True):
            assert _topic_from_split_group_name(group) == topic

    def test_non_split_group_returns_none(self) -> None:
        assert _topic_from_split_group_name(_LEGACY_DERIVED_GROUP) is None

    def test_empty_topic_suffix_returns_none(self) -> None:
        assert _topic_from_split_group_name(f"{_LIVE_BASE}.__t.") is None


@pytest.mark.unit
class TestDiscoverSplitGroupTopics:
    """Discover live split-groups by contract name (the OMN-13554 fix)."""

    def _patch_group_list(
        self, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]
    ) -> None:
        monkeypatch.setattr(
            "omnibase_infra.verification.probes.probe_subscription._rpk_group_list",
            lambda: [{"name": n} for n in names],
        )

    def test_discovers_all_live_split_group_topics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_group_list(monkeypatch, _LIVE_SPLIT_GROUPS)
        assert _discover_split_group_topics("node_registration_orchestrator") == set(
            _LIVE_TOPICS
        )

    def test_ignores_unrelated_node_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = "local.omnibase_infra.node_other.consume.1.0.0.__t.onex.evt.foo.v1"
        self._patch_group_list(monkeypatch, (*_LIVE_SPLIT_GROUPS, other))
        topics = _discover_split_group_topics("node_registration_orchestrator")
        assert topics == set(_LIVE_TOPICS)
        assert "onex.evt.foo.v1" not in topics

    def test_no_match_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_group_list(monkeypatch, (_LEGACY_DERIVED_GROUP,))
        assert _discover_split_group_topics("node_registration_orchestrator") == set()


@pytest.mark.unit
class TestRpkFallbackDiscoversSplitGroupsByContract:
    """End-to-end: the derived base group is stale, discovery saves the probe.

    This is the exact OMN-13554 scenario: the stale monolithic group derivation
    yields a base group that does not exist on the broker, while the live
    runtime subscribes via per-topic split-groups under a different base name.
    The fallback must discover those by contract name and return the real topics
    rather than emitting a false 7/7-missing FAIL.
    """

    def test_stale_base_group_falls_through_to_contract_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class CompletedProcessStub:
            def __init__(
                self, stdout: str, returncode: int = 0, stderr: str = ""
            ) -> None:
                self.stdout = stdout
                self.returncode = returncode
                self.stderr = stderr

        def fake_run(
            args: list[str],
            capture_output: bool,
            text: bool,
            timeout: int,
            check: bool,
            env: dict | None = None,
        ) -> CompletedProcessStub:
            # describe of the stale base group: exists but has no members.
            if args[:3] == ["rpk", "group", "describe"]:
                return CompletedProcessStub(stdout=json.dumps({"members": []}))
            # list returns only the LIVE split-groups (the stale base is absent,
            # so its .__t. anchored search finds nothing).
            if args == ["rpk", "group", "list", "--format", "json"]:
                return CompletedProcessStub(
                    stdout=json.dumps([{"name": n} for n in _LIVE_SPLIT_GROUPS])
                )
            raise AssertionError(f"Unexpected subprocess invocation: {args}")

        monkeypatch.setattr(
            "omnibase_infra.verification.probes.probe_subscription.subprocess.run",
            fake_run,
        )

        assert _rpk_fallback(
            _LEGACY_DERIVED_GROUP,
            contract_name="node_registration_orchestrator",
        ) == set(_LIVE_TOPICS)

    def test_check_subscriptions_passes_against_live_split_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the stale EXACT identity that the runtime_config.yaml derivation
        # produces, then prove the contract-name discovery still resolves PASS.
        from omnibase_infra.models import ModelNodeIdentity

        monkeypatch.setattr(
            "omnibase_infra.verification.probes.probe_subscription._rpk_group_list",
            lambda: [{"name": n} for n in _LIVE_SPLIT_GROUPS],
        )

        class CompletedProcessStub:
            def __init__(self, stdout: str, returncode: int = 0) -> None:
                self.stdout = stdout
                self.returncode = returncode
                self.stderr = ""

        def fake_run(args, **_kwargs):  # type: ignore[no-untyped-def]
            # describe of the stale base group -> no members; list path goes
            # through the patched _rpk_group_list above.
            return CompletedProcessStub(stdout=json.dumps({"members": []}))

        monkeypatch.setattr(
            "omnibase_infra.verification.probes.probe_subscription.subprocess.run",
            fake_run,
        )

        contract = _make_contract(subscribe_topics=_LIVE_TOPICS)
        results = check_subscriptions(
            contract,
            identity=ModelNodeIdentity(
                env="local",
                service="runtime_config",
                node_name="registration-orchestrator",
                version="1.0.0",
            ),
        )
        assert len(results) == len(_LIVE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.PASS for r in results)
