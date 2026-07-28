# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for the managed-staging (mstg1) topic/group broker checker (OMN-15283).

Covers:
    * catalog parity -- the checker's topic set is exactly
      ``build_canary_catalog_from_candidate``'s output, never re-derived;
    * check-only mode makes zero admin-client mutation calls even when
      catalog topics are missing;
    * MSK IAM auth kwargs are built identically to ``TopicProvisioner``'s;
    * negative control -- an out-of-catalog ``onex.mstg1.*`` name is reported
      but never created/deleted.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import omnibase_infra.topics.managed_staging_topic_checker as checker_module
from omnibase_infra.topics.managed_staging_canary_catalog import (
    ModelCanaryNamespace,
    build_canary_catalog_from_candidate,
)
from omnibase_infra.topics.managed_staging_topic_checker import (
    _build_arg_parser,
    _run,
    build_topic_diff,
    create_missing_catalog_topics,
    open_admin_client,
)

pytestmark = [pytest.mark.unit]

_IAM_PATTERNS = ("onex.*", "omninode.*")


@pytest.fixture
def contract_root(tmp_path: Path) -> Path:
    """Minimal contract root: one publishing node, one subscribing node."""
    pub_dir = tmp_path / "node_publisher"
    pub_dir.mkdir()
    (pub_dir / "contract.yaml").write_text(
        "name: node_publisher\n"
        "version: 1.0.0\n"
        "namespace: onex.stamped\n"
        "event_bus:\n"
        "  publish_topics:\n"
        "    - onex.evt.test-producer.example-event.v1\n"
    )
    sub_dir = tmp_path / "node_subscriber"
    sub_dir.mkdir()
    (sub_dir / "contract.yaml").write_text(
        "name: node_subscriber\n"
        "version: 1.0.0\n"
        "namespace: onex.stamped\n"
        "event_bus:\n"
        "  subscribe_topics:\n"
        "    - onex.evt.test-producer.example-event.v1\n"
    )
    return tmp_path


def _namespace(**overrides: object) -> ModelCanaryNamespace:
    base: dict[str, object] = {
        "ticket": "OMN-15283",
        "epoch": "mstg1",
        "topic_prefix": "onex.mstg1.",
        "group_prefix": "onex.mstg1.",
        "iam_topic_patterns": _IAM_PATTERNS,
        "iam_group_patterns": _IAM_PATTERNS,
        "default_partitions": 1,
        "default_replication_factor": 2,
        "candidate_contract_roots": ("nodes",),
    }
    base.update(overrides)
    return ModelCanaryNamespace.model_validate(base)


# --------------------------------------------------------------------------- #
# Catalog parity
# --------------------------------------------------------------------------- #


def test_diff_topic_set_matches_catalog_topic_names(contract_root: Path) -> None:
    """The diff never re-derives the topic set -- it is exactly the catalog's."""
    namespace = _namespace(candidate_contract_roots=("",))
    catalog = build_canary_catalog_from_candidate(namespace, base_dir=contract_root)

    assert catalog.topic_names  # sanity: the fixture produced at least one topic

    diff = build_topic_diff(catalog, existing_topics=[], existing_groups=[])

    assert set(diff.missing_topics) == set(catalog.topic_names)
    assert diff.missing_topics == tuple(sorted(catalog.topic_names))


def test_diff_present_topics_when_all_exist(contract_root: Path) -> None:
    namespace = _namespace(candidate_contract_roots=("",))
    catalog = build_canary_catalog_from_candidate(namespace, base_dir=contract_root)

    diff = build_topic_diff(
        catalog,
        existing_topics=catalog.topic_names,
        existing_groups=catalog.groups,
    )

    assert diff.missing_topics == ()
    assert diff.missing_groups == ()
    assert set(diff.present_topics) == set(catalog.topic_names)
    assert diff.is_fully_present is True


# --------------------------------------------------------------------------- #
# Check-only zero-mutation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_check_only_run_makes_zero_admin_mutation_calls(
    contract_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end ``_run`` in check-only mode never calls create/delete on the admin client."""
    namespace = _namespace(candidate_contract_roots=("",))
    catalog = build_canary_catalog_from_candidate(namespace, base_dir=contract_root)

    fake_admin = MagicMock()
    fake_admin.close = AsyncMock()
    fake_admin.create_topics = AsyncMock(
        side_effect=AssertionError(
            "create_topics must not be called in check-only mode"
        )
    )
    fake_admin.delete_topics = AsyncMock(
        side_effect=AssertionError("delete_topics must never be called")
    )

    monkeypatch.setattr(
        checker_module,
        "load_canary_namespace",
        lambda path: namespace,
    )
    monkeypatch.setattr(
        checker_module,
        "build_canary_catalog_from_candidate",
        lambda ns, base_dir=None: catalog,
    )
    monkeypatch.setattr(
        checker_module,
        "open_admin_client",
        AsyncMock(return_value=fake_admin),
    )
    monkeypatch.setattr(
        checker_module,
        "fetch_live_topics_and_groups",
        AsyncMock(return_value=((), ())),
    )
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "b-1.example:9098")

    parser = _build_arg_parser()
    args = parser.parse_args([])  # --create-missing absent (default False)

    exit_code = await _run(args)

    fake_admin.create_topics.assert_not_called()
    fake_admin.delete_topics.assert_not_called()
    fake_admin.close.assert_awaited_once()
    assert exit_code == 1  # fail-closed: catalog topics are missing


# --------------------------------------------------------------------------- #
# Auth kwargs parity with TopicProvisioner
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_open_admin_client_passes_msk_iam_auth_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "b-1.example:9098")
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "AWS_MSK_IAM")
    monkeypatch.setenv("KAFKA_MSK_REGION", "us-east-1")

    mock_admin_cls = MagicMock()
    mock_admin_instance = AsyncMock()
    mock_admin_instance.start = AsyncMock()
    mock_admin_cls.return_value = mock_admin_instance

    with patch.dict(
        "sys.modules",
        {
            "aiokafka": MagicMock(),
            "aiokafka.admin": MagicMock(AIOKafkaAdminClient=mock_admin_cls),
        },
    ):
        result = await open_admin_client(bootstrap_servers="b-1.example:9098")

    from omnibase_infra.event_bus.kafka_auth import MSKTokenProvider

    assert result is mock_admin_instance
    admin_kwargs = mock_admin_cls.call_args.kwargs
    assert admin_kwargs["security_protocol"] == "SASL_SSL"
    assert admin_kwargs["sasl_mechanism"] == "OAUTHBEARER"
    assert isinstance(admin_kwargs["sasl_oauth_token_provider"], MSKTokenProvider)
    mock_admin_instance.start.assert_awaited_once()


# --------------------------------------------------------------------------- #
# Negative control: out-of-catalog name reported, never mutated
# --------------------------------------------------------------------------- #


def test_out_of_catalog_name_is_reported_not_touched(contract_root: Path) -> None:
    namespace = _namespace(candidate_contract_roots=("",))
    catalog = build_canary_catalog_from_candidate(namespace, base_dir=contract_root)

    stray_topic = "onex.mstg1.onex.evt.some-stray.unlisted-event.v1"
    assert stray_topic not in catalog.topic_names

    diff = build_topic_diff(
        catalog,
        existing_topics=(*catalog.topic_names, stray_topic),
        existing_groups=catalog.groups,
    )

    assert stray_topic in diff.out_of_catalog_topics
    assert stray_topic not in diff.missing_topics
    assert stray_topic not in diff.present_topics
    assert diff.has_out_of_catalog is True


@pytest.mark.asyncio
async def test_create_missing_catalog_topics_never_touches_out_of_catalog_name(
    contract_root: Path,
) -> None:
    namespace = _namespace(candidate_contract_roots=("",))
    catalog = build_canary_catalog_from_candidate(namespace, base_dir=contract_root)

    stray_topic = "onex.mstg1.onex.evt.some-stray.unlisted-event.v1"
    diff = build_topic_diff(
        catalog,
        existing_topics=(stray_topic,),
        existing_groups=(),
    )
    assert stray_topic in diff.out_of_catalog_topics
    assert set(diff.missing_topics) == set(catalog.topic_names)

    fake_admin = MagicMock()
    created_names: list[str] = []

    async def _fake_create_topics(new_topics: list[object]) -> None:
        for nt in new_topics:
            created_names.append(nt.name)  # type: ignore[attr-defined]

    fake_admin.create_topics = AsyncMock(side_effect=_fake_create_topics)
    fake_admin.delete_topics = AsyncMock(
        side_effect=AssertionError("delete_topics must never be called")
    )

    with patch.dict(
        "sys.modules",
        {
            "aiokafka": MagicMock(),
            "aiokafka.admin": MagicMock(NewTopic=_FakeNewTopic),
            "aiokafka.errors": MagicMock(
                TopicAlreadyExistsError=type(
                    "TopicAlreadyExistsError", (Exception,), {}
                )
            ),
        },
    ):
        created, failed = await create_missing_catalog_topics(fake_admin, catalog, diff)

    assert stray_topic not in created_names
    assert set(created_names) == set(catalog.topic_names)
    assert set(created) == set(catalog.topic_names)
    assert failed == ()
    fake_admin.delete_topics.assert_not_called()


class _FakeNewTopic:
    """Minimal stand-in for ``aiokafka.admin.NewTopic``."""

    def __init__(
        self,
        *,
        name: str,
        num_partitions: int,
        replication_factor: int,
        topic_configs: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor
        self.topic_configs = topic_configs or {}
