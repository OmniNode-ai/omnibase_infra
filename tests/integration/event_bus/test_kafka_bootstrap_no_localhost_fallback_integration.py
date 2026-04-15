# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OMN-8783: hard-fail on missing KAFKA_BOOTSTRAP_SERVERS.

Exercises the full TopicProvisioner init path through the real service stack —
no mocks on the env-var resolution. The provisioner must raise KeyError (not
fall back to localhost) when KAFKA_BOOTSTRAP_SERVERS is absent.

Related:
    - OMN-8783: Kafka bootstrap overlay split — remove localhost:19092 fallback
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

pytestmark = [pytest.mark.integration]


@pytest.fixture
def contracts_root(tmp_path: Path) -> Path:
    """Minimal contracts directory with a valid contract.yaml."""
    node_dir = tmp_path / "node_example"
    node_dir.mkdir()
    (node_dir / "contract.yaml").write_text(
        "name: node_example\n"
        "version: 1.0.0\n"
        "namespace: onex.stamped\n"
        "event_bus:\n"
        "  publish_topics:\n"
        "    - onex.evt.test-producer.example-event.v1\n"
    )
    return tmp_path


class TestKafkaBootstrapNoLocalhostFallback:
    """Verify the OMN-8783 hard-fail invariant through the full service init path."""

    def test_raises_key_error_when_env_absent(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TopicProvisioner raises KeyError when KAFKA_BOOTSTRAP_SERVERS is unset.

        This is the core OMN-8783 invariant: no silent fallback to localhost.
        """
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        with pytest.raises(KeyError, match="KAFKA_BOOTSTRAP_SERVERS"):
            TopicProvisioner(contracts_root=contracts_root)

    def test_no_localhost_default_used(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Constructor must not embed 'localhost' when env var is absent."""
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        with pytest.raises(KeyError, match="KAFKA_BOOTSTRAP_SERVERS"):
            TopicProvisioner(contracts_root=contracts_root)

    def test_explicit_bootstrap_servers_accepted(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Constructor accepts an explicit bootstrap_servers arg without env var."""
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        provisioner = TopicProvisioner(
            bootstrap_servers="redpanda:9092",
            contracts_root=contracts_root,
        )
        assert provisioner._bootstrap_servers == "redpanda:9092"

    def test_env_var_overlay_respected(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Constructor reads KAFKA_BOOTSTRAP_SERVERS from env when not passed explicitly."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker.example.com:9092")

        provisioner = TopicProvisioner(contracts_root=contracts_root)
        assert provisioner._bootstrap_servers == "broker.example.com:9092"
        assert "localhost" not in provisioner._bootstrap_servers

    def test_env_var_wins_over_no_arg(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Env var overlay is the only source when bootstrap_servers arg is None."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka1:9092,kafka2:9092")

        provisioner = TopicProvisioner(contracts_root=contracts_root)
        assert provisioner._bootstrap_servers == "kafka1:9092,kafka2:9092"

    def test_missing_contracts_root_raises_file_not_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FileNotFoundError raised for nonexistent contracts_root (service-stack check)."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
        bad_path = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError, match="contracts_root"):
            TopicProvisioner(
                bootstrap_servers="redpanda:9092",
                contracts_root=bad_path,
            )

    def test_bootstrap_servers_env_var_constant(self) -> None:
        """ENV_BOOTSTRAP_SERVERS constant must be KAFKA_BOOTSTRAP_SERVERS (not hardcoded)."""
        from omnibase_infra.event_bus.service_topic_manager import ENV_BOOTSTRAP_SERVERS

        assert ENV_BOOTSTRAP_SERVERS == "KAFKA_BOOTSTRAP_SERVERS"
