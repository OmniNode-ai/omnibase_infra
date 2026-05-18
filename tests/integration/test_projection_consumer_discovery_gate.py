# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: projection consumers must be discoverable when producer package is inactive.

OMN-11185: node_projection_delegation subscribes to onex.evt.omniclaude.* topics but
does not publish to that domain. ONEX_ACTIVE_RUNTIME_PACKAGES must not block it.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths

pytestmark = [
    pytest.mark.integration,
]


def _write_projection_contract(
    directory: Path, name: str, subscribe_topics: list[str]
) -> Path:
    """Write a projection-consumer contract.yaml and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "contract.yaml"
    subs_lines = "".join(f"    - {t!r}\n" for t in subscribe_topics)
    path.write_text(
        f'name: "{name}"\n'
        'node_type: "REDUCER_GENERIC"\n'
        "contract_version:\n"
        "  major: 1\n"
        "  minor: 0\n"
        "  patch: 0\n"
        'node_version: "1.0.0"\n'
        'description: "Projection consumer - read-only subscriber"\n'
        "event_bus:\n"
        "  subscribe_topics:\n"
        f"{subs_lines}"
        "  publish_topics: []\n"
        '  consumer_purpose: "projection"\n'
    )
    return path


class TestProjectionConsumerDiscoveryGate:
    """OMN-11185: verify projection consumers load regardless of ONEX_ACTIVE_RUNTIME_PACKAGES."""

    @pytest.mark.integration
    def test_projection_consumer_subscribing_to_omniclaude_loads_when_omniclaude_inactive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """node_projection_delegation (subscribes to omniclaude topics) must be discoverable
        even when ONEX_ACTIVE_RUNTIME_PACKAGES does not include omniclaude."""
        monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", "omnibase_infra,omnimarket")

        path = _write_projection_contract(
            tmp_path / "node_projection_delegation",
            name="node_projection_delegation",
            subscribe_topics=[
                "onex.evt.omniclaude.task-delegated.v1",
                "onex.evt.omniclaude.delegation-completed.v1",
            ],
        )

        manifest = discover_contracts_from_paths([path])

        assert manifest.total_errors == 0, f"Unexpected errors: {manifest.errors}"
        assert manifest.total_discovered == 1, (
            "node_projection_delegation was filtered by ONEX_ACTIVE_RUNTIME_PACKAGES "
            "even though it only subscribes to omniclaude topics (does not publish). "
            "OMN-11185 regression."
        )
        assert manifest.contracts[0].name == "node_projection_delegation"

    @pytest.mark.integration
    def test_producer_into_inactive_domain_still_filtered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A contract that PUBLISHES into an inactive package domain must still be filtered.

        This verifies the gate still works for its original purpose after the OMN-11185 fix.
        """
        monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", "omnibase_infra,omnimarket")

        node_dir = tmp_path / "node_omniclaude_publisher"
        node_dir.mkdir()
        path = node_dir / "contract.yaml"
        path.write_text(
            dedent("""\
                name: "node_omniclaude_publisher"
                node_type: "EFFECT_GENERIC"
                contract_version:
                  major: 1
                  minor: 0
                  patch: 0
                node_version: "1.0.0"
                description: "Publishes into omniclaude domain"
                event_bus:
                  subscribe_topics:
                    - "onex.cmd.platform.trigger.v1"
                  publish_topics:
                    - "onex.evt.omniclaude.agent-invoked.v1"
                  consumer_purpose: "effects"
            """)
        )

        manifest = discover_contracts_from_paths([path])

        assert manifest.total_discovered == 0, (
            "Contract publishing into inactive omniclaude domain should be filtered "
            "by ONEX_ACTIVE_RUNTIME_PACKAGES gate."
        )
        assert manifest.total_errors == 0

    @pytest.mark.integration
    def test_no_env_filter_discovers_all_contracts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ONEX_ACTIVE_RUNTIME_PACKAGES is unset, all contracts are discovered."""
        monkeypatch.delenv("ONEX_ACTIVE_RUNTIME_PACKAGES", raising=False)

        path1 = _write_projection_contract(
            tmp_path / "proj",
            name="node_proj",
            subscribe_topics=["onex.evt.omniclaude.task-delegated.v1"],
        )
        node_dir = tmp_path / "publisher"
        node_dir.mkdir()
        path2 = node_dir / "contract.yaml"
        path2.write_text(
            dedent("""\
                name: "node_publisher"
                node_type: "EFFECT_GENERIC"
                contract_version:
                  major: 1
                  minor: 0
                  patch: 0
                node_version: "1.0.0"
                description: "Publisher"
                event_bus:
                  subscribe_topics: []
                  publish_topics:
                    - "onex.evt.omniclaude.agent-invoked.v1"
            """)
        )

        manifest = discover_contracts_from_paths([path1, path2])

        assert manifest.total_errors == 0
        assert manifest.total_discovered == 2
