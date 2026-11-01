# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic names the developer principal plan's rules name (OMN-20821).

The one command topic a principal-specific grant addition may write, the lab
issuer's ``SCOPED_WRITE_TOPICS`` (OPERATOR-CONSENT 2026-10-25T23:01:04Z: h105's
``onex scm pr-create``). The topics are declared as ``scoped_write_topics`` in the
plan node's contract (OMN-21623); the parity test holds this set equal to the
issuer's.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "nodes"
    / "node_developer_principal_plan_compute"
    / "contract.yaml"
)


def _contract_scoped_write_topics(contract_path: Path) -> frozenset[str]:
    """Read ``scoped_write_topics`` from the plan node's contract."""
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    topics = raw.get("scoped_write_topics") if isinstance(raw, dict) else None
    if (
        not isinstance(topics, list)
        or not topics
        or not all(
            isinstance(topic, str) and topic.startswith("onex.cmd.") for topic in topics
        )
    ):
        raise ValueError(
            f"contract {contract_path} must declare scoped_write_topics as a "
            "non-empty list of command topics"
        )
    return frozenset(topics)


SCOPED_WRITE_TOPICS: frozenset[str] = _contract_scoped_write_topics(_CONTRACT_PATH)

__all__: list[str] = ["SCOPED_WRITE_TOPICS"]
