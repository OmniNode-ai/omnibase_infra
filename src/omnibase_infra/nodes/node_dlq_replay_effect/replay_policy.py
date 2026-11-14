# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the replay policy from the DLQ replay node's own contract (OMN-24177).

The contract is the source of truth for which topics are never replayed and how
many replays one (topic, listener group) pair gets per window, the same way it
is for the subscribe topics. Fails CLOSED: an unreadable contract or a missing
or invalid ``replay_policy`` block raises, never a hard-coded fallback, because
a silent fallback would replay a broadcast topic in the one situation the
policy exists to prevent.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files

import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_replay_policy import (
    ModelDlqReplayPolicy,
)


def parse_replay_policy(contract_text: str, source: str) -> ModelDlqReplayPolicy:
    """The typed ``replay_policy`` block of ``contract_text``; raises when absent or invalid."""
    try:
        block = yaml.safe_load(contract_text)["replay_policy"]
        return ModelDlqReplayPolicy.model_validate(block)
    except (KeyError, TypeError, ValueError, yaml.YAMLError, ValidationError) as exc:
        raise ProtocolConfigurationError(
            "DLQ replay requires a valid replay_policy block in the node "
            f"contract at {source}: broadcast_topics (a list, possibly empty) "
            "and target_replay_bound (max_replays > 0, window_seconds > 0)"
        ) from exc


@cache
def load_replay_policy() -> ModelDlqReplayPolicy:
    """The policy of the installed contract, read once."""
    package = __package__ or ""
    try:
        text = files(package).joinpath("contract.yaml").read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolConfigurationError(
            f"DLQ replay cannot read the node contract of package {package}"
        ) from exc
    return parse_replay_policy(text, package)


__all__ = ["load_replay_policy", "parse_replay_policy"]
