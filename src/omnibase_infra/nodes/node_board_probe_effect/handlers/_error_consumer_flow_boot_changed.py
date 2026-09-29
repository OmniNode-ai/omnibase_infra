# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A sampling window crossed lane boots."""

from omnibase_infra.nodes.node_board_probe_effect.handlers._error_consumer_flow_input import (
    ConsumerFlowInputError,
)


class ConsumerFlowBootChangedError(ConsumerFlowInputError):
    """The lane boot changed; retry the entire window once."""
