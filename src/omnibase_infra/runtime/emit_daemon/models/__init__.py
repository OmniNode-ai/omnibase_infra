# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Emit Daemon Runtime Models.

Topic-marker input/output models for the node_emit_daemon_runtime contract.
"""

from omnibase_infra.runtime.emit_daemon.models.model_emit_daemon_runtime_input import (
    ModelEmitDaemonRuntimeInput,
)
from omnibase_infra.runtime.emit_daemon.models.model_emit_daemon_runtime_output import (
    ModelEmitDaemonRuntimeOutput,
)

__all__ = [
    "ModelEmitDaemonRuntimeInput",
    "ModelEmitDaemonRuntimeOutput",
]
