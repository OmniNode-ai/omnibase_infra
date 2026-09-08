# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-bus services for node_artifact_change_detector_effect.

OMN-18013: ``ContractFileWatcher`` is a watchdog-driven filesystem
watcher, not a bus consumer -- it has no dispatch entrypoint and no
``handler_routing`` entry. It lived under ``handlers/`` while carrying an
``operation_match`` entry with no ``topic:``, which made
``_topics_for_handler_entry`` assign it EVERY subscribe topic of the node, so
one entry spanned both a ``.cmd.`` and an ``.evt.`` topic. Its home is here,
beside the other node-local services, where nothing claims the bus dispatches
to it.
"""

from omnibase_infra.nodes.node_artifact_change_detector_effect.services.contract_file_watcher import (
    ContractFileWatcher,
)

__all__ = ["ContractFileWatcher"]
