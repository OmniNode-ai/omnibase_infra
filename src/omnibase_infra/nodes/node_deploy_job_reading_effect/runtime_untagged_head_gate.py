# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's idle builds wait for the dev head's dev-merge tag (OMN-25942).

The deploy agent loads this by file path (``load_dev_boot_guard``), like the node's other
host composition roots, and imports nothing from ``deploy_agent``: the agent hands in
itself and the ledger writer, and every read goes through its own idle seams.

WHY THIS EXISTS
---------------
At 2026-11-24T02:46:16Z the dev agent's idle restore (job 7dfb1bc9) fetched the omnibase_infra
dev head 5cafc841f7de, 44 seconds after it was pushed. The Dev Merge Tag workflow for that push
(``.github/workflows/dev-merge-tag.yml``, run 45314704581) ran from 02:45:32Z to 02:48:12Z (the
median of the last 100 runs is 15.5 s), so the tag did not exist when the build staged the tree.
``stage_workspace.sh`` records the tags that point at the staged HEAD, the record was empty, the
dev-lane first-party audit printed ``FAIL omnibase-infra: branch (5cafc841f7de) staged from
branch dev`` and the build ended ``runtime_image_build_errored`` ("a BROKEN BUILD ... retrying it
unchanged buys nothing"). The lane stayed down: dev had been down since the rolls of the night.

Nothing in the agent's idle path read the tag: idle restore and idle converge build the tracking
ref the moment their own rules allow it, and a CI-requested build is posted after the merge, so
it is the idle builds that meet a head the tag workflow has not reached.

THE RULE
--------
An idle build is refused while no dev tag on origin names the dev head it would stage. The
refusal is not an attempt: the restore's hourly retry and the converge's one-attempt-per-head
marks are written only when a job is accepted, so the first check after the tag exists builds
the head. Tags are read with ``git ls-remote`` (the same read the agent's tag-following uses),
so the answer is origin's, not the agent's clone. An unread head or an unread tag list refuses
nothing: that is the state in which the audit is the only check, as before. A request a lane or
the operator signed never reaches this rule: only ``agent/idle-restore`` and
``agent/idle-converge`` do. The audit, its accepted kinds and its dev-lane context are unchanged.

ONE LEDGER ROW PER HEAD, AND HOW IT SURVIVES A RESTART
------------------------------------------------------
The first refusal of a head writes one build decision row (``src=build-decision``,
``result=refused``, ``reason=untagged_head``) through ``runtime_deploy_job_reading._emit``, whose
key (repository, head, result, reason) is held in the agent's state directory and marked before
the append: later refusals of the head, the other idle path's refusal and a refusal after a
restart find the key and write nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_failed_head_gate import (
    IDLE_REQUESTERS,
    INFRA_REPOSITORY,
    FailedHeadGate,
)
from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_post_failure_converge import (
    same_revision,
)
from omnibase_infra.protocols.protocol_command import ProtocolCommand
from omnibase_infra.protocols.protocol_post_failure_host import (
    ProtocolPostFailureHost,
)

logger = logging.getLogger(__name__)


class UntaggedHeadGate:
    """The rule's place in the agent's two idle build paths.

    Holds only the host, the ledger writer and the heads already logged; the need comes
    from origin's tags, so a restart of the agent loses nothing it needs.
    """

    def __init__(
        self,
        host: ProtocolPostFailureHost,
        record: Callable[[str, datetime], object],
    ) -> None:
        self._host = host
        self._record = record
        self._logged: set[str] = set()

    def refuses(self, cmd: ProtocolCommand) -> bool:
        """Whether the idle build ``cmd`` is refused; writes the head's one ledger row."""
        if cmd.requested_by not in IDLE_REQUESTERS:
            return False
        host = self._host
        head = host._idle_read_head_ref(INFRA_REPOSITORY)
        tags = host._idle_read_dev_tags(INFRA_REPOSITORY)
        if head is None or tags is None:
            return False
        if any(same_revision(head, tag.sha) for tag in tags):
            return False
        self._record(head, host._idle_converge_now())
        if head not in self._logged:
            self._logged.add(head)
            logger.warning(
                "%s waits: %s %s has no dev-merge tag on origin yet; builds on the "
                "first check after the tag exists friction_type=untagged_head_idle_build",
                cmd.requested_by,
                INFRA_REPOSITORY,
                head[:12],
            )
        return True


class IdleBuildGate:
    """The agent's one question before an idle build: does any rule refuse this head?

    The rules run in order and the first refusal ends the check, so a head that failed its last
    roll is refused (and logged) by the failed-head rule before the tag is read.
    """

    def __init__(
        self, failed_head: FailedHeadGate, untagged_head: UntaggedHeadGate
    ) -> None:
        self._rules = (failed_head, untagged_head)

    def refuses(self, cmd: ProtocolCommand) -> bool:
        """Whether the idle build ``cmd`` is refused by any rule."""
        return any(rule.refuses(cmd) for rule in self._rules)
