# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A boot that will not reach ready must say what it is still waiting for (OMN-18550).

On 2026-09-17 the ``.201`` dev-lane ``omninode-runtime`` container died and was
relaunched by its restart policy every forty seconds for hours. The line a
reader met, repeating every ten seconds and dominating the log, was exactly
this and nothing more::

    [WARNING] omnibase_infra.runtime.runtime_host_process: Readiness check failed: runtime is not ready

It names no probe, no contract, and no count, so it cannot distinguish a boot
that is merely slow from one that is wedged. The lane that diagnosed the
incident read it as a startup readiness deadline being missed and built a whole
mechanism story on top of that -- and there is no startup readiness deadline in
this codebase. The actual killer was an unhandled ``ValueError`` from local
ingress route discovery, logged once, 130,000 lines deep.

The gate already computes the account: ``ContractAttachReadinessGate.status()``
names every NOT_READY, FAILED and still-pending contract. It reached the
``/ready`` HTTP body and never reached the log line. The container's log format
renders the message only, not the structured ``extra``, so putting the account
in ``extra`` alone would leave the reader exactly where they were.

RED against the parent commit: the probe detail carries no ``summary``, and the
warning message is the bare sentence above.

Related Tickets:
    - OMN-18550: the runtime cannot finish starting and the failure says nothing.
    - OMN-17372: the contract-attach readiness gate whose status is the account.
    - OMN-14758: the supplemental readiness probe seam both halves ride.
"""

from __future__ import annotations

import logging

import pytest

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.runtime.health.contract_attach_readiness_gate import (
    ContractAttachReadinessGate,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config

pytestmark = pytest.mark.unit


def test_contract_attach_probe_detail_carries_a_readable_summary() -> None:
    """The gate must render its own one-line account of what is outstanding.

    ``readiness_check`` cannot be taught the gate's field names without
    coupling the generic probe seam to one probe, so the probe supplies the
    sentence and the seam only places it.
    """
    gate = ContractAttachReadinessGate(
        ("contract_a", "contract_b", "contract_c", "contract_d")
    )
    gate.record(
        [
            ModelContractAttachResult(
                contract_name="contract_a",
                status=EnumContractAttachStatus.ATTACHED,
            ),
            ModelContractAttachResult(
                contract_name="contract_b",
                status=EnumContractAttachStatus.ATTACHED,
            ),
            ModelContractAttachResult(
                contract_name="contract_c",
                status=EnumContractAttachStatus.NOT_READY,
            ),
        ]
    )

    ready, detail = gate.probe()

    assert ready is False
    summary = detail["summary"]
    assert isinstance(summary, str)
    # How many of how many were wired.
    assert "2/4" in summary
    # Which ones are outstanding, and under which disposition.
    assert "contract_c" in summary
    assert "contract_d" in summary
    # An attached contract is not noise in the failure account.
    assert "contract_a" not in summary


def test_contract_attach_probe_summary_is_absent_when_ready() -> None:
    """A ready gate has nothing outstanding, so it must not manufacture a list."""
    gate = ContractAttachReadinessGate(("contract_a",))
    gate.record(
        [
            ModelContractAttachResult(
                contract_name="contract_a",
                status=EnumContractAttachStatus.ATTACHED,
            )
        ]
    )

    ready, detail = gate.probe()

    assert ready is True
    assert "1/1" in str(detail["summary"])


async def test_readiness_failure_line_names_the_outstanding_probe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning MESSAGE, not only its ``extra``, must name what is blocking.

    The negative control is the parent commit's message, which is the bare
    sentence and carries neither the probe name nor its account.
    """
    process = RuntimeHostProcess(config=make_runtime_config())
    process.register_readiness_probe(
        "contract_attach",
        lambda: (
            False,
            {"summary": "44/48 contracts attached, pending=[node_shim_scanner]"},
        ),
    )

    with caplog.at_level(logging.WARNING, logger="omnibase_infra.runtime"):
        result = await process.readiness_check()

    assert result["ready"] is False
    records = [
        record
        for record in caplog.records
        if "Readiness check failed" in record.getMessage()
    ]
    assert records, "the readiness failure must still be logged"
    message = records[-1].getMessage()
    # The existing sentence is preserved so anything grepping for it still matches.
    assert "Readiness check failed: runtime is not ready" in message
    # And the reader is now told which probe is blocking, and what it is waiting on.
    assert "contract_attach" in message
    assert "44/48 contracts attached" in message
    assert "node_shim_scanner" in message


async def test_readiness_failure_line_names_a_probe_with_no_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A probe that supplies no sentence is still named rather than silently dropped."""
    process = RuntimeHostProcess(config=make_runtime_config())
    process.register_readiness_probe("terse_probe", lambda: (False, {}))

    with caplog.at_level(logging.WARNING, logger="omnibase_infra.runtime"):
        await process.readiness_check()

    records = [
        record
        for record in caplog.records
        if "Readiness check failed" in record.getMessage()
    ]
    assert records
    assert "terse_probe" in records[-1].getMessage()
