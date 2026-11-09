# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The runtime port and its protocol both accept ``probe_ladder`` (OMN-17292, OMN-23162)."""

import inspect

import pytest

from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.integration


def test_the_port_and_the_protocol_both_declare_probe_ladder_defaulted_to_none() -> (
    None
):
    runtime = inspect.signature(RuntimeDelegationDispatchPort.dispatch).parameters
    protocol = inspect.signature(ProtocolDelegationDispatchPort.dispatch).parameters
    assert runtime["probe_ladder"].default is None
    assert protocol["probe_ladder"].default is None
    assert str(runtime["probe_ladder"].annotation) == str(
        protocol["probe_ladder"].annotation
    )
