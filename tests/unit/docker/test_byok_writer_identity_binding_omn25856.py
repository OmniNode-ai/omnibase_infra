# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25856: the dev runtime carries the BYOK writer identity.

The intake writes a customer's provider key with its own Infisical machine
identity (``INFISICAL_TENANT_CREDENTIAL_WRITER_CLIENT_ID`` / ``_CLIENT_SECRET``)
because the runtime process identity is read-only on ``/tenant-inference-credentials``
and a create with it answers 403. Both processes that can run the intake on the
dev lane bind the pair from operator env names; a value is only ever an
interpolation, never a literal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.helpers.lane_overlay_fixtures import load_compose
from tests.unit.docker.test_dev_tenant_projection_fold_omn21288 import (
    dev_stack_services,
)

pytestmark = pytest.mark.unit

DEV_LANE = (
    Path(__file__).resolve().parents[3] / "docker" / "docker-compose.dev-lane.yml"
)

WRITER_VARS = {
    "INFISICAL_TENANT_CREDENTIAL_WRITER_CLIENT_ID": (
        "RUNTIME_EFFECTS_INFISICAL_TENANT_CREDENTIAL_WRITER_CLIENT_ID"
    ),
    "INFISICAL_TENANT_CREDENTIAL_WRITER_CLIENT_SECRET": (
        "RUNTIME_EFFECTS_INFISICAL_TENANT_CREDENTIAL_WRITER_CLIENT_SECRET"
    ),
}


def _assert_bound(environment: dict[str, str]) -> None:
    for name, operator_name in WRITER_VARS.items():
        assert environment.get(name) == f"${{{operator_name}:-}}", name


def test_folded_main_binds_the_writer_identity() -> None:
    _assert_bound(dev_stack_services()["omninode-runtime"]["environment"])


def test_runtime_effects_binds_the_writer_identity() -> None:
    _assert_bound(load_compose(DEV_LANE)["services"]["runtime-effects"]["environment"])


def test_no_writer_value_is_a_literal() -> None:
    for service in (
        dev_stack_services()["omninode-runtime"],
        load_compose(DEV_LANE)["services"]["runtime-effects"],
    ):
        for name in WRITER_VARS:
            assert re.fullmatch(r"\$\{\w+:-\}", service["environment"][name]), name


def test_the_writer_pair_never_replaces_the_process_identity() -> None:
    env = dev_stack_services()["omninode-runtime"]["environment"]
    assert env["INFISICAL_CLIENT_ID"] == "${RUNTIME_EFFECTS_INFISICAL_CLIENT_ID:-}"
    assert (
        env["INFISICAL_CLIENT_SECRET"] == "${RUNTIME_EFFECTS_INFISICAL_CLIENT_SECRET:-}"
    )
