# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pool driver's numbers, prefixes and git host come from its overlay (OMN-22153)."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.models.prepr_pool import ModelPreprPoolOverlay
from tests.helpers.prepr_pool_overlay import POOL_CONFIG_ROOT, pool_overlay_env
from tests.unit.scripts.test_prepr_runtime_pool_omn18893 import POOL_PY, pool

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("pool_overlay_env")]

OVERLAY_FILE = POOL_CONFIG_ROOT / "config/onex/overlays/prepr_runtime_pool/overlay.yaml"


def test_the_overlay_loads_through_the_driver() -> None:
    overlay = pool.overlay()
    assert overlay.probe_poll_s > 0
    assert overlay.remote_command_prefix.startswith("export PATH=")


def test_an_overlay_missing_a_key_is_refused() -> None:
    """Negative control: every key is required, none has a default."""
    raw = yaml.safe_load(OVERLAY_FILE.read_text(encoding="utf-8"))
    del raw["probe_timeout_s"]
    with pytest.raises(ValidationError, match="probe_timeout_s"):
        ModelPreprPoolOverlay.model_validate(raw)


def test_an_unknown_overlay_key_is_refused() -> None:
    raw = yaml.safe_load(OVERLAY_FILE.read_text(encoding="utf-8"))
    raw["unreviewed_knob"] = 1
    with pytest.raises(ValidationError, match="unreviewed_knob"):
        ModelPreprPoolOverlay.model_validate(raw)


def test_the_clone_base_joins_the_overlay_host_and_the_layer_map_org() -> None:
    base = pool.overlay().clone_base_url()
    assert base.startswith("https://git.example.invalid/")
    assert base.count("/") == 3
    assert base.rsplit("/", 1)[1]


def test_the_driver_injects_the_overlay_values_the_script_reads() -> None:
    assert {"CLONE_BASE_URL", "PROBE_POLL_S"} <= set(pool.RESERVED_PARAMS)


def test_the_driver_holds_no_pool_number_of_its_own() -> None:
    text = POOL_PY.read_text(encoding="utf-8")
    for needle in ("timeout=30", "timeout=60", "timeout=120", "Path.home()"):
        assert needle not in text
