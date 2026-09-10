# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The publisher must know `ledger_readback`, the THIRD key of this class.

OMN-16964. omnimarket#2437 (`9d14eecf`, merged 2026-09-10T04:17:04Z) added a
``ledger_readback`` block to the dev lane of omnimarket
``config/ci_bus_lanes.yaml``. ``scripts/trigger_rebuild_on_merge.py`` sparse-
checks that same file and validates it with ``extra="forbid"``, and it had not
learned the key, so it rejected the WHOLE overlay::

    lanes.dev.ledger_readback
      Extra inputs are not permitted

That is not a cosmetic red. The trigger is what publishes the
rebuild-requested event, so while it is broken the deploy agent does not
rebuild the `.201` dev lane on merge — which is the automatic lab pass of
CLAUDE.md rule 24(a). Every runtime-affecting merge in the window lands on
`dev` with no lab pass behind it. Confirmed live on two unrelated branches one
and two minutes after that merge (omnibase_infra#3379 at 04:18:09Z and the
OMN-18108 lane at 04:19:24Z), with the last green trigger run predating it at
02:08:49Z.

THIS IS THE THIRD OCCURRENCE and the model's own docstring names the first
two: the transport keys under OMN-18012 (run 34160709151) and
``projection_readback`` under OMN-18060. The rule that docstring states is the
one followed here — *"Modelling the key is therefore the fix; loosening the
model is not."* No ``extra="allow"``, no ignored key: the whole point of the
strictness is that an unknown key is indistinguishable from a typo that would
route a publisher to a default it never declared.

The structural cause is that the strict half of the contract lives in a
different repository from the file it validates, so a key added in omnimarket
is only discovered later, by an unrelated omnibase_infra PR. These tests are
the narrow guard; the cross-repo parity gate that would move the failure to
the PR that causes it is filed separately.

Hermetic: the overlay fixtures are written to tmp_path. No network, no
sparse checkout, no bus.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "trigger_rebuild_on_merge.py"
)


def _import_trigger_module():
    """Import the publisher by file path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "trigger_rebuild_on_merge_ledger_readback", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _overlay(tmp_path: Path, dev_lane: dict[str, object]) -> Path:
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(
        yaml.safe_dump({"default": "inmemory", "lanes": {"dev": dev_lane}}),
        encoding="utf-8",
    )
    return path


_LIVE_DEV_LANE: dict[str, object] = {
    "broker": "example.invalid:19092",
    "security_protocol": "SASL_PLAINTEXT",
    "sasl_mechanism": "SCRAM-SHA-256",
    "projection_readback": {"dsn_env": "CHAIN_CANARY_PROJECTION_DSN"},
    "ledger_readback": {"dsn_env": "CHAIN_CANARY_PROJECTION_DSN"},
}


@pytest.mark.unit
def test_the_live_overlay_shape_is_accepted(tmp_path: Path) -> None:
    """The dev lane as omnimarket dev actually declares it, both blocks."""
    mod = _import_trigger_module()

    model = mod.load_ci_bus_overlay(_overlay(tmp_path, dict(_LIVE_DEV_LANE)))

    lane = model.lanes["dev"]
    assert lane.ledger_readback is not None
    assert lane.ledger_readback.dsn_env == "CHAIN_CANARY_PROJECTION_DSN"
    assert lane.projection_readback is not None


@pytest.mark.unit
def test_the_block_stays_optional(tmp_path: Path) -> None:
    """A lane that declares no ledger readback is still valid.

    Every lane but `dev` declares none, so a required field here would reject
    the live overlay just as surely as an unknown key did.
    """
    lane_without = {
        key: value for key, value in _LIVE_DEV_LANE.items() if key != "ledger_readback"
    }
    mod = _import_trigger_module()

    model = mod.load_ci_bus_overlay(_overlay(tmp_path, lane_without))

    assert model.lanes["dev"].ledger_readback is None


@pytest.mark.unit
def test_a_dsn_value_where_a_name_belongs_is_refused(tmp_path: Path) -> None:
    """The block carries a NAME. A connection string is refused by shape."""
    lane = dict(_LIVE_DEV_LANE)
    lane["ledger_readback"] = {
        "dsn_env": "postgresql://someone:somevalue@example.invalid:5432/somedb"
    }
    mod = _import_trigger_module()

    with pytest.raises(Exception, match="dsn_env"):
        mod.load_ci_bus_overlay(_overlay(tmp_path, lane))


@pytest.mark.unit
def test_an_empty_name_is_refused(tmp_path: Path) -> None:
    lane = dict(_LIVE_DEV_LANE)
    lane["ledger_readback"] = {"dsn_env": ""}
    mod = _import_trigger_module()

    with pytest.raises(Exception, match="dsn_env"):
        mod.load_ci_bus_overlay(_overlay(tmp_path, lane))


@pytest.mark.unit
def test_an_unknown_key_inside_the_block_is_still_refused(tmp_path: Path) -> None:
    """Learning the key must not weaken the strictness that motivated it."""
    lane = dict(_LIVE_DEV_LANE)
    lane["ledger_readback"] = {
        "dsn_env": "CHAIN_CANARY_PROJECTION_DSN",
        "dsn": "postgresql://someone:somevalue@example.invalid:5432/somedb",
    }
    mod = _import_trigger_module()

    with pytest.raises(Exception):
        mod.load_ci_bus_overlay(_overlay(tmp_path, lane))


@pytest.mark.unit
def test_a_genuinely_unknown_lane_key_is_still_refused(tmp_path: Path) -> None:
    """The typo this model exists to catch must still be caught.

    This is the assertion that keeps the fix honest: the remedy for a missing
    key is to model it, never to relax `extra`.
    """
    lane = dict(_LIVE_DEV_LANE)
    lane["ledger_raedback"] = {"dsn_env": "CHAIN_CANARY_PROJECTION_DSN"}
    mod = _import_trigger_module()

    with pytest.raises(Exception, match=r"ledger_raedback|Extra inputs"):
        mod.load_ci_bus_overlay(_overlay(tmp_path, lane))
