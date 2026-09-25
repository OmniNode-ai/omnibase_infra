# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16999: pin every lab binding to a RECORDED ``/v1/models`` probe.

This module is the durable half of OMN-16999. The YAML and table edits it ships
alongside fix today's drift; this file is what stops the third occurrence.

The failure it guards has now happened twice and is invisible to every other
test in the repo. ``served_model_id`` is asserted, everywhere else, only for
*internal* consistency — the renderer checks a lane overlay's served id against
the base contract's ``model_name``, and neither is checked against the world.
So when the .201 endpoint is redeployed and starts serving a different id,
every one of those assertions still passes, in perfect agreement, about a value
that is wrong. What actually happens at runtime is that the OMN-16419
fail-closed guard in ``HandlerLlmDelegationCall`` refuses the call
(``model_attribution_mismatch``) and delegation climbs to the metered cloud
ceiling — a silent, billable degradation of the entire local tier.

The fixture is the missing external referent: a recorded readback of what each
endpoint actually served, with the probe command and its controls. Binding a
served id therefore requires producing evidence for it in the same commit.

OMN-17099 moved the lab bindings out of the product. They used to live in a
hardcoded authorization table in ``src/``, which this module pinned; they now
live only in the committed lab lane overlays (``docker/lane-overlays``), so this
module pins every binding in every lab overlay directly. A backend a lane ADDS
is covered the day it is added, with no test edit: it fails here until someone
records a probe of its endpoint.

What this deliberately does NOT do: probe the network. A unit test that called
the endpoint would be green only when the lab happens to be up, would be
nondeterministic in CI, and would re-introduce the exact "empty result reads as
success" hazard CLAUDE.md rule 16 warns about. The obligation this enforces is
*evidentiary* — that a human ran the probe and recorded it — which is the half
no amount of test automation can supply.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    ModelBifrostLaneBackendBinding,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _ROOT / "fixtures" / "bifrost_served_models_probe.json"
_LANE_OVERLAYS = sorted(
    (_ROOT.parent / "docker" / "lane-overlays").glob("*.bifrost.yaml")
)

LabBinding = tuple[str, ModelBifrostLaneBackendBinding]


def _probes() -> dict[tuple[str, int], dict[str, object]]:
    """Recorded probes, keyed by the ``(host, port)`` they were taken against."""
    payload = json.loads(_FIXTURE.read_text())
    by_authority: dict[tuple[str, int], dict[str, object]] = {}
    for probe in payload["probes"]:
        parsed = urlsplit(str(probe["endpoint"]))
        assert parsed.hostname is not None and parsed.port is not None, (
            f"probe endpoint {probe['endpoint']!r} must carry an explicit host and port"
        )
        by_authority[(parsed.hostname, parsed.port)] = probe
    return by_authority


def _authority(binding: ModelBifrostLaneBackendBinding) -> tuple[str, int]:
    parsed = urlsplit(binding.endpoint_url)
    assert parsed.hostname is not None and parsed.port is not None, (
        f"{binding.backend_key!r} binds {binding.endpoint_url!r}; a lab binding "
        "must carry an explicit host and port so it can be matched to a probe"
    )
    return parsed.hostname, parsed.port


def _lab_bindings() -> list[LabBinding]:
    """Every ``(overlay file name, binding)`` pair across the lab lane overlays."""
    pairs: list[LabBinding] = []
    for path in _LANE_OVERLAYS:
        overlay = ModelBifrostLaneOverlay.model_validate(
            yaml.safe_load(path.read_text())
        )
        if overlay.locale is not EnumBifrostLaneLocale.LAB:
            continue
        pairs.extend((path.name, binding) for binding in overlay.backends)
    return pairs


def _pair_id(pair: LabBinding) -> str:
    return f"{pair[0]}:{pair[1].backend_key}"


_LAB_BINDINGS = _lab_bindings()
_SERVING = [pair for pair in _LAB_BINDINGS if pair[1].serving]
_DARK = [pair for pair in _LAB_BINDINGS if not pair[1].serving]


def test_the_lab_overlays_bind_a_serving_backend() -> None:
    """Positive control: an empty parametrization would pass every test below.

    OMN-19251 removed the last ``serving: false`` lab binding (the dead
    ``local-ds-v4-flash`` rung on the retired .200:8101 host), so there is no
    longer a live dark-binding class to control for here. The parametrized
    ``test_non_serving_bindings_have_a_failed_probe_on_record`` below still
    covers one correctly, as an empty parametrization, the moment one exists
    again.
    """
    assert _SERVING, "no serving lab binding found under docker/lane-overlays"


@pytest.mark.parametrize("pair", _LAB_BINDINGS, ids=_pair_id)
def test_every_lab_binding_has_a_recorded_probe(pair: LabBinding) -> None:
    """No backend may be bound on a lab lane without a recorded readback."""
    overlay_name, binding = pair
    host, port = _authority(binding)
    assert (host, port) in _probes(), (
        f"{overlay_name} binds {binding.backend_key!r} to {host}:{port} but "
        f"{_FIXTURE.name} records no probe for that endpoint. Probe it "
        "(GET /v1/models), record the readback, and bind in the same commit — "
        "a served id with no evidence behind it is the OMN-16419/OMN-16999 "
        "drift class."
    )


@pytest.mark.parametrize("pair", _SERVING, ids=_pair_id)
def test_serving_binding_matches_the_ids_the_endpoint_reported(
    pair: LabBinding,
) -> None:
    """A rung offered to routing must advertise an id the endpoint really serves.

    This is the assertion that would have caught both flips of the .201 id. It
    reads the probe's ``data[].id`` list, not a restatement of it.
    """
    overlay_name, binding = pair
    host, port = _authority(binding)
    probe = _probes()[(host, port)]

    assert probe["reachable"] is True, (
        f"{overlay_name} marks {binding.backend_key!r} serving, but the recorded "
        f"probe of {host}:{port} is a FAILURE "
        f"(http_status={probe['http_status']!r}). Either the endpoint came back "
        "— re-probe and update the fixture — or the binding should carry "
        "serving: false."
    )
    served: list[str] = list(probe["served_model_ids"])  # type: ignore[call-overload]
    assert served, "a reachable probe must record at least one served id"
    assert binding.advertised_model in served, (
        f"{overlay_name} binds {binding.backend_key!r} to served_model_id "
        f"{binding.advertised_model!r}, but {host}:{port} reported {served}. "
        "Every delegation call through this backend would be refused by the "
        "OMN-16419 fail-closed attribution guard and would climb to the metered "
        "cloud ceiling. Bind what the endpoint reports."
    )


@pytest.mark.parametrize("pair", _SERVING, ids=_pair_id)
def test_serving_binding_context_window_matches_the_probe(pair: LabBinding) -> None:
    """``context_window`` is a probe result too, and drifted with the model id.

    OMN-16419 left 122880 pinned after the endpoint moved to a 131072-token
    deployment. Half-correcting a binding (id fixed, window stale) is its own
    defect class: routing sizes escalation decisions off this number.
    """
    overlay_name, binding = pair
    host, port = _authority(binding)
    reported = _probes()[(host, port)].get("max_model_len")
    if reported is None:
        pytest.skip(f"probe of {host}:{port} records no max_model_len")
    assert binding.context_window == reported, (
        f"{overlay_name} declares context_window {binding.context_window} for "
        f"{binding.backend_key!r}, but {host}:{port} reported max_model_len "
        f"{reported!r}."
    )


@pytest.mark.parametrize("pair", _DARK, ids=_pair_id)
def test_non_serving_bindings_have_a_failed_probe_on_record(pair: LabBinding) -> None:
    """``serving: false`` is a claim about the world and needs its evidence.

    Without this, ``serving: false`` degrades into a way to silence the probe
    assertions above — mark a rung dark and it stops being checked at all. The
    fixture must carry the failed readback that justifies it.
    """
    overlay_name, binding = pair
    host, port = _authority(binding)
    probe = _probes()[(host, port)]
    assert probe["reachable"] is False, (
        f"{overlay_name} marks {binding.backend_key!r} serving: false, but the "
        f"recorded probe of {host}:{port} SUCCEEDED. A rung that answers must "
        "be offered to routing; disabling a live endpoint strands the tier it "
        "was the local rung for."
    )
    assert probe.get("control"), (
        f"the failed probe of {host}:{port} must record its positive control — "
        "an http=000 with no control is indistinguishable from a probe that "
        "never ran (CLAUDE.md rule 16)."
    )
