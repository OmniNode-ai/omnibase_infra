# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16999: pin every lab binding to a RECORDED ``/v1/models`` probe.

This module is the durable half of OMN-16999. The YAML and table edits it ships
alongside fix today's drift; this file is what stops the third occurrence.

The failure it guards has now happened twice and is invisible to every other
test in the repo. ``served_model_id`` is asserted, everywhere else, only for
*internal* consistency — the lane overlays are checked against
``_AUTHORIZED_BINDINGS`` and ``_AUTHORIZED_BINDINGS`` is checked against
nothing. So when the .201 endpoint is redeployed and starts serving a different
id, every one of those assertions still passes, in perfect agreement, about a
value that is wrong. What actually happens at runtime is that the OMN-16419
fail-closed guard in ``HandlerLlmDelegationCall`` refuses the call
(``model_attribution_mismatch``) and delegation climbs to the metered cloud
ceiling — a silent, billable degradation of the entire local tier.

The fixture is the missing external referent: a recorded readback of what each
endpoint actually served, with the probe command and its controls. Binding a
served id therefore requires producing evidence for it in the same commit.

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

from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    _AUTHORIZED_BINDINGS,
    ACTIVE_BACKEND_KEYS,
    SERVING_BACKEND_KEYS,
    AuthorizedLabBinding,
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


@pytest.mark.parametrize("backend_key", sorted(ACTIVE_BACKEND_KEYS))
def test_every_authorized_binding_has_a_recorded_probe(backend_key: str) -> None:
    """No backend may be authorized without a recorded readback of its endpoint."""
    binding = _AUTHORIZED_BINDINGS[backend_key]
    probes = _probes()
    assert (binding.host, binding.port) in probes, (
        f"{backend_key!r} binds {binding.host}:{binding.port} but "
        f"{_FIXTURE.name} records no probe for that endpoint. Probe it "
        "(GET /v1/models), record the readback, and bind in the same commit — "
        "a served id with no evidence behind it is the OMN-16419/OMN-16999 "
        "drift class."
    )


@pytest.mark.parametrize("backend_key", sorted(SERVING_BACKEND_KEYS))
def test_serving_binding_matches_the_ids_the_endpoint_reported(
    backend_key: str,
) -> None:
    """A rung offered to routing must advertise an id the endpoint really serves.

    This is the assertion that would have caught both flips of the .201 id. It
    reads the probe's ``data[].id`` list, not a restatement of it.
    """
    binding = _AUTHORIZED_BINDINGS[backend_key]
    probe = _probes()[(binding.host, binding.port)]

    assert probe["reachable"] is True, (
        f"{backend_key!r} is marked serving, but the recorded probe of "
        f"{binding.host}:{binding.port} is a FAILURE "
        f"(http_status={probe['http_status']!r}). Either the endpoint came back "
        "— re-probe and update the fixture — or the binding should carry "
        "serving=False."
    )
    served: list[str] = list(probe["served_model_ids"])  # type: ignore[arg-type]
    assert served, "a reachable probe must record at least one served id"
    assert binding.served_model_id in served, (
        f"{backend_key!r} advertises served_model_id "
        f"{binding.served_model_id!r}, but {binding.host}:{binding.port} "
        f"reported {served}. Every delegation call through this backend would "
        "be refused by the OMN-16419 fail-closed attribution guard and would "
        "climb to the metered cloud ceiling. Bind what the endpoint reports."
    )


@pytest.mark.parametrize("backend_key", sorted(SERVING_BACKEND_KEYS))
def test_serving_binding_context_window_matches_the_probe(backend_key: str) -> None:
    """``context_window`` is a probe result too, and drifted with the model id.

    OMN-16419 left 122880 pinned after the endpoint moved to a 131072-token
    deployment. Half-correcting a binding (id fixed, window stale) is its own
    defect class: routing sizes escalation decisions off this number.
    """
    binding = _AUTHORIZED_BINDINGS[backend_key]
    probe = _probes()[(binding.host, binding.port)]
    reported = probe.get("max_model_len")
    if reported is None:
        pytest.skip(f"probe of {binding.host}:{binding.port} records no max_model_len")
    assert binding.context_window == reported, (
        f"{backend_key!r} declares context_window {binding.context_window}, but "
        f"{binding.host}:{binding.port} reported max_model_len {reported!r}."
    )


def test_non_serving_bindings_have_a_failed_probe_on_record() -> None:
    """``serving=False`` is a claim about the world and needs its evidence.

    Without this, ``serving=False`` degrades into a way to silence the probe
    assertions above — mark a rung dark and it stops being checked at all. The
    fixture must carry the failed readback that justifies it.
    """
    dark = sorted(ACTIVE_BACKEND_KEYS - SERVING_BACKEND_KEYS)
    probes = _probes()
    for backend_key in dark:
        binding: AuthorizedLabBinding = _AUTHORIZED_BINDINGS[backend_key]
        probe = probes[(binding.host, binding.port)]
        assert probe["reachable"] is False, (
            f"{backend_key!r} is marked serving=False, but the recorded probe of "
            f"{binding.host}:{binding.port} SUCCEEDED. A rung that answers must "
            "be offered to routing; disabling a live endpoint strands the tier "
            "it was the local rung for."
        )
        assert probe.get("control"), (
            f"the failed probe of {binding.host}:{binding.port} must record its "
            "positive control — an http=000 with no control is indistinguishable "
            "from a probe that never ran (CLAUDE.md rule 16)."
        )


@pytest.mark.parametrize("overlay_path", _LANE_OVERLAYS, ids=lambda p: p.name)
def test_every_lab_lane_overlay_agrees_with_the_probe_fixture(
    overlay_path: Path,
) -> None:
    """One assertion covering dev, judge and lakshman at once.

    The three overlays are validated against ``_AUTHORIZED_BINDINGS`` by
    set-equality, so they cannot disagree with the table — but they can all be
    wrong together, which is exactly what happened. Driving this off the table
    (rather than restating ids per lane) means a future lane is covered the day
    it is added, with no test edit.
    """
    overlay = ModelBifrostLaneOverlay.model_validate(
        yaml.safe_load(overlay_path.read_text())
    )
    probes = _probes()
    for binding in overlay.backends:
        authorized = _AUTHORIZED_BINDINGS[binding.backend_key]
        probe = probes[(authorized.host, authorized.port)]
        if not binding.serving:
            assert probe["reachable"] is False
            continue
        assert binding.advertised_model in list(probe["served_model_ids"]), (  # type: ignore[arg-type]
            f"{overlay_path.name} binds {binding.backend_key!r} to "
            f"{binding.advertised_model!r}, which {authorized.host}:"
            f"{authorized.port} does not serve."
        )
