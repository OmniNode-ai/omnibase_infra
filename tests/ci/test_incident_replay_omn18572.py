# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay: the onex-api the lab lane actually ran on 2026-09-17 (OMN-18572).

THE INCIDENT, NOT A RECONSTRUCTION OF IT
----------------------------------------
`omninode_infra#1523` merged at 09:00:53Z. At 11:15Z the `.201` compose dev
lane's `onex-api` was still running
`onex-lab/omnicloud-core:f37261c2-20260917T050425Z` -- the PARENT of that squash.
Tenant creation on the lab was impossible for the whole window.

The artifact replayed here is the label map of THAT IMAGE, read from the lab
host's own docker daemon with
`docker image inspect ... --format '{{json .Config.Labels}}'`. It is the single
JSON token `null`: that image was built by the lab-overlay applier BEFORE
OMN-18113 taught it to stamp `org.opencontainers.image.revision`, so the lane
carried no provenance at all.

`null` is the case the guard has to get right, and it is the case a synthetic
fixture would have missed. A test author writing a fixture by hand would write
`{}` -- an empty object. Docker does not produce `{}` for an unlabelled image;
it produces `null`, and `{{ index .Config.Labels "x" }}` against it is a template
ERROR with a non-zero exit, which is the exact trap that made the OMN-18113
repoint refuse a lane it should have advanced (`template parsing error: map has
no entry for key "Labels"`). A guard that treated an unreadable label map as
"nothing changed" would have reported that lane converged.

THE DISCRIMINATOR IS LOAD-BEARING
---------------------------------
A guard that rejected every label map would satisfy the reject case trivially
and red every delivery forever. So the same guard is required to ACCEPT the
second captured artifact: the live `onex-api` CONTAINER's label map after the
11:41:55Z delivery, carrying
`org.opencontainers.image.revision=99fdbd375f3b6c17d564b588f505754160b1d1f2`
alongside the nineteen compose and omninode labels a real container has. Both
files are bytes from the lab host, re-fetchable at the locators in
`tests/incident_replays/registry.yaml`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "check_lane_onex_api_revision.py"
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "omn18572"

#: The image the lane ran at 11:15Z, and the one it ran after the delivery.
STALE_LABELS = _FIXTURES / "onex-api-labels-f37261c2.json.captured"
DELIVERED_LABELS = _FIXTURES / "onex-api-container-labels-99fdbd37.json.captured"

STALE_SHA256 = "38e0b9de817f645c4bec37c0d4a3e58baecccb040f5718dc069a72c7385a0bed"
DELIVERED_SHA256 = "cbb9261bc778fb93c778bf6e0ab6257ac1ce17d7ba42eb10f3ab8967094cd6ae"

MERGED = "99fdbd375f3b6c17d564b588f505754160b1d1f2"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("_onex_api_replay", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_onex_api_replay"] = module
    spec.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_captured_artifacts_are_the_bytes_the_registry_records() -> None:
    """A fixture whose digest moved is no longer the artifact that failed."""
    assert _digest(STALE_LABELS) == STALE_SHA256
    assert _digest(DELIVERED_LABELS) == DELIVERED_SHA256


def test_docker_renders_an_unlabelled_image_as_null_not_empty_object() -> None:
    """The premise of the reject case, asserted on the bytes rather than assumed.

    If this ever reads `{}`, the reject case below is testing a shape docker
    does not produce and the replay has stopped replaying anything.
    """
    assert STALE_LABELS.read_text(encoding="utf-8").strip() == "null"


def test_the_real_guard_rejects_the_image_the_lane_was_actually_running() -> None:
    module = _load()
    payload = json.loads(STALE_LABELS.read_text(encoding="utf-8"))
    with pytest.raises(ValueError) as excinfo:
        module.parse_onex_api_revision(payload)
    # The refusal must say WHY, or it is a refusal nobody can act on.
    assert "UNKNOWN" in str(excinfo.value)


def test_the_same_guard_accepts_the_delivered_container() -> None:
    """The discriminator. Without it, a guard that refused everything would pass."""
    module = _load()
    payload = json.loads(DELIVERED_LABELS.read_text(encoding="utf-8"))
    assert module.parse_onex_api_revision(payload) == MERGED


def test_the_delivered_capture_is_a_real_container_not_a_bare_image() -> None:
    """Positive control on the discriminator's own fidelity.

    An image inspect and a container inspect are different labelled surfaces,
    and the guard reads the CONTAINER. A discriminator taken from the image
    would prove the guard works on a surface it never sees in production.
    """
    payload = json.loads(DELIVERED_LABELS.read_text(encoding="utf-8"))
    assert payload["com.docker.compose.service"] == "onex-api"
    assert payload["com.docker.compose.project"] == "omnibase-infra"


def test_the_lane_fence_would_have_read_the_right_compose_project() -> None:
    """The captured container is the dev lane's, so the fence passes on it."""
    module = _load()
    payload = json.loads(DELIVERED_LABELS.read_text(encoding="utf-8"))
    module.assert_lane_fence(
        payload["com.docker.compose.project"], "onex-api", "omnibase-infra"
    )
