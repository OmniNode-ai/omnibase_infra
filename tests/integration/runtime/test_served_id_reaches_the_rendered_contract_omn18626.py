# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18626: the served id the endpoint ANSWERED WITH reaches the wire.

The unit tests beside this one each check one link of the chain. This checks
that the chain joins up, end to end, on the artifact routing actually consumes:

    recorded /v1/models readback
      -> _AUTHORIZED_BINDINGS
        -> the committed lane overlay
          -> the rendered contract's ``model_name``
            -> the string POSTed as ``model``

Why that end-to-end assertion is the one worth having. On 2026-09-17 every
link in this chain was internally consistent and the whole chain was wrong: the
table agreed with three overlays, the overlays agreed with a CI assertion that
restated the same literal, and all of them named a model ``.201:8000`` had
stopped serving about four hours earlier. Every static test passed. What failed
was a real delegation, with ``HTTP 404 "The model `Qwen3.6-35B-A3B` does not
exist."`` on its local rung, and a climb to a metered cloud provider.

The only site in this repository with an EXTERNAL referent is
``tests/fixtures/bifrost_served_models_probe.json`` -- a transcript of what the
endpoint answered. So this file starts there, not at the table.

It also pins the cross-repo refusal that makes this a two-repo change. The
renderer rejects a base contract whose ``model_name`` disagrees with the
overlay's ``served_model_id``. That refusal is what stops a half-applied repoint
from shipping, and it is the reason this change and its omnimarket twin have to
merge together: in the window between them a lane rebuild does not route badly,
it fails to render its delegation contract at all. Deleting or loosening that
comparison would make a half-applied repoint silent again, which is the failure
this ticket exists to remove.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    _AUTHORIZED_BINDINGS,
)
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_OVERLAY_DIR = _ROOT / "docker" / "lane-overlays"
_PROBE_FIXTURE = _ROOT / "tests" / "fixtures" / "bifrost_served_models_probe.json"

_LAB_OVERLAYS = ("dev.bifrost.yaml", "judge.bifrost.yaml", "lakshman.bifrost.yaml")
_LOCAL_201_BACKENDS = ("local-coder", "local-heavy-reasoning")
_LOCAL_201_MODELS_URL = "http://192.168.86.201:8000/v1/models"  # onex-allow-internal-ip OMN-18626 reason="keyed to the recorded lab probe"

#: The env hint each backend carries in the base contract. The renderer strips
#: these -- the overlay owns the real endpoint -- but a base contract without
#: them is rejected.
_ENDPOINT_URL_ENV = {
    "local-coder": "LLM_CODER_URL",
    "local-heavy-reasoning": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
    "local-ds-v4-flash": "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL",
}


def _recorded_served_ids(models_url: str) -> list[str]:
    """The ids the endpoint ANSWERED with, from the recorded readback."""
    probes = json.loads(_PROBE_FIXTURE.read_text(encoding="utf-8"))["probes"]
    for probe in probes:
        if probe["endpoint"] == models_url:
            return list(probe["served_model_ids"])
    raise AssertionError(
        f"{_PROBE_FIXTURE.name} carries no readback for {models_url!r}. "
        "Every authorized endpoint must have one -- an unprobed binding is a "
        "value with no external referent, which is how this drifts."
    )


def _write_base_contract(path: Path, *, model_name_for: dict[str, str]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": backend_id,
                        "model_name": model_name_for[backend_id],
                        "endpoint_url_env": env_name,
                        "required": True,
                    }
                    for backend_id, env_name in _ENDPOINT_URL_ENV.items()
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _authorized_model_names() -> dict[str, str]:
    return {
        backend_id: _AUTHORIZED_BINDINGS[backend_id].served_model_id
        for backend_id in _ENDPOINT_URL_ENV
    }


@pytest.mark.parametrize("overlay_name", _LAB_OVERLAYS)
def test_the_rendered_wire_model_is_an_id_the_endpoint_answered_with(
    overlay_name: str, tmp_path: Path
) -> None:
    """End to end: recorded probe -> table -> overlay -> rendered model_name.

    Not "the overlay agrees with the table" -- that was true on 2026-09-17 while
    both were wrong. This asserts the value the runtime will POST is one the
    endpoint said it serves.
    """
    served = _recorded_served_ids(_LOCAL_201_MODELS_URL)
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source, model_name_for=_authorized_model_names())

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=_OVERLAY_DIR / overlay_name,
        target_path=target,
        environ={},
    )
    assert rendered == target

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    for backend_id in _LOCAL_201_BACKENDS:
        wire_model = by_id[backend_id]["model_name"]
        assert wire_model in served, (
            f"{overlay_name} renders {backend_id!r} onto the wire as "
            f"{wire_model!r}, which the recorded readback of "
            f"{_LOCAL_201_MODELS_URL} does not list ({served}). vLLM refuses an "
            f"unknown model by name, so this renders a rung that 404s on its "
            f"first call. Re-probe the endpoint and update the probe fixture, "
            f"the binding table and every lane overlay in ONE commit."
        )


def test_a_base_contract_naming_a_different_model_is_refused(tmp_path: Path) -> None:
    """The cross-repo cross-check still bites, and is why this is two PRs.

    ``omnimarket``'s ``configs/bifrost_delegation.yaml`` supplies the base
    ``model_name``; this repo's overlay supplies ``served_model_id``. The
    renderer refuses them when they disagree. That refusal is the mechanism that
    turns a half-applied repoint into a loud failure instead of a silent one, so
    it is asserted here BEHAVIOURALLY -- drive the real renderer with a real
    committed overlay and a base that names the retired id.
    """
    stale = dict(_authorized_model_names())
    stale["local-coder"] = "Qwen3.6-35B-A3B"

    source = tmp_path / "base.yaml"
    _write_base_contract(source, model_name_for=stale)

    with pytest.raises(ProtocolConfigurationError) as excinfo:
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=_OVERLAY_DIR / "dev.bifrost.yaml",
            target_path=tmp_path / "rendered.yaml",
            environ={},
        )

    message = str(excinfo.value)
    assert "local-coder" in message
    assert "Qwen3.6-35B-A3B" in message
    assert _AUTHORIZED_BINDINGS["local-coder"].served_model_id in message
    assert not (tmp_path / "rendered.yaml").exists(), (
        "a refused render must leave no artifact behind -- a partial write is a "
        "contract the runtime would read"
    )


def test_the_null_model_name_escape_is_available_and_is_not_how_this_is_fixed(
    tmp_path: Path,
) -> None:
    """Positive control on the refusal above, and a standing warning.

    ``_merge_lane_overlay`` skips its comparison when the base declares
    ``model_name: null``. That branch is real and this test proves it, so the
    refusal in the test above is a genuine comparison rather than a renderer
    that rejects every base it is given.

    It is also the cheap way out of the two-repo sequencing this ticket
    describes, and it must not be taken: with a null base the two repos can no
    longer disagree because one of them has stopped saying anything, and a
    repoint applied to one repo alone goes silent again. If you are reading this
    because you are tempted, the answer is to land both PRs together.
    """
    nulled = dict(_authorized_model_names())
    nulled["local-coder"] = None  # type: ignore[assignment]

    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source, model_name_for=nulled)  # type: ignore[arg-type]

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=_OVERLAY_DIR / "dev.bifrost.yaml",
        target_path=target,
        environ={},
    )
    assert rendered == target

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    # The overlay still supplies the value, which is exactly why the escape is
    # dangerous: the artifact looks correct while the cross-check is gone.
    assert (
        by_id["local-coder"]["model_name"]
        == _AUTHORIZED_BINDINGS["local-coder"].served_model_id
    )
