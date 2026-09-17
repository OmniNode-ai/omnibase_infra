# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18570: drive every COMMITTED lab overlay through the real renderer.

The unit test beside this one
(``tests/unit/runtime/test_bifrost_parameter_count_matches_served_id.py``)
checks the authorized binding table against the served model id. That is the
referent half. This is the reach half: it proves the corrected value is
load-bearing on the artifact routing actually consumes, and that the three
committed lab overlays agree with the table rather than only the table agreeing
with itself.

Why both are needed, concretely. ``parameter_count`` lives in four places: the
table, and the dev, judge and lakshman overlays. Correcting the table alone
leaves three overlays stating the retired figure, and correcting the overlays
alone leaves the table stating it. Either half-correction is exactly the
"half-corrected binding" class OMN-16419 named when it fixed a served id and
left the context window behind.

The negative control is the assertion that makes this test worth running: a
copy of the real dev overlay, byte-identical except for the retired parameter
count, must be REFUSED. Without it, a renderer that silently dropped the field
would pass the positive case and prove nothing.
"""

from __future__ import annotations

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

#: The lab lanes, each of which binds the .201 rungs. ``onex-dev`` is excluded
#: deliberately: it is a ``cloud`` locale overlay that declares no local
#: backends at all, so it has no parameter count to agree about.
_LAB_OVERLAYS = ("dev.bifrost.yaml", "judge.bifrost.yaml", "lakshman.bifrost.yaml")

_LOCAL_201_BACKENDS = ("local-coder", "local-heavy-reasoning")


#: The env hint each backend carries in the base contract. The renderer strips
#: these — the overlay owns the real endpoint — but the base contract is
#: rejected without them.
_ENDPOINT_URL_ENV = {
    "local-coder": "LLM_CODER_URL",
    "local-heavy-reasoning": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
    "local-ds-v4-flash": "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL",
}


def _write_base_contract(path: Path) -> None:
    """A minimal base contract carrying the backend ids the overlays bind.

    ``model_name`` is read from the authorized table rather than retyped: the
    renderer refuses a base contract whose model name disagrees with the
    overlay's served id, so a hardcoded literal here would turn every future
    re-pin of that id into a failure in this file instead of a real finding.
    """
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": backend_id,
                        "model_name": _AUTHORIZED_BINDINGS[backend_id].served_model_id,
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


@pytest.mark.parametrize("overlay_name", _LAB_OVERLAYS)
def test_committed_lab_overlay_renders_and_carries_the_corrected_binding(
    overlay_name: str, tmp_path: Path
) -> None:
    """Each committed lab overlay renders, and renders the authorized values."""
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)

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
        authorized = _AUTHORIZED_BINDINGS[backend_id]
        assert by_id[backend_id]["model_name"] == authorized.served_model_id
        assert (
            by_id[backend_id]["endpoint_url"]
            == f"http://{authorized.host}:{authorized.port}/v1/chat/completions"
        )


def test_an_overlay_restating_the_retired_parameter_count_is_refused(
    tmp_path: Path,
) -> None:
    """NEGATIVE CONTROL: the corrected value is enforced, not merely written.

    This is the falsifier for the whole change. The fixture is the real dev
    overlay with one field reverted to the 35B-A3B value the contract carried
    until 2026-09-17, so a pass here would mean the retired figure is still
    acceptable and the correction is decoration.
    """
    overlay = yaml.safe_load(
        (_OVERLAY_DIR / "dev.bifrost.yaml").read_text(encoding="utf-8")
    )
    reverted = 0
    for backend in overlay["backends"]:
        if backend["backend_id"] in _LOCAL_201_BACKENDS:
            backend["parameter_count"] = "35B-A3B"
            reverted += 1
    assert reverted == len(_LOCAL_201_BACKENDS), (
        "the dev overlay no longer declares both .201 rungs, so this control "
        "is not exercising what it claims to"
    )

    poisoned = tmp_path / "poisoned.bifrost.yaml"
    poisoned.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)

    with pytest.raises(ProtocolConfigurationError):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=poisoned,
            target_path=target,
            environ={},
        )
    assert not target.exists(), (
        "a refused overlay must leave no rendered contract behind — a partial "
        "write here would be read by the next process to start"
    )
