# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18570: drive every COMMITTED lab overlay through the real renderer.

The unit test ``tests/unit/runtime/test_bifrost_parameter_count_matches_served_id.py``
checks each committed binding's ``parameter_count`` against its served model id,
and ``test_bifrost_served_model_probe_fixture.py`` checks the served id against a
recorded probe. That is the referent half. This is the reach half: it proves the
committed lab overlays render into the artifact routing actually consumes, and
that the renderer still REFUSES a binding whose served id disagrees with the base
contract, which is the load-bearing check that survives OMN-17099.

OMN-17099 removed the hardcoded authorization table this module used to read its
expected values from. The expected values are now the lab overlays' own
declarations; what the renderer enforces against them is the base contract.

The negative control is the assertion that makes this test worth running: a copy
of the real dev overlay, identical except for a served id the base contract does
not declare, must be REFUSED and leave no artifact. Without it, a renderer that
silently dropped the check would pass the positive case and prove nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
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


def _dev_overlay() -> dict:
    loaded = yaml.safe_load((_OVERLAY_DIR / "dev.bifrost.yaml").read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _served_ids() -> dict[str, str]:
    """The served id the committed dev overlay binds, per backend."""
    return {
        backend["backend_id"]: backend["served_model_id"]
        for backend in _dev_overlay()["backends"]
    }


def _write_base_contract(path: Path) -> None:
    """A minimal base contract carrying the backend ids the overlays bind.

    ``model_name`` is read from the committed dev overlay rather than retyped:
    the renderer refuses a base contract whose model name disagrees with the
    overlay's served id, so a hardcoded literal here would turn every future
    re-pin of that id into a failure in this file instead of a real finding.
    """
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": backend_id,
                        "model_name": _served_ids()[backend_id],
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
    """Each committed lab overlay renders, and renders its own declared values."""
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
    declared = {
        backend["backend_id"]: backend
        for backend in yaml.safe_load(
            (_OVERLAY_DIR / overlay_name).read_text(encoding="utf-8")
        )["backends"]
    }
    for backend_id in _LOCAL_201_BACKENDS:
        assert (
            by_id[backend_id]["model_name"] == declared[backend_id]["served_model_id"]
        )
        assert by_id[backend_id]["endpoint_url"] == declared[backend_id]["endpoint_url"]


def test_an_overlay_binding_a_served_id_the_base_does_not_declare_is_refused(
    tmp_path: Path,
) -> None:
    """NEGATIVE CONTROL: the served id is enforced, not merely written.

    The fixture is the real dev overlay with the .201 rungs' served id set to
    something the base contract does not declare, so a pass here would mean the
    renderer no longer checks a binding against its base contract. The poison
    is DERIVED from the committed value (OMN-18626: a literal poison went stale
    the day the endpoint was re-pinned onto it, and the control silently stopped
    controlling for anything).
    """
    overlay = _dev_overlay()
    poisoned_count = 0
    for backend in overlay["backends"]:
        if backend["backend_id"] in _LOCAL_201_BACKENDS:
            poison = f"{backend['served_model_id']}-not-the-served-id"
            assert poison != backend["served_model_id"]
            backend["served_model_id"] = poison
            poisoned_count += 1
    assert poisoned_count == len(_LOCAL_201_BACKENDS), (
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
