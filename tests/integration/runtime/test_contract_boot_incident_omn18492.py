# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The 2026-09-16 duplicate-ingress-alias contract fails the boot check, then passes (OMN-18492).

On 2026-09-16 omnimarket#2598 gave the second ``handler_routing`` entry of
``node_redeploy_deploy_effect`` its honest input model while both entries still
shared one operation. The two local-ingress routes became inequivalent, the
alias registry refused the second one, and ``omninode-runtime`` crash-looped
(restart count 101) until omnimarket#2603 gave the entry its own operation.
Nothing at pull-request time asked whether the contract set would boot.

These tests replay the incident through the one function the runtime calls at
boot, ``discover_runtime_local_ingress_routes``, over the two verbatim contract
blobs: the pre-fix bytes (omnimarket ``ea48b81dd``, the parent of the fix) must
raise, and the post-fix bytes (omnimarket ``c1dc7e689``) must discover. A check
that only passes proves nothing, so the first is the positive control for the
second and for the shipped-corpus test in
``test_local_ingress_self_collision_omn18550.py``; the ``candidate-boot-gate``
job runs both and reports the outcome as the ``contract_boot_invariants`` check
of the ``onex-lab`` lab-pass receipt.

The fixtures are captured, not retyped: the sha256 pins below fail the test if
the bytes drift, and the locator is a git object anyone can re-fetch.

Related Tickets:
    - OMN-18492: a contract boot test catches a boot-fatal change before merge.
    - OMN-17888: the omnimarket contract that carried the live duplicate.
    - OMN-18550: the boot that cannot finish names the contract and the probe.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_infra.runtime.runtime_local_ingress import (
    discover_runtime_local_ingress_routes,
)

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "omn18492"

#: git-object:OmniNode-ai/omnimarket@ea48b81ddd9a5bc69641fdff4b378009e0ff0946:
#: src/omnimarket/nodes/node_redeploy_deploy_effect/contract.yaml
PRE_FIX = (
    FIXTURES / "node_redeploy_deploy_effect.contract.pre-fix-ea48b81dd.yaml.captured"
)
PRE_FIX_SHA256 = "a66f6d111062c41faf490f77668e701910ca6a8909f97b8ac326e9874d9a89a8"

#: git-object:OmniNode-ai/omnimarket@c1dc7e6895b2b0b1a89f12cf347d0bbe258a2daf:
#: src/omnimarket/nodes/node_redeploy_deploy_effect/contract.yaml
POST_FIX = (
    FIXTURES / "node_redeploy_deploy_effect.contract.post-fix-c1dc7e689.yaml.captured"
)
POST_FIX_SHA256 = "1d550c024373e9b0ac5e5da88ffd19a514f8ee0f4c77df71de8ee437e483c9a0"

#: The alias the live runtime raised on, byte for byte from the incident.
INCIDENT_ALIAS = (
    "omnimarket.node_redeploy_deploy_effect.redeploy.deploy.publish_monitor"
)

#: The alias the post-fix contract gives the completion entry (omnimarket#2603).
FIXED_ALIAS = (
    "omnimarket.node_redeploy_deploy_effect.redeploy.deploy.rebuild_completed_observe"
)


def _install_as_omnimarket(
    contract: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lay the captured contract out as the installed ``omnimarket`` package."""
    package_root = tmp_path / "omnimarket"
    node_dir = package_root / "nodes" / "node_redeploy_deploy_effect"
    node_dir.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(contract, node_dir / "contract.yaml")

    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_local_ingress.importlib.import_module",
        lambda _name: SimpleNamespace(__file__=str(package_root / "__init__.py")),
    )


@pytest.mark.parametrize(
    ("fixture", "expected_sha256"),
    [(PRE_FIX, PRE_FIX_SHA256), (POST_FIX, POST_FIX_SHA256)],
    ids=["pre-fix", "post-fix"],
)
def test_the_captured_contracts_are_the_bytes_that_were_captured(
    fixture: Path, expected_sha256: str
) -> None:
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == expected_sha256


def test_the_pre_fix_contract_fails_the_boot_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Red first: the contract that took the dev lane down must not discover."""
    _install_as_omnimarket(PRE_FIX, tmp_path, monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        discover_runtime_local_ingress_routes(("omnimarket",))

    message = str(excinfo.value)
    assert INCIDENT_ALIAS in message
    assert "Duplicate local ingress route alias" in message


def test_the_post_fix_contract_passes_the_boot_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Green after: distinct operations discover, and both are still routed."""
    _install_as_omnimarket(POST_FIX, tmp_path, monkeypatch)

    routes = discover_runtime_local_ingress_routes(("omnimarket",))

    # Non-vacuous: the pass is not an empty discovery. The fix gave the second
    # entry its own operation, so each entry owns one qualified alias and the
    # two route to different input models.
    publish = routes[INCIDENT_ALIAS]
    observe = routes[FIXED_ALIAS]
    assert publish.input_model_name == "ModelDeployPublishCommand"
    assert observe.input_model_name == "ModelDeployRebuildCompleted"
