# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17985 -- the health monitor's ownership filter reads one validated name.

OMN-17985 consolidated every ``RUNTIME_PROFILE`` read that decides CONTRACT
OWNERSHIP onto :func:`resolve_runtime_profile_name`, so an unregistered role
name is refused instead of quietly emptying the manifest. The consolidation
claimed to be complete. It was not: two raw reads in this module still fed
``filter_manifest_for_runtime_profile`` directly --
``os.getenv("RUNTIME_PROFILE", "main")`` in ``_discover_contracts`` and
``os.environ.get("RUNTIME_PROFILE", "main")`` in
``_filter_manifest_for_runtime_profile``.

Why it matters even though both spell the same ``"main"`` fallback: the health
monitor is the surface that reports whether the fleet's subscriptions exist. An
unregistered profile name reaching it is filtered against a list no contract
declares, so every contract is skipped and the monitor concludes the runtime
owns nothing -- and reports that as a finding about the FLEET rather than as
the misconfiguration it is. The refusal has to happen at the read.

The test that matters is the behavioural one; the source assertion exists only
so a future raw read cannot silently re-open the same gap.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.services import service_runtime_health_monitor as monitor

pytestmark = pytest.mark.unit

_ENV_VAR = "RUNTIME_PROFILE"
_SOURCE = Path(inspect.getsourcefile(monitor) or "")


def test_discover_contracts_refuses_an_unregistered_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact shape that empties a manifest while the process stays Ready."""
    monkeypatch.setenv(_ENV_VAR, "projection-writer-delegation-typo")
    with pytest.raises(ProtocolConfigurationError):
        monitor._discover_contracts()


def test_filter_manifest_refuses_an_unregistered_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV_VAR, "not-a-registered-profile")

    from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
        ModelAutoWiringManifest,
    )

    manifest = ModelAutoWiringManifest()
    with pytest.raises(ProtocolConfigurationError):
        monitor._filter_manifest_for_runtime_profile(manifest)


def test_a_registered_profile_still_flows_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSITIVE CONTROL: the refusal rejects unknown names, not every name.

    A test that only ever asserts a raise cannot tell a working gate from one
    that refuses everything.
    """
    monkeypatch.setenv(_ENV_VAR, "effects")

    from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
        ModelAutoWiringManifest,
    )

    manifest = ModelAutoWiringManifest()
    assert monitor._filter_manifest_for_runtime_profile(manifest) is not None


def test_unset_profile_still_resolves_to_the_ownership_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consolidating adds validation; it must not move the ``main`` default."""
    monkeypatch.delenv(_ENV_VAR, raising=False)

    from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
        ModelAutoWiringManifest,
    )

    manifest = ModelAutoWiringManifest()
    assert monitor._filter_manifest_for_runtime_profile(manifest) is not None


def test_no_raw_runtime_profile_read_remains_in_this_module() -> None:
    """A future raw read must not silently re-open the gap the two above close."""
    body = _SOURCE.read_text(encoding="utf-8")
    for raw in (
        'os.getenv("RUNTIME_PROFILE"',
        "os.getenv('RUNTIME_PROFILE'",
        'os.environ.get("RUNTIME_PROFILE"',
        "os.environ.get('RUNTIME_PROFILE'",
        'os.environ["RUNTIME_PROFILE"]',
    ):
        assert raw not in body, f"raw RUNTIME_PROFILE read remains: {raw}"
    assert "resolve_runtime_profile_name" in body


def test_the_stale_service_kernel_comment_is_gone() -> None:
    """OMN-17985 changed the fallback to a hard raise but left its own comment.

    ``service_kernel`` still documented ``unknown values fall back to
    "default" ... with a structured warning`` immediately above the
    ``load_runtime_profile()`` call whose behaviour the same change replaced
    with a refusal. A comment describing a removed defect as current behaviour
    is what the next reader acts on.
    """
    kernel = _SOURCE.parent.parent / "runtime" / "service_kernel.py"
    body = kernel.read_text(encoding="utf-8")
    assert 'unknown values fall back to "default"' not in body
