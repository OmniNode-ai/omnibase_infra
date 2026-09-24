# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel owns secret resolution on behalf of domain plugins (OMN-19129).

`PluginLlm` used to have no way to authenticate its health probes: the LLM
endpoint health service resolved each backend's credential only to decide
whether the endpoint was worth probing, then threw the value away and probed
anonymously. An auth-gated vendor surface rejects an unauthenticated request at
the auth layer before routing, so it answers 401 for every path including paths
that do not exist, and the service classified that as a rejected credential.

The fix cannot be "let the plugin read the environment": this repo forbids new
`os.environ` reads outside a small approved set, and the plugin's own contract
is that the kernel supplies resolved configuration. So the kernel builds one
resolver and hands it down on `ModelDomainPluginConfig`.

This suite pins the resolver's precedence and its fail-soft behaviour. The
callable shape matters as much as the values: a resolved credential is returned
to the caller and never stored on the config, so it cannot reach a repr, a log
line or an event payload.

Related Tickets:
    - OMN-19129: the prober never sent the credential it classified against
    - OMN-19127: the GLM rung suppressed on the dev lane
    - OMN-12634: overlay config as the plugin's source of resolved settings
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.service_kernel import make_plugin_secret_resolver

_VAR = "LLM_GLM_API_KEY"
# Assembled at runtime so no credential-shaped literal exists in the source.
_FROM_OVERLAY = "-".join(("resolved", "by", "overlay"))
_FROM_ENV = "-".join(("resolved", "by", "environment"))


@pytest.mark.unit
def test_overlay_answers_before_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where an overlay is loaded it is the authoritative source."""
    monkeypatch.setenv(_VAR, _FROM_ENV)
    resolve = make_plugin_secret_resolver({_VAR: _FROM_OVERLAY})
    assert resolve(_VAR) == _FROM_OVERLAY


@pytest.mark.unit
def test_environment_answers_in_legacy_boot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No overlay means legacy env-var boot, which is how the lab lanes run.

    This is not a convenience path. Measured on the dev lane 2026-09-21,
    `~/.omnibase/overlay.yaml` is absent in the runtime container and the
    credential is present in the process environment. A resolver that refused
    to read it would leave the rung unprobeable and the fix inert.
    """
    monkeypatch.setenv(_VAR, _FROM_ENV)
    resolve = make_plugin_secret_resolver(None)
    assert resolve(_VAR) == _FROM_ENV


@pytest.mark.unit
def test_overlay_miss_falls_through_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loaded overlay that does not carry this key is not a veto."""
    monkeypatch.setenv(_VAR, _FROM_ENV)
    resolve = make_plugin_secret_resolver({"SOMETHING_ELSE": "x"})
    assert resolve(_VAR) == _FROM_ENV


@pytest.mark.unit
def test_empty_overlay_value_is_treated_as_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty string is an absent credential, not a credential."""
    monkeypatch.setenv(_VAR, _FROM_ENV)
    resolve = make_plugin_secret_resolver({_VAR: ""})
    assert resolve(_VAR) == _FROM_ENV


@pytest.mark.unit
def test_unresolvable_name_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither source has it, so the caller must decline to probe."""
    monkeypatch.delenv(_VAR, raising=False)
    resolve = make_plugin_secret_resolver({})
    assert resolve(_VAR) is None
