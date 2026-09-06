# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An unregistered ``RUNTIME_PROFILE`` must refuse to boot (OMN-17985).

Before this change the same env var was read twice, by two code paths that
disagreed about what it resolved to:

* ``load_runtime_profile`` fell back to the ``"default"`` profile with a
  ``logger.warning``. The resolved ``profile.name`` became ``"default"``, so
  the role identity the value carries was silently discarded.
* The auto-wiring ownership filter never consulted that resolved profile at
  all -- it read ``os.getenv("RUNTIME_PROFILE", "main")`` raw and handed the
  unvalidated string to ``filter_manifest_for_runtime_profile``.

An unregistered string matches no contract's declared ``runtime_profiles`` list
AND is ``!= "main"``, so every contract is skipped: the manifest empties, zero
subscriptions are wired, and the process still passes readiness. The only
signal is one warning emitted from a different function. That is the OMN-12950
orphan mechanism, and a fallback is the wrong shape for it -- a profile name is
role identity, and a role the runtime cannot resolve is not a milder version of
a role, it is an unknown deployment.

So: refuse. A pod carrying a ``RUNTIME_PROFILE`` that ``_PROFILES`` does not
know now raises during boot and never becomes Ready, and both readers resolve
the value through one validated function.
"""

from __future__ import annotations

import pytest

from omnibase_core.constants.constants_runtime_profiles import (
    REGISTERED_RUNTIME_PROFILES,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.runtime_profile import (
    _PROFILES,
    load_runtime_profile,
    resolve_runtime_profile_name,
)

pytestmark = pytest.mark.unit

_ENV_VAR = "RUNTIME_PROFILE"


class TestLoadRuntimeProfileFailsClosed:
    def test_unknown_explicit_name_raises(self) -> None:
        with pytest.raises(ProtocolConfigurationError) as excinfo:
            load_runtime_profile("unknown-lane")
        message = str(excinfo.value)
        assert "unknown-lane" in message

    def test_refusal_names_the_known_profiles(self) -> None:
        """The operator must be able to fix the value from the error alone."""
        with pytest.raises(ProtocolConfigurationError) as excinfo:
            load_runtime_profile("projection-writer-typo")
        message = str(excinfo.value)
        for known in ("main", "effects"):
            assert known in message

    def test_unknown_env_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV_VAR, "all")
        with pytest.raises(ProtocolConfigurationError):
            load_runtime_profile()

    def test_blank_env_value_resolves_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blank is 'unset', not 'unknown' -- it must not become a refusal."""
        monkeypatch.setenv(_ENV_VAR, "")
        assert load_runtime_profile().name == "default"

    def test_unset_env_resolves_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert load_runtime_profile().name == "default"

    @pytest.mark.parametrize("profile_name", sorted(_PROFILES))
    def test_every_registered_profile_still_loads(self, profile_name: str) -> None:
        assert load_runtime_profile(profile_name).name == profile_name

    def test_surrounding_whitespace_and_case_are_normalized(self) -> None:
        assert load_runtime_profile("  MAIN  ").name == "main"


class TestResolveRuntimeProfileName:
    """One validated resolution, used by both readers of the variable."""

    def test_unset_resolves_to_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset keeps the auto-wiring ownership default it always had."""
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert resolve_runtime_profile_name() == "main"

    def test_blank_resolves_to_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV_VAR, "   ")
        assert resolve_runtime_profile_name() == "main"

    def test_registered_value_is_returned_normalized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV_VAR, " Effects ")
        assert resolve_runtime_profile_name() == "effects"

    def test_unknown_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The exact shape that emptied the manifest and stayed Ready."""
        monkeypatch.setenv(_ENV_VAR, "projection-writer-delegation-typo")
        with pytest.raises(ProtocolConfigurationError):
            resolve_runtime_profile_name()

    @pytest.mark.parametrize("profile_name", sorted(_PROFILES))
    def test_agrees_with_load_runtime_profile_on_every_known_name(
        self, profile_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two readers must not disagree about one variable again."""
        monkeypatch.setenv(_ENV_VAR, profile_name)
        assert resolve_runtime_profile_name() == load_runtime_profile().name


def test_core_registry_names_are_all_bootable() -> None:
    """Positive control: the refusal above rejects unknown names, not all names.

    A test that only ever asserts a raise cannot tell a working gate from one
    that refuses everything.
    """
    for profile_name in sorted(REGISTERED_RUNTIME_PROFILES):
        assert load_runtime_profile(profile_name).name == profile_name
