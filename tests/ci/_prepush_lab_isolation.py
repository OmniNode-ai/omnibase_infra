# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Keep hook-subprocess tests from spending a real lab host's cores (OMN-16991).

Several tests in this directory run the REAL pre-push hook with the heavy
escalation forced (``PREPUSH_FULL_SUITE=1``) and every designated host
de-designated, to prove the refusal. Until OMN-16991 that was harmless: the
hook's host scan was truncated after its first ssh probe, so the picker only
ever saw ``h200`` and never had a remote host to dispatch to.

Fixing the scan removed that accidental containment. Observed live on
2026-08-30, minutes after the fix: `pytest tests/ci/` shipped a real git bundle
to ``omnibook``, took that host's exclusive slot, and started the full
``tests/unit/`` suite there -- ORIGIN on the remote wrapper named this very test
process. That is the OMN-16425/OMN-16489 F-01 recursion in its distributed form,
reached from a unit test instead of a push, and it burns a lab host for an hour
per test run.

The isolation below uses the picker's OWN deterministic override surface rather
than a new knob. ``PREPUSH_SLOT_OVERRIDE_MAP`` is consulted before any network
call, and a label absent from the map resolves to "slot unknown", which the
picker treats as unfit and skips -- the same fail-closed posture it applies to
an unreachable host. A map naming no real label therefore makes EVERY row unfit
with zero ssh, and stays correct when a row is added.

It can only make the gate stricter. With no host placeable the lab leg produces
no evidence and the hook falls through to its pre-existing precedence
(GitHub-hosted verify -> grant -> die), which is exactly what these tests assert.

OMN-20063 closes the third network surface: the GitHub-hosted remote-verify
leg (OMN-16688) reads the real Actions API for HEAD's sha. On a queue-landed
dev push, a green merge_group full-suite run on that exact sha already exists,
so the hook passes instead of refusing. An empty gh config directory and empty
authentication tokens keep that leg offline. This can only remove evidence,
preserving the same strict direction as the lab isolation.
"""

from __future__ import annotations

import tempfile

_GH_CONFIG_DIR = tempfile.mkdtemp(prefix="prepush-gh-isolation-")

#: Deliberately names no real row label. See the module docstring.
#:
#: ``PREPUSH_REACH_OVERRIDE_MAP`` (OMN-17280) closes the second network surface
#: this module exists to close. The same-host route probes lab reachability
#: with a real ``ssh ... true`` before it may fire, and a hook-subprocess test
#: that reached a designated row would otherwise open real connections from
#: pytest. ``default=up`` reports EVERY row -- including rows added later --
#: as reachable, which makes the same-host route DECLINE. That is the strict
#: direction: the leg produces no evidence and the hook falls through to its
#: pre-existing precedence, which is exactly what these tests assert.
LAB_ISOLATION_ENV = {
    "PREPUSH_SLOT_OVERRIDE_MAP": "no-such-host=unknown",
    "PREPUSH_REACH_OVERRIDE_MAP": "default=up",
}


def network_free_lab_env() -> dict[str, str]:
    """Env fragment that makes the lab and GitHub verification legs network-free."""
    return {
        **LAB_ISOLATION_ENV,
        "GH_CONFIG_DIR": _GH_CONFIG_DIR,
        "GH_TOKEN": "",
        "GITHUB_TOKEN": "",
        "GH_ENTERPRISE_TOKEN": "",
        "GITHUB_ENTERPRISE_TOKEN": "",
    }
