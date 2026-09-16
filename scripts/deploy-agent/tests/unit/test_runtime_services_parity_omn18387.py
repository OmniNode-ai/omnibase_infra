# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18387 -- the deploy agent's runtime up-target/verification set must
carry everything ``scripts/deploy-runtime.sh``'s ``RUNTIME_SERVICES`` array
restarts via ``up -d --no-deps``.

``deploy_agent.events.SCOPE_SERVICES[Scope.RUNTIME]`` is the single source
both the up-target list (``_requested_services_for_up`` in ``executor.py``)
and the post-up container verification (``verify_containers_up``) resolve
from -- see the comment on that call site. So the up-target and
verification lists cannot drift from EACH OTHER by construction; the gap
this ticket found was that shared source itself never carried
``projection-api``, while the bash array always has.

Measured 2026-09-15: a re-publish through
``scripts/deploy-agent/deploy-agent-trigger.sh`` (correlation ``79427973``)
ran the runtime phase to SUCCESS and built a fresh ``projection-api`` image
(distinct sha ``647f0663418d`` from the running container's), but the
container was never recreated -- silent because nothing in the up-target or
verification set named it.

This test parses the bash array (the same ``_bash_array`` pattern
``test_dev_lane_only_scope_omn18108.py`` uses for
``DEV_LANE_ONLY_RUNTIME_SERVICES``) and asserts every entry in it is also in
the Python dict, so editing one without the other is a red test. The
reverse direction is not asserted with a bare set-equality: three Python
entries reach the lane through a DIFFERENT bash mechanism than the
``up -d --no-deps`` array (a one-shot preflight wait, a dev-lane-only
broker-credential addendum, and an unbuilt sidecar with no source to
rebuild) rather than being an actual omission on the bash side, and are
named explicitly below so a FOURTH unexplained Python-only entry still
fails closed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from deploy_agent.events import SCOPE_SERVICES, Scope

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"

# Documented, one Python-only entry per reason -- never a catch-all escape
# hatch. A new entry landing on either side without a matching change here
# is exactly the drift this test exists to catch.
#
#   context-audit-consumer -- reached via deploy-runtime.sh's
#       DEV_LANE_EXTRA_BROKER_CLIENTS (OMN-18012), a dev-lane addendum
#       array, not the lane-agnostic RUNTIME_SERVICES array this test reads.
#   intelligence-migration -- reached via deploy-runtime.sh's
#       RUNTIME_MIGRATION_SERVICES (OMN-13220): a one-shot preflight
#       `docker wait` target applied before the --no-deps runtime restart,
#       not a service the restart itself names.
#   autoheal -- a sidecar with no build context of its own on this compose
#       lane (no `build:` stanza deploy-runtime.sh's build phase could
#       target); named only in a deploy-runtime.sh comment, never in any of
#       its restart arrays.
PYTHON_ONLY_RUNTIME_EXTRAS = frozenset(
    {"context-audit-consumer", "intelligence-migration", "autoheal"}
)


def _bash_array(name: str) -> list[str]:
    """Return the entries of a ``readonly NAME=( ... )`` array in the script.

    Mirrors ``test_dev_lane_only_scope_omn18108.py``'s helper of the same
    name and behavior: parsed rather than duplicated, so this test reads the
    live script instead of a second hand-copied list.
    """
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"^readonly\s+{re.escape(name)}=\((?P<body>.*?)^\)", text, re.M | re.S
    )
    if match is None:  # pragma: no cover - defended by its own test below
        raise AssertionError(f"{name} array not found in {DEPLOY_SCRIPT}")
    entries: list[str] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.extend(line.split())
    return entries


class TestTheArrayIsFoundAtAll:
    """A positive control: an empty parse must never read as agreement."""

    def test_runtime_services_array_is_non_empty(self) -> None:
        assert len(_bash_array("RUNTIME_SERVICES")) >= 8


class TestRuntimeServicesCannotDriftFromTheBashArray:
    def test_every_bash_runtime_service_is_in_the_python_scope(self) -> None:
        from_bash = set(_bash_array("RUNTIME_SERVICES"))
        from_python = set(SCOPE_SERVICES[Scope.RUNTIME])
        missing = from_bash - from_python
        assert not missing, (
            "scripts/deploy-runtime.sh RUNTIME_SERVICES names service(s) the "
            "deploy agent's up-target/container-verification set "
            "(deploy_agent.events.SCOPE_SERVICES[Scope.RUNTIME]) never "
            f"reaches, so an agent-path deploy can leave them stale: {sorted(missing)}"
        )

    def test_every_python_only_entry_is_a_documented_extra(self) -> None:
        from_bash = set(_bash_array("RUNTIME_SERVICES"))
        from_python = set(SCOPE_SERVICES[Scope.RUNTIME])
        python_only = from_python - from_bash
        undocumented = python_only - PYTHON_ONLY_RUNTIME_EXTRAS
        assert not undocumented, (
            "deploy_agent.events.SCOPE_SERVICES[Scope.RUNTIME] carries "
            "service(s) absent from scripts/deploy-runtime.sh RUNTIME_SERVICES "
            "with no documented reason in PYTHON_ONLY_RUNTIME_EXTRAS above: "
            f"{sorted(undocumented)}"
        )

    def test_no_duplicate_entries_on_either_side(self) -> None:
        from_bash = _bash_array("RUNTIME_SERVICES")
        assert len(from_bash) == len(set(from_bash))
        runtime_list = SCOPE_SERVICES[Scope.RUNTIME]
        assert len(runtime_list) == len(set(runtime_list))
