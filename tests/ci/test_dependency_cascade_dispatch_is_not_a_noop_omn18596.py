# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time gate: the cascade's manual path works, and cannot pass silently.

Why this file exists
--------------------
Run 35260585795 dispatched ``dependency-cascade.yml`` against the released
``v0.38.30`` while all three fan-out targets were genuinely behind it
(omniintelligence on 0.38.21, omnimemory on 0.38.24, omniclaude on 0.38.27).
Every leg reported ``success``. Not one of them opened, updated or even
examined a pull request, and nothing about the result said so.

The cause is a platform constraint, and it was checked rather than assumed.
``dependency-cascade.yml`` declares ``on: workflow_call: secrets:``, and a
workflow carrying that declaration has its ``secrets`` context restricted to
the DECLARED names under every trigger. An org-level name therefore resolves
EMPTY inside it even on a manual run, ``has_token`` went false, every later
step's ``if:`` guard was false, and the job exited 0 having done nothing.

It is not an entitlement problem: the org secret lists ``omnibase_infra``
among its selected repositories. The control that settles it is
``release-train-nightly.yml``, which reads the same org-level names on the
same trigger in this same repository and mints successfully -- it simply
declares no ``workflow_call``. Same value, same repo, same trigger, opposite
result; the difference is the declaration.

A first attempt added an org-level fallback inside the reusable workflow.
Re-dispatch run 35267246993 disproved it: the legs failed loudly (the
fail-closed half working) but the mint still errored with ``client-id`` unset.
That fallback was dead code in a credential expression, which is worse than no
fallback, because the next reader believes the manual path is covered.

So the manual trigger now lives in ``dependency-cascade-dispatch.yml``, which
declares no ``workflow_call`` and passes the values through explicitly -- the
shape ``release.yml`` already uses for the automatic path.

Three things are asserted, and the third is the one that would have caught the
original defect:

* the reusable workflow does NOT carry a manual trigger it cannot serve;
* the wrapper does, declares no ``workflow_call``, and passes both values;
* a leg with no credentials FAILS. A cascade that cannot authenticate has not
  skipped a bump, it has failed to perform one, and the two must not look the
  same from outside.

Ticket: OMN-18596
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"
_REUSABLE = _WORKFLOWS / "dependency-cascade.yml"
_WRAPPER = _WORKFLOWS / "dependency-cascade-dispatch.yml"


def _parsed(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(path: Path) -> dict[str, Any]:
    doc = _parsed(path)
    # `on` parses as the boolean True in YAML 1.1, which is how every YAML 1.1
    # loader reads a GitHub workflow file.
    return doc.get("on", doc.get(True, {})) or {}


def _token_check_step() -> dict[str, Any]:
    for job in _parsed(_REUSABLE)["jobs"].values():
        for step in job.get("steps", []) or []:
            if step.get("id") == "token_check":
                return step
    raise AssertionError("no step with id token_check in the reusable cascade")


class TestTheReusableWorkflowDoesNotCarryATriggerItCannotServe:
    def test_the_reusable_cascade_declares_no_manual_trigger(self) -> None:
        """A manual trigger there can only ever fail, or worse, no-op.

        This is the assertion that keeps the constraint from being re-learned:
        re-adding `workflow_dispatch` to a workflow that declares
        `workflow_call: secrets:` recreates run 35260585795 exactly.
        """
        assert "workflow_dispatch" not in _triggers(_REUSABLE), (
            "dependency-cascade.yml declares workflow_call secrets, so its "
            "secrets context is restricted to the declared names under every "
            "trigger and a manual run cannot resolve them. The manual path "
            "belongs in dependency-cascade-dispatch.yml"
        )
        assert "workflow_call" in _triggers(_REUSABLE)

    def test_the_reusable_cascade_reads_one_source_for_each_value(self) -> None:
        """No dead fallback in a credential expression.

        An org-level name is always empty here, so reading one would look like
        coverage of the manual path while providing none.
        """
        env = _token_check_step()["env"]
        for var, declared, org in (
            ("APP_ID", "secrets.onexbot-occ-app-id", "ONEXBOT_OCC_APP_ID"),
            (
                "APP_PRIVATE_KEY",
                "secrets.onexbot-occ-private-key",
                "ONEXBOT_OCC_PRIVATE_KEY",
            ),
        ):
            expr = env[var]
            assert declared in expr
            assert org not in expr, (
                f"{var} reads an org-level name inside a workflow whose secrets "
                "context cannot contain one; run 35267246993 proved that "
                "resolves empty. Pass it from a caller instead"
            )


class TestTheWrapperOwnsTheManualPath:
    def test_the_wrapper_exists_and_is_dispatch_only(self) -> None:
        assert _WRAPPER.exists(), (
            "the manual cascade path has no wrapper; without it there is no "
            "way to re-run a failed cascade or catch a downstream repo up"
        )
        triggers = _triggers(_WRAPPER)
        assert "workflow_dispatch" in triggers
        assert "workflow_call" not in triggers, (
            "declaring workflow_call here would restrict this workflow's own "
            "secrets context and reintroduce the very defect it exists to fix"
        )

    def test_the_wrapper_passes_both_values_to_the_reusable_workflow(
        self,
    ) -> None:
        job = next(iter(_parsed(_WRAPPER)["jobs"].values()))
        assert job["uses"].endswith("dependency-cascade.yml"), (
            "the wrapper must call the reusable cascade rather than "
            "reimplementing it; two copies of the fan-out would drift"
        )
        secrets = job["secrets"]
        assert "ONEXBOT_OCC_APP_ID" in secrets["onexbot-occ-app-id"]
        assert "ONEXBOT_OCC_PRIVATE_KEY" in secrets["onexbot-occ-private-key"]

    def test_the_wrapper_offers_only_the_packages_the_cascade_accepts(
        self,
    ) -> None:
        """The reusable workflow refuses an unknown package. Offer no others."""
        package = _triggers(_WRAPPER)["workflow_dispatch"]["inputs"]["package"]
        assert set(package["options"]) == {
            "omnibase_core",
            "omnibase_spi",
            "omnibase_infra",
        }

    def test_neither_workflow_substitutes_the_default_token(self) -> None:
        """OMN-18273: a push by the default token suppresses downstream CI."""
        for path in (_REUSABLE, _WRAPPER):
            body = path.read_text(encoding="utf-8")
            assert re.search(r"\|\|\s*secrets\.GITHUB_TOKEN", body) is None, (
                f"{path.name} substitutes the default workflow token"
            )
            assert "secrets.CROSS_REPO_PAT" not in body, (
                f"{path.name} reintroduces the long-lived personal token "
                "OMN-16373 retired"
            )


class TestACredentiallessLegFailsRatherThanReportingGreen:
    def test_the_token_check_exits_nonzero_when_credentials_are_absent(
        self,
    ) -> None:
        run = _token_check_step()["run"]
        assert "exit 1" in run, (
            "a leg that cannot authenticate must FAIL. Reporting success makes "
            "a cascade that performed no bump indistinguishable from one that "
            "had nothing to bump, which is how run 35260585795 read as four "
            "green jobs having done nothing at all"
        )
        assert "::error::" in run, (
            "the failure must be annotated as an error, not a warning; a "
            "warning on a green job is what nobody read"
        )

    def test_the_run_no_longer_writes_a_has_token_false_path(self) -> None:
        run = _token_check_step()["run"]
        assert "has_token=false" not in run, (
            "writing has_token=false keeps the silent-skip path alive; the "
            "step should fail instead"
        )

    def test_every_later_step_still_guards_on_the_token_check(self) -> None:
        """Kept as a regression assertion.

        If the guards were dropped WITHOUT the check failing, a
        credential-less leg would run `gh` unauthenticated.
        """
        steps = None
        for job in _parsed(_REUSABLE)["jobs"].values():
            if any(s.get("id") == "token_check" for s in job.get("steps", []) or []):
                steps = job["steps"]
        assert steps is not None
        guarded = [
            s for s in steps if "token_check.outputs.has_token" in str(s.get("if", ""))
        ]
        assert guarded, (
            "no step guards on the token check; combined with a failing check "
            "this is survivable, but losing both leaves an unauthenticated leg"
        )
