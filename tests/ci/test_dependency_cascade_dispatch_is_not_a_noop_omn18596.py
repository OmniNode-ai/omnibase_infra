# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time gate: the cascade's dispatch path works, and cannot pass silently.

Why this file exists
--------------------
Run 35260585795 dispatched ``dependency-cascade.yml`` against the released
``v0.38.30`` while all three fan-out targets were genuinely behind it
(omniintelligence on 0.38.21, omnimemory on 0.38.24, omniclaude on 0.38.27).
Every leg reported ``success``. Not one of them opened, updated or even
examined a pull request.

The cause is a context difference the workflow never accounted for. Its
credentials are declared as ``workflow_call`` secret inputs and read as
``secrets.onexbot-occ-app-id``. On the ``workflow_call`` path ``release.yml``
passes them explicitly, so they arrive. On the ``workflow_dispatch`` path
those input names do not exist, so both resolve empty, ``has_token`` is
``false``, and every subsequent step's ``if:`` guard is false. The job then
has nothing left to do and exits 0.

So the workflow declared a manual trigger that could never do anything, and
reported green every time it was used. That is the failure this repo's own
lab-receipt design is written against: a result that only appears when things
worked cannot distinguish "it failed" from "nobody ran it". Here it was worse
than indistinguishable -- it was actively reassuring.

Two things are fixed and both are asserted below:

* the credential resolves from the passed-in secret OR the org secret, so the
  dispatch path has real credentials;
* a leg with no credentials FAILS. A cascade that cannot authenticate has not
  skipped a bump, it has failed to perform one, and the two must not look the
  same from outside.

Ticket: OMN-18596
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "dependency-cascade.yml"


def _parsed() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _token_check_step() -> dict:
    for job in _parsed()["jobs"].values():
        for step in job.get("steps", []) or []:
            if step.get("id") == "token_check":
                return step
    raise AssertionError("no step with id token_check in the cascade")


class TestTheDispatchPathHasRealCredentials:
    def test_the_workflow_still_declares_a_manual_trigger(self) -> None:
        """If the trigger were removed instead, the rest of this file is moot."""
        triggers = _parsed().get("on", _parsed().get(True, {}))
        assert "workflow_dispatch" in triggers, (
            "this file exists to make the manual trigger work; if the decision "
            "was instead to remove it, delete these tests in the same change "
            "rather than leaving them asserting a path that is gone"
        )

    def test_each_credential_resolves_from_either_source(self) -> None:
        """The whole defect in one assertion.

        Under `workflow_call` only the declared inputs are populated; under
        `workflow_dispatch` only the org secrets are. Reading one name can
        therefore never serve both triggers.
        """
        env = _token_check_step()["env"]
        for var, passed_in, org in (
            ("APP_ID", "secrets.onexbot-occ-app-id", "secrets.ONEXBOT_OCC_APP_ID"),
            (
                "APP_PRIVATE_KEY",
                "secrets.onexbot-occ-private-key",
                "secrets.ONEXBOT_OCC_PRIVATE_KEY",
            ),
        ):
            expr = env[var]
            assert passed_in in expr, (
                f"{var} must still read the workflow_call input, or the release "
                "path that passes it explicitly stops working"
            )
            assert org in expr, (
                f"{var} must also fall back to the org secret, or the "
                "workflow_dispatch path has no credentials and every leg "
                "no-ops while reporting success, as run 35260585795 did"
            )

    def test_the_fallback_is_between_two_forms_of_the_same_app_credential(
        self,
    ) -> None:
        """Not a degradation to the default workflow token (OMN-18273).

        A push authored by the default token has its downstream workflows
        suppressed, so that substitution is the confound OMN-18273 corrected
        and is forbidden. This fallback stays inside the App's own credential.
        """
        body = _WORKFLOW.read_text(encoding="utf-8")
        assert re.search(r"\|\|\s*secrets\.GITHUB_TOKEN", body) is None, (
            "the cascade must never substitute the default workflow token"
        )
        assert "secrets.CROSS_REPO_PAT" not in body, (
            "CROSS_REPO_PAT was retired by OMN-16373; a fallback to it would "
            "reintroduce a long-lived personal token"
        )


class TestACredentiallessLegFailsRatherThanReportingGreen:
    def test_the_token_check_exits_nonzero_when_credentials_are_absent(
        self,
    ) -> None:
        run = _token_check_step()["run"]
        assert "exit 1" in run, (
            "a leg that cannot authenticate must FAIL. Reporting success "
            "makes a cascade that performed no bump indistinguishable from "
            "one that had nothing to bump, which is how run 35260585795 read "
            "as four green jobs having done nothing at all"
        )
        assert "::error::" in run, (
            "the failure must be annotated as an error, not a warning; a "
            "warning on a green job is what nobody read"
        )

    def test_the_run_no_longer_writes_a_has_token_false_path(self) -> None:
        """There is no surviving 'carry on without credentials' branch.

        Asserted on the shipped script rather than on the guards, because a
        `has_token=false` output is only useful to a downstream `if:` that
        silently does nothing -- the shape being removed.
        """
        run = _token_check_step()["run"]
        assert "has_token=false" not in run, (
            "writing has_token=false keeps the silent-skip path alive; the "
            "step should fail instead"
        )

    def test_every_later_step_still_guards_on_the_token_check(self) -> None:
        """The guards may remain, but they must not be the only protection.

        Kept as a regression assertion: if the guards were dropped WITHOUT the
        step failing, a credential-less leg would run `gh` unauthenticated.
        """
        steps = None
        for job in _parsed()["jobs"].values():
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
