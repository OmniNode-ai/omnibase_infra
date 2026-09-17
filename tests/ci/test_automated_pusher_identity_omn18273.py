# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every automated branch push carries the App identity and names its run (OMN-18273).

OMN-18273's first two acceptance criteria are that every automated push carries a
distinct bot identity rather than a human's, and that every automated commit carries
a trailer naming the workflow and the run that produced it. Both were unmet in this
repository: the contract-pin and sibling-lock bots minted an ``onexbot-occ-writer``
installation token and then committed as ``omninode-bot <bot@omninode.ai>`` — a name
that resolves to no GitHub account and to no run — while the plugin-pin cascade
pushed with the default workflow token as ``github-actions[bot]``.

Operating Rule #5: detection that is not a gate is advisory. This file is the gate.

Two properties are asserted, and the second is the one that keeps the first honest:

1. every job that pushes a **branch** mints the App token, checks that mint fail-closed
   (never ``|| secrets.GITHUB_TOKEN``), sets the App's real committer identity, and
   stamps ``Onex-Workflow`` / ``Onex-Run`` trailers on the commit it pushes;
2. the set of workflow files containing ``git push`` is exactly the classified set
   below. A new pusher added without a classification fails this test rather than
   silently inheriting no requirement at all.

The App's committer identity is not invented here. It is read off real commits the
App authored, e.g. ``onex_change_control`` ``30caec1a``:
``onexbot-occ-writer[bot] <307849072+onexbot-occ-writer[bot]@users.noreply.github.com>``.

Why an App-token push is required rather than merely allowed: a push made with the
default workflow token does not start push-driven CI, so a bot PR opened that way sits
with no checks. An App-token push does start it — verified 2026-09-16 on
``omnibase_spi`` run ``35134173847`` (``event: push``,
``actor: onexbot-occ-writer[bot]``, ``conclusion: success``). The earlier org finding
that App tokens were suppressed too was a credential confound: a default
``actions/checkout`` persists a basic-auth ``extraheader`` carrying the workflow token,
which overrides any credential in the remote URL. That is why property 1 requires the
mint to fail closed — a fallback expression reintroduces exactly that confound.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

APP_BOT_NAME = "onexbot-occ-writer[bot]"
APP_BOT_EMAIL = "307849072+onexbot-occ-writer[bot]@users.noreply.github.com"

# Workflow file -> job ids that push a branch and must therefore comply.
BRANCH_PUSHERS: dict[str, tuple[str, ...]] = {
    "omnimarket-contract-pin-refresh.yml": ("refresh",),
    "sibling-lock-refresh.yml": ("refresh",),
    "plugin-pin-cascade.yml": ("update-dockerfile-pins",),
    "dependency-cascade.yml": ("open-bump-pr",),
    "dev-baseline-publisher.yml": ("publish-baseline",),
    # The scheduled release train (OMN-18595). Its `cut` job pushes the
    # changelog-only release branch, so it is a branch pusher on exactly the
    # terms below: it mints the App token with no fallback, commits as the App
    # identity, and stamps both trailers. It is NOT exempt the way the tag
    # pushers above are -- it pushes a commit, and a commit can carry a trailer.
    "release-train-nightly.yml": ("cut",),
    # The lane-census refresh leg (OMN-18606). Its `refresh` job pushes the
    # census bump branch, so it complies on exactly the terms below: the App
    # token is minted with no GITHUB_TOKEN fallback, the commit is attributed to
    # the App identity, and both trailers are stamped. It is NOT exempt the way
    # the tag pushers are -- it pushes a commit, and a commit can carry a
    # trailer. The push identity is load-bearing here rather than cosmetic: a
    # bump PR whose CI never runs cannot clear the staleness gate it exists to
    # clear, which is the whole point of the leg.
    "lane-census-refresh.yml": ("refresh",),
}

# Workflow file -> why it is out of scope. Every exemption is a stated reason, not a
# blanket. `release.yml` and `release-train-lab.yml` are deliberately untouched: an
# App-token force-sync of a release branch fires the production-feeding image builds,
# so changing their push identity is a separate decision with its own blast radius.
EXEMPT: dict[str, str] = {
    "release.yml": (
        "release + main-sync train; already pushes as the App for the jobs that "
        "commit, and its tag/main-sync pushes are a production-adjacent surface "
        "deliberately out of scope for OMN-18273"
    ),
    "release-train-lab.yml": (
        "pushes a tag only; a tag carries no commit, so no trailer can attach to it"
    ),
    "auto-tag-on-merge.yml": (
        "pushes a tag only; a tag carries no commit, so no trailer can attach to it"
    ),
}

TRAILER_WORKFLOW = "Onex-Workflow:"
TRAILER_RUN = "Onex-Run:"

# A push whose refspec names a tag. Tags carry no commit message, so the trailer
# requirement cannot apply to them.
_TAG_PUSH = re.compile(
    r"git push\s+\S*\s*(\"?refs/tags/|\"?\$\{?\{?\s*steps\.\w+\.outputs\.tag)"
)


def _workflow_files() -> list[Path]:
    return sorted(p for p in WORKFLOWS.glob("*.yml"))


def _files_containing_git_push() -> set[str]:
    found = set()
    for path in _workflow_files():
        text = path.read_text(encoding="utf-8")
        if "git push" in text:
            found.add(path.name)
    return found


def _job_run_text(workflow: str, job_id: str) -> str:
    data = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    jobs = data.get("jobs", {})
    assert job_id in jobs, f"{workflow}: job '{job_id}' not found; jobs={sorted(jobs)}"
    job = jobs[job_id]
    chunks: list[str] = []
    for step in job.get("steps", []) or []:
        chunks.append(yaml.safe_dump(step, default_flow_style=False))
    return "\n".join(chunks)


def _declares_workflow_call_secrets(workflow: str) -> bool:
    """True when this workflow is REUSABLE and declares its own secret inputs."""
    data = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    triggers = data.get("on", data.get(True, {})) or {}
    call = triggers.get("workflow_call") or {}
    return bool(call.get("secrets"))


def _in_repo_callers(workflow: str) -> dict[str, dict[str, str]]:
    """Every workflow in this repo that `uses:` ``workflow``, and what it passes.

    Keyed by caller file name; the value is that caller's ``secrets:`` mapping.
    A caller that passes nothing maps to an empty dict, which fails the
    assertions above rather than being skipped.
    """
    callers: dict[str, dict[str, str]] = {}
    for path in _workflow_files():
        if path.name == workflow:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in (data.get("jobs") or {}).values():
            uses = str(job.get("uses", ""))
            if uses.endswith((f"/{workflow}", workflow)):
                callers[path.name] = job.get("secrets") or {}
    return callers


def test_every_git_push_workflow_is_classified() -> None:
    """A new automated pusher cannot appear without a classification.

    Without this, adding a workflow that pushes a branch would inherit no requirement
    and this gate would report green over it.
    """
    classified = set(BRANCH_PUSHERS) | set(EXEMPT)
    actual = _files_containing_git_push()

    unclassified = actual - classified
    assert not unclassified, (
        "workflow(s) run `git push` but are neither listed in BRANCH_PUSHERS nor "
        f"given a stated EXEMPT reason: {sorted(unclassified)}. Classify them in "
        f"{Path(__file__).name} — an unclassified pusher is an unenforced one."
    )

    stale = classified - actual
    assert not stale, (
        f"classified workflow(s) no longer contain `git push`: {sorted(stale)}. "
        "Remove the stale entries so the registry keeps meaning something."
    )


@pytest.mark.parametrize(
    ("workflow", "job_id"),
    [(wf, job) for wf, jobs in BRANCH_PUSHERS.items() for job in jobs],
)
def test_branch_pusher_mints_the_app_token_fail_closed(
    workflow: str, job_id: str
) -> None:
    text = _job_run_text(workflow, job_id)

    assert "actions/create-github-app-token" in text, (
        f"{workflow}:{job_id} pushes a branch but never mints an App installation "
        "token. A push made with the default workflow token does not start "
        "push-driven CI, and it attributes the commit to github-actions[bot] rather "
        "than to a bot this org owns."
    )
    if _declares_workflow_call_secrets(workflow):
        # A REUSABLE pusher cannot name the org-level values in its own job
        # text, and must not: a workflow declaring `workflow_call: secrets:`
        # has its secrets context restricted to the DECLARED names under every
        # trigger, so an org-level name there resolves EMPTY. Writing one in
        # anyway to satisfy a text match would be dead code in a credential
        # expression -- run 35267246993 proved that shape mints nothing while
        # reading like coverage (OMN-18596).
        #
        # The requirement is unchanged, it just moves one level out: the job
        # mints from its declared inputs, and EVERY in-repo caller passes the
        # org-level values into them. Both halves are asserted, so a caller
        # that quietly passed something else is still caught.
        assert "secrets.onexbot-occ-app-id" in text, (
            f"{workflow}:{job_id} is a reusable pusher and must mint from its "
            "declared onexbot-occ-app-id secret input."
        )
        callers = _in_repo_callers(workflow)
        assert callers, (
            f"{workflow} is a reusable pusher that no workflow in this repo "
            "calls; an uncallable pusher cannot be shown to mint from the "
            "onexbot-occ-writer App at all."
        )
        for caller, passed in callers.items():
            assert "ONEXBOT_OCC_APP_ID" in passed.get("onexbot-occ-app-id", ""), (
                f"{caller} calls {workflow} without passing "
                "ONEXBOT_OCC_APP_ID; that leg mints nothing."
            )
            assert "ONEXBOT_OCC_PRIVATE_KEY" in passed.get(
                "onexbot-occ-private-key", ""
            ), (
                f"{caller} calls {workflow} without passing "
                "ONEXBOT_OCC_PRIVATE_KEY; that leg mints nothing."
            )
    else:
        assert "ONEXBOT_OCC_APP_ID" in text and "ONEXBOT_OCC_PRIVATE_KEY" in text, (
            f"{workflow}:{job_id} must mint from the onexbot-occ-writer org secrets."
        )

    # Fail closed. A fallback silently restores the workflow token and with it the
    # exact credential confound that produced the wrong org-wide finding in August.
    fallback = re.search(
        r"steps\.[\w-]+\.outputs\.token\s*\|\|\s*secrets\.GITHUB_TOKEN", text
    )
    assert fallback is None, (
        f"{workflow}:{job_id} falls back to secrets.GITHUB_TOKEN when the mint "
        "fails. That silently pushes as the workflow token, which is suppressed and "
        "mis-attributed, while the job still reports success."
    )


@pytest.mark.parametrize(
    ("workflow", "job_id"),
    [(wf, job) for wf, jobs in BRANCH_PUSHERS.items() for job in jobs],
)
def test_branch_pusher_commits_as_the_app_identity(workflow: str, job_id: str) -> None:
    text = _job_run_text(workflow, job_id)

    assert APP_BOT_NAME in text, (
        f"{workflow}:{job_id} does not set user.name to {APP_BOT_NAME!r}. The "
        "identity a bot commits under is what makes attribution recoverable from "
        "git alone (OMN-18273 AC-4)."
    )
    assert APP_BOT_EMAIL in text, (
        f"{workflow}:{job_id} does not set user.email to {APP_BOT_EMAIL!r}. A "
        "fabricated address such as bot@omninode.ai resolves to no GitHub account, "
        "so the commit renders unlinked and the identity proves nothing."
    )
    # Match the assignment, not any mention. Prose that explains why an old
    # identity was replaced must not read as the identity still being set —
    # otherwise the only way to pass is to leave the change unexplained.
    for forbidden in (
        "bot@omninode.ai",
        "41898282+github-actions[bot]",
        "omninode-bot",
    ):
        assignment = re.search(
            r"git config (?:--global )?user\.(?:name|email) [\"']?"
            + re.escape(forbidden),
            text,
        )
        assert assignment is None, (
            f"{workflow}:{job_id} still assigns the {forbidden!r} identity via "
            "`git config`."
        )


@pytest.mark.parametrize(
    ("workflow", "job_id"),
    [(wf, job) for wf, jobs in BRANCH_PUSHERS.items() for job in jobs],
)
def test_branch_pusher_stamps_workflow_and_run_trailers(
    workflow: str, job_id: str
) -> None:
    text = _job_run_text(workflow, job_id)

    assert TRAILER_WORKFLOW in text, (
        f"{workflow}:{job_id} does not stamp an {TRAILER_WORKFLOW} trailer. "
        "OMN-18273 AC-2 requires every automated commit to name the workflow that "
        "produced it."
    )
    assert TRAILER_RUN in text, (
        f"{workflow}:{job_id} does not stamp an {TRAILER_RUN} trailer. AC-2 "
        "requires the run to be nameable from the commit, with no ledger row and no "
        "session transcript in the path."
    )
    assert "GITHUB_RUN_ID" in text or "github.run_id" in text, (
        f"{workflow}:{job_id} stamps an {TRAILER_RUN} trailer that does not "
        "interpolate the run id, so it names no run."
    )


def test_tag_only_pushes_are_not_silently_counted_as_branch_pushes() -> None:
    """Positive control for the tag/branch split the exemptions rely on.

    A zero from the branch-pusher scan would look identical whether the split works
    or the regex matches nothing at all. This pins a known tag push and a known
    branch push against it.
    """
    tag_line = 'git push origin "refs/tags/${TAG}"'
    branch_line = 'git push origin "$BRANCH"'
    assert _TAG_PUSH.search(tag_line) is not None, "tag push should classify as a tag"
    assert _TAG_PUSH.search(branch_line) is None, (
        "branch push must not classify as a tag"
    )
