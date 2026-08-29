# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15796 — the sending half of dev-merge -> onex-dev delivery.

Before this, a merge to this repo's ``dev`` produced nothing deployable. The
runtime image came from ``build-workspace-candidate-runtime.yml``, which was
``workflow_dispatch``-only by design; the migration bundle came from
``build-and-push-migrate-image.yml``, which push-triggers on ``main`` while
migrations land on ``dev``. Both images therefore existed only when a human
remembered to dispatch them, read two digests out of two runs, hand-edited two
pins in ``omninode_infra``'s k8s manifests, and merged that. In practice the
sequence was skipped and the omission was invisible: the deploy re-applied the
unchanged pins and reported green while onex-dev ran a four-day-old candidate
(OMN-15796 evidence, run 31328815070).

``deliver-dev-candidate-to-staging.yml`` makes the chain the merge's own
consequence: build both images from ONE dev commit, then tell
``omninode_infra`` about them via ``repository_dispatch`` — the same cross-repo
mechanism this org already runs for omniweb -> omninode_infra.

The invariant these tests exist to protect is SINGLE_SOURCE_REV_BUNDLE: the
runtime image and the migration bundle must come from the same revision, and
a bundle that is only half-built must not be announced at all. A runtime
rolled forward against an unmigrated schema is a worse outcome than not
deploying.

Static artifact assertions only — no network, no Docker.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
DELIVER = WORKFLOWS / "deliver-dev-candidate-to-staging.yml"
RUNTIME_BUILD = WORKFLOWS / "build-workspace-candidate-runtime.yml"
MIGRATE_BUILD = WORKFLOWS / "build-and-push-migrate-image.yml"

DISPATCH_EVENT_TYPE = "omnibase-infra-dev-candidate"
TARGET_REPO = "OmniNode-ai/omninode_infra"


def _load(path: Path) -> dict[Any, Any]:
    assert path.is_file(), f"missing {path}"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def deliver() -> dict[Any, Any]:
    return _load(DELIVER)


@pytest.fixture(scope="module")
def deliver_text() -> str:
    return DELIVER.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The two build workflows must be callable, and must say what they built
# ---------------------------------------------------------------------------


class TestBuildWorkflowsAreCallable:
    def test_runtime_build_is_callable_and_reports_its_image(self) -> None:
        """``workflow_call`` is what turns the manual dispatch into a link in
        a chain. Without the outputs the caller has an image it cannot name,
        and naming it is the whole point — the digest is what gets pinned."""
        doc = _load(RUNTIME_BUILD)
        triggers = doc[True]
        assert "workflow_call" in triggers, (
            "build-workspace-candidate-runtime.yml must be callable so the "
            "delivery workflow can build a candidate from a dev merge "
            "(OMN-15796 (a) / OMN-14044)"
        )
        outputs = triggers["workflow_call"]["outputs"]
        assert "image-ref" in outputs, outputs

    def test_runtime_build_keeps_its_manual_dispatch(self) -> None:
        """The documented one-dispatch escape hatch stays: rebuilding a
        candidate for a sibling-repo change is not a dev merge here."""
        assert "workflow_dispatch" in _load(RUNTIME_BUILD)[True]

    def test_migrate_build_is_callable_at_a_caller_chosen_ref(self) -> None:
        """The caller must pin the ref, not inherit whatever the runner's
        default checkout resolves to — that is how the two images stay on one
        revision."""
        triggers = _load(MIGRATE_BUILD)[True]
        assert "workflow_call" in triggers
        assert "git-ref" in triggers["workflow_call"]["inputs"]
        assert "image-ref" in triggers["workflow_call"]["outputs"]

    def test_callee_input_defaults_do_not_break_the_callers_graph(self) -> None:
        """OMN-16906 — "callable" is not the same as "callable without killing
        the caller".

        This suite asserted the ``workflow_call`` blocks existed and that their
        input/output NAMES lined up, and stayed green for the entire two-day
        outage, because the defect was one level down: ``no-cache`` was declared
        ``type: boolean`` with ``default: "false"``, a YAML string. GitHub
        type-checks a callee's input defaults while compiling the CALLER's
        graph, so the delivery workflow returned ``startup_failure`` — no job,
        no log — on every run while both callees kept passing their own
        dispatch runs.

        Bisect evidence: control run 33224162317 ``startup_failure``; the same
        file with that one scalar unquoted, run 33224392268, compiled.

        The repo-wide ratchet is
        ``tests/ci/test_workflow_input_default_type_parity.py``; this assertion
        keeps the delivery chain's own suite from passing through a recurrence.
        """
        from scripts.ci.check_workflow_input_default_types import scan_document

        for path in (RUNTIME_BUILD, MIGRATE_BUILD, DELIVER):
            violations = scan_document(path.name, _load(path))
            assert not violations, "\n".join(v.render() for v in violations)

    def test_migrate_build_resolves_a_real_digest_after_push(self) -> None:
        """A ``repo:tag`` reference is not an identity. The consumer
        (``run-migrations.sh``) refuses anything without ``@sha256:``, so an
        un-resolved output would fail the deploy — resolve it here, from ECR,
        after the push."""
        text = MIGRATE_BUILD.read_text(encoding="utf-8")
        assert "aws ecr describe-images" in text
        assert "imageDigest" in text


# ---------------------------------------------------------------------------
# The delivery workflow
# ---------------------------------------------------------------------------


class TestDeliveryTrigger:
    def test_fires_on_a_dev_merge(self, deliver: dict[Any, Any]) -> None:
        triggers = deliver[True]
        assert "dev" in triggers["push"]["branches"], (
            "the delivery chain must be the merge's own consequence; a "
            "trigger a human has to remember is the defect, not the fix"
        )

    def test_watches_both_the_runtime_and_the_migration_source(
        self, deliver: dict[Any, Any]
    ) -> None:
        """One workflow covers both halves, so its path filter must be the
        UNION. ``docker/**`` carries ``docker/migrations/**``; narrowing to
        runtime sources alone would silently drop OMN-15054's half."""
        paths = deliver[True]["push"]["paths"]
        assert "src/**" in paths, paths
        assert any(p.startswith("docker/") for p in paths), paths

    def test_keeps_a_manual_dispatch(self, deliver: dict[Any, Any]) -> None:
        assert "workflow_dispatch" in deliver[True]

    def test_collapses_a_burst_of_merges(self, deliver: dict[Any, Any]) -> None:
        """A 75-minute image build per merge would queue indefinitely under
        normal merge traffic, and every superseded run delivers code that is
        already stale. Cancel-in-progress makes the delivered candidate dev
        HEAD by construction."""
        concurrency = deliver["concurrency"]
        assert concurrency["cancel-in-progress"] is True, concurrency


class TestSingleSourceRevBundle:
    def test_both_images_are_built_from_the_same_commit(
        self, deliver: dict[Any, Any]
    ) -> None:
        """The migration bundle must be pinned to ``github.sha``.

        The runtime build reaches the same commit implicitly — a called
        workflow's ``github.sha`` IS the caller's, so its default checkout is
        this merge. The migrate build has nothing to infer that from, so it is
        stated. Anything else (``dev``, a branch name, an empty ref that falls
        back to the default branch) lets the bundle drift onto a later commit
        the moment a second PR lands mid-build, and nothing anywhere goes red:
        the two images are simply from different revisions.
        """
        jobs = deliver["jobs"]
        runtime_job = jobs["build-runtime-candidate"]
        migrate_job = jobs["build-migrate-bundle"]
        assert runtime_job["uses"].endswith("build-workspace-candidate-runtime.yml")
        assert migrate_job["uses"].endswith("build-and-push-migrate-image.yml")
        assert migrate_job["with"]["git-ref"].strip() == "${{ github.sha }}", (
            "SINGLE_SOURCE_REV_BUNDLE: the migration bundle must be built from "
            "this merge's commit, or the schema the Job applies and the code "
            f"that reads it disagree. Got: {migrate_job['with']['git-ref']!r}"
        )

    def test_sibling_ref_is_never_this_repos_commit_sha(
        self, deliver: dict[Any, Any]
    ) -> None:
        """``sibling_ref`` is a different axis from the bundle revision.

        It names the ref that omnibase_core / omnibase_compat / omnimarket are
        CLONED at, so it has to be a ref those repos have. Wiring
        ``github.sha`` into it — the natural-looking mistake, since every other
        ref in this workflow is that commit — makes the build clone
        omnibase_core at an omnibase_infra SHA and fail outright.
        """
        sibling_ref = deliver["jobs"]["build-runtime-candidate"]["with"]["sibling_ref"]
        assert "github.sha" not in str(sibling_ref), sibling_ref
        assert "dev" in str(sibling_ref), sibling_ref

    def test_a_half_built_bundle_is_never_announced(
        self, deliver: dict[Any, Any]
    ) -> None:
        """If either build fails, the dispatch must not fire. Announcing one
        half rolls the runtime forward onto a schema that was never
        migrated — strictly worse than not deploying."""
        dispatch = deliver["jobs"]["dispatch-to-staging"]
        needs = dispatch["needs"]
        assert "build-runtime-candidate" in needs
        assert "build-migrate-bundle" in needs
        condition = str(dispatch.get("if", ""))
        assert "always()" not in condition, (
            "an always() dispatch job would announce a bundle whose builds "
            f"failed; got if: {condition!r}"
        )


class TestDispatch:
    def test_targets_omninode_infra_with_the_agreed_event_type(
        self, deliver_text: str
    ) -> None:
        """The event type is a contract with the receiving workflow's
        ``repository_dispatch.types`` list; a typo here is a silent no-op on
        GitHub's side, which is precisely the failure shape this ticket is
        about."""
        assert DISPATCH_EVENT_TYPE in deliver_text
        assert f"repos/{TARGET_REPO}/dispatches" in deliver_text

    def test_payload_carries_both_refs_and_the_source_revision(
        self, deliver_text: str
    ) -> None:
        for field in (
            "runtime_image_ref",
            "migrate_image_ref",
            "source_repo",
            "source_sha",
        ):
            assert field in deliver_text, f"payload field {field} missing"

    def test_payload_is_built_with_jq_not_string_interpolation(
        self, deliver_text: str
    ) -> None:
        """Hand-concatenated JSON breaks on the first value containing a quote
        and is unreviewable; jq -n --arg is injection-safe by construction."""
        assert "jq -n" in deliver_text

    def test_uses_the_onexbot_app_credentials_not_a_long_lived_pat(
        self, deliver_text: str
    ) -> None:
        """OMN-16373 retired CROSS_REPO_PAT in favour of a short-lived,
        repo-scoped App installation token. ``repository_dispatch`` needs only
        Contents:write on the target."""
        assert "ONEXBOT_OCC_APP_ID" in deliver_text
        assert "ONEXBOT_OCC_PRIVATE_KEY" in deliver_text
        assert "secrets.CROSS_REPO_PAT" not in deliver_text

    def test_app_token_is_scoped_to_the_target_repo_only(
        self, deliver_text: str
    ) -> None:
        assert "repositories: omninode_infra" in deliver_text
        assert "permission-contents: write" in deliver_text


class TestWorkflowHygiene:
    def test_third_party_actions_are_pinned_to_a_commit_sha(
        self, deliver_text: str
    ) -> None:
        refs = re.findall(r"^\s*uses:\s*([^\s#]+)", deliver_text, re.MULTILINE)
        assert refs
        for ref in refs:
            if ref.startswith("./"):
                continue  # local reusable workflow, versioned with this repo
            assert "@" in ref, f"{ref} is missing an immutable ref"
            _, sha = ref.rsplit("@", 1)
            assert re.fullmatch(r"[0-9a-f]{40}", sha), (
                f"{ref} is not pinned to a 40-character commit SHA"
            )

    def test_runs_on_the_policy_trusted_runner(self, deliver: dict[Any, Any]) -> None:
        """config/runner_routing_policy.yaml allowlists hosted runners
        per-file; this workflow is not on that list, so it must use the
        trusted-runner variable."""
        dispatch = deliver["jobs"]["dispatch-to-staging"]
        assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" in str(dispatch["runs-on"])

    def test_has_no_skip_or_bypass_tokens(self, deliver_text: str) -> None:
        assert "[skip-" not in deliver_text
        assert "--no-verify" not in deliver_text
        assert "continue-on-error" not in deliver_text
