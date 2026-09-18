# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The build wiring that stamps the node-inventory label (OMN-18708).

The Python that BUILDS and PARSES the label is falsified in
``tests/unit/runtime/test_node_inventory_image_label_omn18708.py``. What is
pinned here is the part no unit test of that module can reach: that the
Dockerfile actually declares the argument, runs the in-image guard and stamps
the label, and that the sanctioned build path actually performs the second
pass. Each of these is a one-line deletion away from a build that silently
ships an unlabelled image while every Python test stays green.

Read as assertions about text, these are weak. They are here because the
alternative -- a live `docker build` -- needs a daemon this suite does not
have, and because the strong check already exists and runs in the place that
matters: the Dockerfile's own `verify` step re-derives the inventory INSIDE the
image and refuses a stamped value that disagrees with it. These tests exist to
prove that step is still wired, not to prove it works.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omnibase_infra.runtime.node_inventory import NODE_INVENTORY_LABEL

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.runtime"
_DEPLOY_SCRIPT = _REPO_ROOT / "scripts" / "deploy-runtime.sh"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return _DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def deploy_script() -> str:
    return _DEPLOY_SCRIPT.read_text(encoding="utf-8")


class TestTheDockerfileStampsTheLabel:
    def test_the_runtime_stage_declares_the_build_argument(
        self, dockerfile: str
    ) -> None:
        assert re.search(r'^ARG NODE_INVENTORY=""$', dockerfile, re.MULTILINE)

    def test_the_label_is_stamped_from_that_argument(self, dockerfile: str) -> None:
        assert f'LABEL {NODE_INVENTORY_LABEL}="${{NODE_INVENTORY}}"' in dockerfile, (
            "the Dockerfile must stamp the inventory under the same label key the "
            "readback reads, or an image inspection resolves nothing"
        )

    def test_the_in_image_guard_runs_the_verify_subcommand(
        self, dockerfile: str
    ) -> None:
        """Without this, the label says whatever the build script was told to say."""
        assert (
            "python -m omnibase_infra.runtime.node_inventory verify "
            '--expect "${NODE_INVENTORY}"' in dockerfile
        )

    def test_the_argument_is_declared_after_every_copy_in_the_runtime_stage(
        self, dockerfile: str
    ) -> None:
        """Placement is load-bearing, not cosmetic.

        An ARG invalidates the cache for every instruction after it. Declared
        anywhere earlier, the second pass of the two-pass build would re-run
        every COPY and apt layer instead of only the guard.
        """
        lines = dockerfile.splitlines()
        runtime_stage_start = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("FROM ") and " AS runtime" in line
        )
        arg_line = next(
            i for i, line in enumerate(lines) if line.strip() == 'ARG NODE_INVENTORY=""'
        )
        later_copies = [
            i
            for i, line in enumerate(lines)
            if i > arg_line and line.startswith(("COPY ", "RUN apt", "ADD "))
        ]

        assert arg_line > runtime_stage_start
        assert not later_copies, (
            f"instructions that create layers appear after ARG NODE_INVENTORY "
            f"(lines {[i + 1 for i in later_copies]}); the stamping pass would "
            "rebuild them"
        )

    def test_the_guard_has_no_force_or_skip_escape(self, dockerfile: str) -> None:
        guard_block = dockerfile.split('ARG NODE_INVENTORY=""', 1)[1].split("LABEL", 1)[
            0
        ]

        assert "--force" not in guard_block
        assert "--skip" not in guard_block
        assert "|| true" not in guard_block


class TestTheSanctionedBuildPathStampsIt:
    def test_the_first_pass_builds_with_an_empty_inventory(
        self, deploy_script: str
    ) -> None:
        """The inventory is a property of the image and cannot precede it."""
        assert '--build-arg "NODE_INVENTORY="' in deploy_script

    def test_the_second_pass_stamps_what_the_image_emitted(
        self, deploy_script: str
    ) -> None:
        assert '--build-arg "NODE_INVENTORY=${inventory}"' in deploy_script

    def test_the_inventory_is_read_from_the_image_not_the_build_context(
        self, deploy_script: str
    ) -> None:
        """A host-side scan would enumerate the wrong set.

        Contracts come from `onex.nodes` entry points across every installed
        sibling distribution, not from this repository's tree.
        """
        assert 'docker run --rm --entrypoint python "${image_ref}"' in deploy_script
        assert "node_inventory emit" in deploy_script

    def test_build_images_calls_the_stamping_pass(self, deploy_script: str) -> None:
        assert "stamp_node_inventory cmd build_scope compose_args" in deploy_script

    def test_every_stamping_failure_fails_the_deploy(self, deploy_script: str) -> None:
        """Fail-closed: an ABSENT label must mean 'not from the sanctioned path'.

        If any branch here returned 0, an unstamped image could reach a lane
        from the governed path and the label's absence would stop being a
        signal at all.
        """
        body = deploy_script.split("stamp_node_inventory() {", 1)[1].split("\n}\n", 1)[
            0
        ]
        branches = re.findall(r"^\s+log_error .*\n\s+(return \d+)", body, re.MULTILINE)

        assert branches, "expected the stamping function to have error branches"
        assert set(branches) == {"return 1"}
        assert "|| true" not in body.replace("|| true)", "")


class TestThereIsOnlyOneHasher:
    def test_the_inventory_module_imports_the_canonical_hasher(self) -> None:
        """OMN-18709 declares one canonical form; a second one would drift."""
        source = (
            _REPO_ROOT / "src" / "omnibase_infra" / "runtime" / "node_inventory.py"
        ).read_text(encoding="utf-8")

        assert (
            "from omnibase_infra.runtime.util_contract_content_hash import "
            "contract_content_hash" in source
        )
        assert "hashlib" not in source, (
            "node_inventory must not hash anything itself; it calls the one "
            "canonical function so an outside consumer's reproduction stays valid"
        )

    def test_the_discovery_pass_hashes_with_the_same_function(self) -> None:
        source = (
            _REPO_ROOT
            / "src"
            / "omnibase_infra"
            / "runtime"
            / "auto_wiring"
            / "discovery.py"
        ).read_text(encoding="utf-8")

        assert "contract_content_hash(contract_path)" in source
        assert "hashlib" not in source


class TestTheReceiptModuleStaysStdlibOnly:
    def test_the_triple_model_did_not_introduce_a_repo_import(self) -> None:
        """Run 34235502322 died at import on a bare runner and took a delivery.

        The existing suite has a general version of this guard; this one names
        the specific temptation OMN-18708 created -- importing
        ``ModelNodeInventoryEntry`` from ``omnibase_infra.runtime`` instead of
        defining the stdlib twin.
        """
        source = (_REPO_ROOT / "scripts" / "ci" / "lab_pass_receipt.py").read_text(
            encoding="utf-8"
        )

        assert "from omnibase_infra" not in source
        assert "import omnibase_infra" not in source
