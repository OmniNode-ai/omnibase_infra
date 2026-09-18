# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The node-inventory image label (OMN-18708).

These are the falsifiers named on the ticket's acceptance criteria:

AC1  the label parses into one entry per contract the build's discovery pass
     found, and a changed node set produces a different label value
AC2  a blank or unparseable inventory is refused rather than emitted empty
AC4  editing a contract's BODY, with its name and version untouched, moves the
     hash -- which is the whole reason the third field is a content hash and
     not the identity hash it replaced

AC3 lives in ``tests/ci/test_lab_pass_receipt_node_inventory_omn18708.py``,
beside the receipt module it is about.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.errors.error_node_inventory import NodeInventoryError
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.node_inventory import (
    NODE_INVENTORY_LABEL,
    NODE_INVENTORY_SCHEMA,
    ModelNodeInventoryEntry,
    build_node_inventory,
    main,
    parse_node_inventory_label,
    read_image_node_inventory,
    render_node_inventory_label,
    resolve_entry_to_contract_path,
)
from omnibase_infra.runtime.util_contract_content_hash import contract_content_hash

pytestmark = pytest.mark.unit


CONTRACT_BODY = """\
name: {name}
node_type: COMPUTE_GENERIC
description: a fixture contract
node_version: {version}
contract_version:
  major: 1
  minor: 0
  patch: 0
"""


def _write_contract(root: Path, name: str, version: str = "1.0.0") -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "contract.yaml"
    path.write_text(CONTRACT_BODY.format(name=name, version=version), encoding="utf-8")
    return path


def _discovered(
    path: Path, name: str, version: str = "1.0.0"
) -> ModelDiscoveredContract:
    """A contract as the discovery pass records one, hash included."""
    return ModelDiscoveredContract(
        name=name,
        node_type="COMPUTE_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        node_version=version,
        contract_path=path,
        contract_content_hash=contract_content_hash(path),
        entry_point_name=name,
        package_name="omnibase_infra",
    )


def _manifest(*contracts: ModelDiscoveredContract) -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(contracts=tuple(contracts))


# ---------------------------------------------------------------------------
# AC1 -- one entry per contract, and a changed node set moves the value
# ---------------------------------------------------------------------------
class TestAc1LabelEnumeratesTheDiscoveredContracts:
    def test_one_entry_per_discovered_contract(self, tmp_path: Path) -> None:
        paths = {name: _write_contract(tmp_path, name) for name in ("alpha", "beta")}
        manifest = _manifest(*(_discovered(p, n) for n, p in paths.items()))

        entries = build_node_inventory(manifest)

        assert len(entries) == len(manifest.contracts)
        assert [e.name for e in entries] == ["alpha", "beta"]
        for entry in entries:
            assert entry.contract_content_hash == contract_content_hash(
                paths[entry.name]
            )

    def test_the_rendered_label_parses_back_to_the_same_entries(
        self, tmp_path: Path
    ) -> None:
        manifest = _manifest(
            *(
                _discovered(_write_contract(tmp_path, n), n)
                for n in ("alpha", "beta", "gamma")
            )
        )
        entries = build_node_inventory(manifest)

        assert (
            parse_node_inventory_label(render_node_inventory_label(entries)) == entries
        )

    def test_a_changed_node_set_produces_a_different_label_value(
        self, tmp_path: Path
    ) -> None:
        alpha = _discovered(_write_contract(tmp_path, "alpha"), "alpha")
        beta = _discovered(_write_contract(tmp_path, "beta"), "beta")

        one_node = render_node_inventory_label(build_node_inventory(_manifest(alpha)))
        two_nodes = render_node_inventory_label(
            build_node_inventory(_manifest(alpha, beta))
        )

        assert one_node != two_nodes

    def test_the_value_is_stable_across_iteration_order(self, tmp_path: Path) -> None:
        """Two builds of one commit must render byte-identical strings.

        Otherwise a byte comparison of two labels compares iteration orders
        rather than node sets, and the guard inside the image would fail on a
        build that changed nothing.
        """
        alpha = _discovered(_write_contract(tmp_path, "alpha"), "alpha")
        beta = _discovered(_write_contract(tmp_path, "beta"), "beta")

        forwards = render_node_inventory_label(
            build_node_inventory(_manifest(alpha, beta))
        )
        backwards = render_node_inventory_label(
            build_node_inventory(_manifest(beta, alpha))
        )

        assert forwards == backwards

    def test_the_label_value_is_one_line(self, tmp_path: Path) -> None:
        """It crosses a --build-arg and a GITHUB_OUTPUT boundary, both line-oriented."""
        manifest = _manifest(_discovered(_write_contract(tmp_path, "alpha"), "alpha"))

        assert "\n" not in render_node_inventory_label(build_node_inventory(manifest))

    def test_the_label_declares_its_schema(self, tmp_path: Path) -> None:
        manifest = _manifest(_discovered(_write_contract(tmp_path, "alpha"), "alpha"))

        payload = json.loads(
            render_node_inventory_label(build_node_inventory(manifest))
        )

        assert payload["schema"] == NODE_INVENTORY_SCHEMA


# ---------------------------------------------------------------------------
# AC2 -- an empty or unparseable inventory is refused, never emitted
# ---------------------------------------------------------------------------
class TestAc2EmptyOrUnparseableIsRefused:
    def test_the_label_builder_raises_on_an_empty_inventory(self) -> None:
        with pytest.raises(NodeInventoryError, match="empty node inventory"):
            render_node_inventory_label([])

    def test_a_populated_inventory_is_the_positive_control(
        self, tmp_path: Path
    ) -> None:
        """The refusal above means something only if the same call works."""
        manifest = _manifest(_discovered(_write_contract(tmp_path, "alpha"), "alpha"))

        assert render_node_inventory_label(build_node_inventory(manifest))

    def test_building_from_a_manifest_with_no_contracts_raises(self) -> None:
        with pytest.raises(NodeInventoryError, match="no contracts"):
            build_node_inventory(_manifest())

    def test_a_contract_with_no_content_hash_raises(self, tmp_path: Path) -> None:
        """A null hash is 'not established', never 'assume it matches'."""
        unhashed = ModelDiscoveredContract(
            name="alpha",
            node_type="COMPUTE_GENERIC",
            contract_version=ModelContractVersion(major=1, minor=0, patch=0),
            contract_path=_write_contract(tmp_path, "alpha"),
            entry_point_name="alpha",
            package_name="omnibase_infra",
        )

        with pytest.raises(NodeInventoryError, match="carries no content hash"):
            build_node_inventory(_manifest(unhashed))

    def test_duplicate_node_names_are_refused(self, tmp_path: Path) -> None:
        path_a = _write_contract(tmp_path / "a", "alpha")
        path_b = _write_contract(tmp_path / "b", "alpha", version="2.0.0")

        with pytest.raises(NodeInventoryError, match="duplicate node name"):
            build_node_inventory(
                _manifest(
                    _discovered(path_a, "alpha"),
                    _discovered(path_b, "alpha", version="2.0.0"),
                )
            )

    @pytest.mark.parametrize(
        ("raw", "reason"),
        [
            (None, "absent or blank"),
            ("", "absent or blank"),
            ("   ", "absent or blank"),
            ("not json at all", "not valid JSON"),
            ("[]", "must be a JSON object"),
            ('{"schema":"node_inventory.v0","nodes":[]}', "declares schema"),
            ('{"schema":"node_inventory.v1","nodes":[]}', "empty node list"),
            ('{"schema":"node_inventory.v1","nodes":{}}', "must be a list"),
            ('{"schema":"node_inventory.v1","nodes":[],"extra":1}', "unknown"),
            ('{"schema":"node_inventory.v1","nodes":[{"name":"a"}]}', "unreadable"),
        ],
    )
    def test_parsing_refuses_everything_it_does_not_fully_understand(
        self, raw: str | None, reason: str
    ) -> None:
        with pytest.raises(NodeInventoryError, match=reason):
            parse_node_inventory_label(raw)

    def test_a_truncated_hash_is_refused_rather_than_normalised(self) -> None:
        with pytest.raises(NodeInventoryError, match="64 lowercase hex"):
            ModelNodeInventoryEntry(
                name="alpha", node_version="1.0.0", contract_content_hash="deadbeef"
            )

    def test_an_uppercase_hash_is_refused(self) -> None:
        with pytest.raises(NodeInventoryError, match="64 lowercase hex"):
            ModelNodeInventoryEntry(
                name="alpha", node_version="1.0.0", contract_content_hash="A" * 64
            )


# ---------------------------------------------------------------------------
# AC4 -- the hash is a CONTENT hash, so a rewritten body is detectable
# ---------------------------------------------------------------------------
class TestAc4ADriftedContractBodyMovesTheHash:
    def test_editing_the_body_with_name_and_version_untouched_changes_the_label(
        self, tmp_path: Path
    ) -> None:
        path = _write_contract(tmp_path, "alpha")
        before = render_node_inventory_label(
            build_node_inventory(_manifest(_discovered(path, "alpha")))
        )

        # The drift class the field exists for: the body changes, the identity
        # does not. Under the name-and-version hash this replaced, the value
        # here would be unchanged and the drift invisible.
        path.write_text(
            path.read_text(encoding="utf-8") + "description: rewritten\n",
            encoding="utf-8",
        )
        after = render_node_inventory_label(
            build_node_inventory(_manifest(_discovered(path, "alpha")))
        )

        assert before != after
        assert json.loads(before)["nodes"][0]["name"] == "alpha"
        assert json.loads(after)["nodes"][0]["name"] == "alpha"
        assert (
            json.loads(before)["nodes"][0]["node_version"]
            == json.loads(after)["nodes"][0]["node_version"]
        )

    def test_the_label_hash_is_the_one_canonical_hasher_not_a_second_one(
        self, tmp_path: Path
    ) -> None:
        """No second implementation: the value must equal the declared function's.

        A shell reproduction of the same canonical form is asserted by
        OMN-18709's own suite; what matters here is that this module does not
        introduce a competing definition.
        """
        path = _write_contract(tmp_path, "alpha")

        entries = build_node_inventory(_manifest(_discovered(path, "alpha")))

        assert entries[0].contract_content_hash == contract_content_hash(path)

    def test_a_crlf_checkout_hashes_identically(self, tmp_path: Path) -> None:
        """Line endings are the one difference that is never a contract change."""
        lf = tmp_path / "lf" / "contract.yaml"
        lf.parent.mkdir(parents=True)
        lf.write_bytes(b"name: alpha\nnode_version: 1.0.0\n")
        crlf = tmp_path / "crlf" / "contract.yaml"
        crlf.parent.mkdir(parents=True)
        crlf.write_bytes(b"name: alpha\r\nnode_version: 1.0.0\r\n")

        assert contract_content_hash(lf) == contract_content_hash(crlf)


# ---------------------------------------------------------------------------
# the readback -- the R5 direction, with no cluster access
# ---------------------------------------------------------------------------
class TestReadbackFromAnImage:
    def test_a_stamped_label_reads_back_into_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = _manifest(_discovered(_write_contract(tmp_path, "alpha"), "alpha"))
        value = render_node_inventory_label(build_node_inventory(manifest))

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a,
                returncode=0,
                stdout=json.dumps({NODE_INVENTORY_LABEL: value}),
                stderr="",
            ),
        )

        assert read_image_node_inventory("sha256:deadbeef")[0].name == "alpha"

    def test_an_image_with_no_inventory_label_is_refused_not_reported_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The negative the ticket cites: today's images carry no such label."""
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a,
                returncode=0,
                stdout=json.dumps({"org.opencontainers.image.revision": "abc"}),
                stderr="",
            ),
        )

        with pytest.raises(NodeInventoryError, match="absent or blank"):
            read_image_node_inventory("sha256:deadbeef")

    def test_a_failing_docker_inspect_is_an_error_not_an_empty_inventory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a, returncode=1, stdout="", stderr="No such image"
            ),
        )

        with pytest.raises(NodeInventoryError, match="exited 1"):
            read_image_node_inventory("sha256:nope")

    def test_an_entry_resolves_back_to_the_contract_file_it_was_built_from(
        self, tmp_path: Path
    ) -> None:
        path = _write_contract(tmp_path, "alpha")
        _write_contract(tmp_path, "beta")
        entry = build_node_inventory(_manifest(_discovered(path, "alpha")))[0]

        assert resolve_entry_to_contract_path(entry, [tmp_path]) == path

    def test_resolution_matches_on_the_hash_so_a_renamed_contract_still_resolves(
        self, tmp_path: Path
    ) -> None:
        path = _write_contract(tmp_path, "alpha")
        entry = ModelNodeInventoryEntry(
            name="a-name-nothing-on-disk-uses",
            node_version="1.0.0",
            contract_content_hash=contract_content_hash(path),
        )

        assert resolve_entry_to_contract_path(entry, [tmp_path]) == path

    def test_a_drifted_body_fails_to_resolve_and_says_why(self, tmp_path: Path) -> None:
        path = _write_contract(tmp_path, "alpha")
        entry = build_node_inventory(_manifest(_discovered(path, "alpha")))[0]
        path.write_text(
            "name: alpha\nnode_version: 1.0.0\nnew: field\n", encoding="utf-8"
        )

        with pytest.raises(NodeInventoryError, match="this checkout does not contain"):
            resolve_entry_to_contract_path(entry, [tmp_path])


# ---------------------------------------------------------------------------
# the CLI surface the build and the board resolver call
# ---------------------------------------------------------------------------
class TestCliSurface:
    @pytest.mark.parametrize("command", ["emit", "verify", "inspect-image", "resolve"])
    def test_every_command_the_build_and_the_resolver_depend_on_exists(
        self, command: str
    ) -> None:
        """Drift-proofing: renaming one of these breaks a Dockerfile or a caller."""
        from omnibase_infra.runtime.node_inventory import build_parser

        actions = build_parser()._subparsers._group_actions
        assert command in actions[0].choices

    def test_verify_declares_no_force_or_skip_option(self) -> None:
        """A guard with an escape hatch is not a guard.

        Read from the parser's own option strings, so adding one is a red test
        rather than a review catch.
        """
        from omnibase_infra.runtime.node_inventory import build_parser

        verify = build_parser()._subparsers._group_actions[0].choices["verify"]
        options = {s for action in verify._actions for s in action.option_strings}

        assert options == {"-h", "--help", "--expect"}

    def test_inspect_image_reports_a_missing_label_as_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a, returncode=0, stdout="{}", stderr=""
            ),
        )

        assert main(["inspect-image", "sha256:deadbeef"]) == 1
        assert "absent or blank" in capsys.readouterr().err
