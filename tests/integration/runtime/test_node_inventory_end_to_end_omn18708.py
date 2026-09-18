# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End to end: a real discovery pass reaches a receipt as triples (OMN-18708).

The unit suites for this change each hold one end still. This one holds none
of them: it runs the repository's REAL contract discovery pass over the real
installed entry points, serialises the manifest the way the health server
serialises it, serves it over a real HTTP socket, and reads it back with the
receipt module's real check -- the same call the ``.201`` compose-dev emitter
makes against the live lane.

What that buys over the unit tests, concretely. Every unit test of the check
hands it a fixture body somebody wrote by hand, so all of them would still
pass if the discovery pass stopped populating the hash, if the field were
renamed on the model, if ``model_dump_json`` dropped it, or if the check read a
key the manifest does not serve. Each of those is a silent break of the hop
this ticket exists to close, and each of them fails HERE.

The two ends it ties together:

* the **label** end -- what an image ships, stamped by the build;
* the **manifest** end -- what a lane wired, read by the receipt.

They are deliberately not asserted equal: a lane wires the profile-filtered
subset of what the image ships. What is asserted is that the third field
AGREES for every node in both, because that is what makes a drifted contract
body detectable from either end.

Measured against the live ``.201`` dev lane on 2026-09-18, before this change:
303 contracts served, 303 with no content hash, and no
``com.omninode.node_inventory`` label on the running image. This test is the
standing form of that readback.
"""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.node_inventory import (
    build_node_inventory,
    parse_node_inventory_label,
    render_node_inventory_label,
    resolve_entry_to_contract_path,
)
from omnibase_infra.runtime.util_contract_content_hash import contract_content_hash
from scripts.ci.lab_pass_receipt import (
    INTROSPECTION_MANIFEST_PATH,
    NODE_INVENTORY_CHECK,
    EnumLabLane,
    EnumLabPassResult,
    build_receipt,
    check_node_inventory,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def discovered() -> ModelAutoWiringManifest:
    """The real discovery pass, over the real installed entry points."""
    manifest = discover_contracts()
    if not manifest.contracts:
        pytest.skip(
            "no onex.nodes entry points are installed in this environment, so "
            "there is no discovery pass to exercise"
        )
    return manifest


@pytest.fixture(scope="module")
def served_manifest_url(discovered: ModelAutoWiringManifest) -> Iterator[str]:
    """Serve the manifest over a real socket, as the health server does.

    ``model_dump_json`` is the exact call
    ``ServiceHealth._handle_introspection_manifest`` makes, so a field that
    does not survive serialisation fails here rather than in production.
    """
    body = discovered.model_dump_json().encode("utf-8")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            """Silence the default stderr access log."""

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}{INTROSPECTION_MANIFEST_PATH}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestTheRealDiscoveryPassReachesAReceipt:
    def test_the_check_passes_against_a_real_served_manifest(
        self, served_manifest_url: str, discovered: ModelAutoWiringManifest
    ) -> None:
        probe = check_node_inventory(served_manifest_url, 30)

        assert probe.check.outcome.value == "pass", probe.check.evidence
        assert probe.check.name == NODE_INVENTORY_CHECK
        assert len(probe.triples) == len(discovered.contracts)

    def test_the_triples_equal_the_served_manifests(
        self, served_manifest_url: str, discovered: ModelAutoWiringManifest
    ) -> None:
        """AC3, against real data rather than a hand-written fixture."""
        probe = check_node_inventory(served_manifest_url, 30)

        served = {
            c.name: (c.node_version, c.contract_content_hash)
            for c in discovered.contracts
        }
        assert {
            t.name: (t.node_version, t.contract_content_hash) for t in probe.triples
        } == served

    def test_a_receipt_built_from_the_probe_carries_them(
        self, served_manifest_url: str
    ) -> None:
        from datetime import UTC, datetime

        probe = check_node_inventory(served_manifest_url, 30)
        now = datetime.now(tz=UTC)

        receipt = build_receipt(
            sha="0" * 40,
            lane=EnumLabLane.COMPOSE_DEV,
            started_at=now,
            finished_at=now,
            checks=[probe.check],
            agent_command_id=None,
            node_inventory=probe.triples,
        )

        assert receipt.result is EnumLabPassResult.PASS
        assert len(receipt.node_inventory) == len(probe.triples)
        assert "node_inventory" in json.loads(receipt.to_json())

    def test_every_served_contract_carries_a_content_hash(
        self, discovered: ModelAutoWiringManifest
    ) -> None:
        """The gap this change closes, asserted as the standing state.

        On the live lane before this change: 303 served, 303 with none.
        """
        missing = [c.name for c in discovered.contracts if not c.contract_content_hash]

        assert not missing, (
            f"{len(missing)} contract(s) carry no content hash: {missing[:10]}"
        )

    def test_no_served_node_version_is_a_stringified_mapping(
        self, discovered: ModelAutoWiringManifest
    ) -> None:
        """The live lane served ``"{'major': 1, 'minor': 0, 'patch': 0}"``.

        It is one third of a triple a gate compares, so a Python repr is not a
        cosmetic defect here.
        """
        reprs = [
            (c.name, c.node_version)
            for c in discovered.contracts
            if c.node_version.startswith("{")
        ]

        assert not reprs, reprs[:10]


class TestTheLabelAndTheManifestAgree:
    def test_the_label_hash_equals_the_manifest_hash_for_every_node(
        self, discovered: ModelAutoWiringManifest
    ) -> None:
        """AC4 at the join: both ends compute it with the one canonical hasher."""
        entries = build_node_inventory(discovered)
        served = {c.name: c.contract_content_hash for c in discovered.contracts}

        assert entries
        for entry in entries:
            assert entry.contract_content_hash == served[entry.name]

    def test_the_rendered_label_round_trips(
        self, discovered: ModelAutoWiringManifest
    ) -> None:
        entries = build_node_inventory(discovered)

        assert (
            parse_node_inventory_label(render_node_inventory_label(entries)) == entries
        )

    def test_the_label_is_a_plausible_size_for_an_oci_label(
        self, discovered: ModelAutoWiringManifest
    ) -> None:
        """A label is image config, not a layer; this bounds what is stamped."""
        value = render_node_inventory_label(build_node_inventory(discovered))

        assert len(value) < 1_000_000, (
            f"the inventory label is {len(value)} bytes for "
            f"{len(discovered.contracts)} contracts; an image config that large "
            "is a registry problem, not an annotation"
        )

    def test_an_entry_resolves_back_to_a_real_contract_file_in_this_checkout(
        self, discovered: ModelAutoWiringManifest
    ) -> None:
        """The R5 direction, walked against the repository rather than a fixture.

        Scoped to contracts whose file is under THIS repository's ``src``.
        Most discovered contracts are not: they come from installed sibling
        distributions (``omnibase_core`` and friends) and resolve into
        ``.venv/lib/.../site-packages``, which is under the repository root but
        is not the repository. An earlier draft filtered on the root and picked
        an ``omnibase_core`` contract, which correctly failed to resolve -- the
        refusal was right and the fixture was wrong.
        """
        src_root = _REPO_ROOT / "src"
        in_repo = [
            c
            for c in discovered.contracts
            if src_root in Path(c.contract_path).resolve().parents
        ]
        if not in_repo:
            pytest.skip(
                "every discovered contract resolves outside this repository's "
                "src (installed siblings only), so there is nothing to resolve "
                "back to here"
            )
        entry = next(
            e for e in build_node_inventory(discovered) if e.name == in_repo[0].name
        )

        resolved = resolve_entry_to_contract_path(entry, [src_root])

        assert contract_content_hash(resolved) == entry.contract_content_hash
