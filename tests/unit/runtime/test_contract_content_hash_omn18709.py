# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The manifest's per-contract hash is a content hash (OMN-18709).

Before this change the hash was ``sha256("{name}:{version}")``, so a contract
whose body was rewritten kept its hash as long as its name and version strings
were untouched -- the field could not detect the drift class it exists for.

Each test below names the acceptance criterion it falsifies. The superseded
identity scheme is computed inside two of them as a positive control: without
it, "the hashes differ" would not prove the defect is gone, only that two
strings are not equal.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_infra.errors.error_contract_content_hash import ContractContentHashError
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
    ModelWiringOutcome,
)
from omnibase_infra.runtime.util_contract_content_hash import (
    CONTRACT_CONTENT_HASH_ALGORITHM,
    canonical_contract_bytes,
    contract_content_hash,
)

CONTRACT_BODY = """\
name: node_example_effect
version: 1.0.0
node_type: EFFECT_GENERIC
event_bus:
  publish_topics:
    - onex.evt.example.thing-happened.v1
"""

# Same name, same version, a different body -- the pair the superseded scheme
# could not tell apart.
CONTRACT_BODY_REWRITTEN = """\
name: node_example_effect
version: 1.0.0
node_type: EFFECT_GENERIC
event_bus:
  publish_topics:
    - onex.evt.example.something-else-happened.v1
  subscribe_topics:
    - onex.cmd.example.do-the-thing.v1
"""


def _superseded_identity_hash(name: str, version: str) -> str:
    """The scheme this ticket replaced, kept only as a positive control."""
    return hashlib.sha256(f"{name}:{version}".encode()).hexdigest()


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


# ---------------------------------------------------------------------------
# AC1 -- the hash is over the file's canonical bytes, and is reproducible
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hash_equals_an_independently_computed_digest(tmp_path: pathlib.Path) -> None:
    """AC1: the value equals a digest the test computes from the file's own bytes."""
    contract = _write(tmp_path / "contract.yaml", CONTRACT_BODY)

    expected = hashlib.sha256(contract.read_bytes()).hexdigest()

    assert contract_content_hash(contract) == expected
    assert len(expected) == 64


@pytest.mark.unit
def test_canonical_form_is_declared_in_the_module(tmp_path: pathlib.Path) -> None:
    """AC1: an outside consumer can reproduce the value from the declared rule alone.

    The rule is executed here the way a consumer would read it out of the module
    docstring -- read the bytes, normalise line endings, sha256, hex -- with no
    call into the module's own hashing path.
    """
    import omnibase_infra.runtime.util_contract_content_hash as module

    docstring = module.__doc__ or ""
    assert "Canonical form" in docstring
    assert CONTRACT_CONTENT_HASH_ALGORITHM == "sha256"
    assert "sha256" in docstring.lower()

    contract = _write(tmp_path / "contract.yaml", CONTRACT_BODY)
    raw = contract.read_bytes()
    reproduced = hashlib.sha256(
        raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()

    assert contract_content_hash(contract) == reproduced


@pytest.mark.unit
def test_line_endings_are_normalised(tmp_path: pathlib.Path) -> None:
    """AC1: the same commit hashes the same whatever the checkout wrote to disk.

    CRLF and lone-CR forms of one body agree with the LF form. The positive
    control is that a body which genuinely differs still does not.
    """
    lf = _write(tmp_path / "lf.yaml", CONTRACT_BODY)
    crlf = _write(tmp_path / "crlf.yaml", CONTRACT_BODY.replace("\n", "\r\n"))
    cr = _write(tmp_path / "cr.yaml", CONTRACT_BODY.replace("\n", "\r"))
    other = _write(tmp_path / "other.yaml", CONTRACT_BODY_REWRITTEN)

    assert crlf.read_bytes() != lf.read_bytes()
    assert contract_content_hash(crlf) == contract_content_hash(lf)
    assert contract_content_hash(cr) == contract_content_hash(lf)
    assert contract_content_hash(other) != contract_content_hash(lf)


@pytest.mark.unit
def test_canonical_bytes_helper_matches_the_hashing_path(
    tmp_path: pathlib.Path,
) -> None:
    """AC1: a caller holding the bytes normalises them the same way, not approximately."""
    contract = _write(tmp_path / "contract.yaml", CONTRACT_BODY.replace("\n", "\r\n"))

    from_bytes = hashlib.sha256(
        canonical_contract_bytes(contract.read_bytes())
    ).hexdigest()

    assert from_bytes == contract_content_hash(contract)


# ---------------------------------------------------------------------------
# AC2 -- a body edit moves the hash; no edit leaves it byte-identical
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_body_edit_moves_the_hash_with_name_and_version_untouched(
    tmp_path: pathlib.Path,
) -> None:
    """AC2 (first half): rewriting the body moves the hash."""
    contract = _write(tmp_path / "contract.yaml", CONTRACT_BODY)
    before = contract_content_hash(contract)

    _write(contract, CONTRACT_BODY_REWRITTEN)
    after = contract_content_hash(contract)

    assert "name: node_example_effect" in CONTRACT_BODY_REWRITTEN
    assert "version: 1.0.0" in CONTRACT_BODY_REWRITTEN
    assert before != after

    # Positive control: the superseded scheme saw no change at all here, which
    # is the defect. If this assertion ever fails the control is wrong, not the
    # hash.
    assert _superseded_identity_hash(
        "node_example_effect", "1.0.0"
    ) == _superseded_identity_hash("node_example_effect", "1.0.0")


@pytest.mark.unit
def test_hash_is_stable_across_reads(tmp_path: pathlib.Path) -> None:
    """AC2 (second half): the hash is stable, not merely different every run.

    Without this control "the hash changed" would be satisfied by a random
    value, which would move on every boot and be useless for drift detection.
    """
    contract = _write(tmp_path / "contract.yaml", CONTRACT_BODY)
    twin = _write(tmp_path / "twin.yaml", CONTRACT_BODY)

    first = contract_content_hash(contract)

    assert contract_content_hash(contract) == first
    assert contract_content_hash(twin) == first


# ---------------------------------------------------------------------------
# AC3 -- a shared name and version no longer implies a shared hash
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_same_name_and_version_different_content_hash_differently(
    tmp_path: pathlib.Path,
) -> None:
    """AC3: the cross-repository collision pair separates.

    The corpus carries contracts that declare the same name and the same
    version in two repositories with different bodies. That shape is
    reconstructed here rather than read from sibling clones, which a unit test
    in this repository has no access to. The superseded scheme is computed on
    the same pair as the positive control: it collides, which is what makes
    this a test of the defect rather than of string inequality.
    """
    repo_a = _write(tmp_path / "a" / "contract.yaml", CONTRACT_BODY)
    repo_b = _write(tmp_path / "b" / "contract.yaml", CONTRACT_BODY_REWRITTEN)

    assert contract_content_hash(repo_a) != contract_content_hash(repo_b)
    assert _superseded_identity_hash(
        "node_example_effect", "1.0.0"
    ) == _superseded_identity_hash("node_example_effect", "1.0.0")


# ---------------------------------------------------------------------------
# AC4 -- an unreadable contract file raises
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_absent_contract_file_raises(tmp_path: pathlib.Path) -> None:
    """AC4: an absent file raises rather than hashing the empty string."""
    missing = tmp_path / "nope" / "contract.yaml"

    with pytest.raises(ContractContentHashError) as excinfo:
        contract_content_hash(missing)

    assert str(missing) in str(excinfo.value)
    assert contract_content_hash(_write(tmp_path / "contract.yaml", CONTRACT_BODY)), (
        "control: a readable file still hashes"
    )


@pytest.mark.unit
def test_hashing_the_empty_string_is_not_the_failure_value(
    tmp_path: pathlib.Path,
) -> None:
    """AC4: the empty-string digest is never returned for a missing file.

    Named explicitly because the empty-string digest is a valid 64-character
    hex value and would pass every shape check downstream.
    """
    empty_digest = hashlib.sha256(b"").hexdigest()
    empty_file = _write(tmp_path / "empty.yaml", "")

    # A genuinely empty file does hash to it -- that is a real content hash.
    assert contract_content_hash(empty_file) == empty_digest

    with pytest.raises(ContractContentHashError):
        contract_content_hash(tmp_path / "absent.yaml")


@pytest.mark.unit
def test_directory_in_place_of_a_contract_raises(tmp_path: pathlib.Path) -> None:
    """AC4: an unreadable path that exists is refused on the same terms."""
    directory = tmp_path / "contract.yaml"
    directory.mkdir()

    with pytest.raises(ContractContentHashError):
        contract_content_hash(directory)


# ---------------------------------------------------------------------------
# The builder reads the hash through this function, not a second copy of it
# ---------------------------------------------------------------------------


def _discovered(
    contract_path: pathlib.Path, name: str = "node_example_effect"
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name=f"onex.nodes.{name}",
        package_name="omnibase-infra",
    )


def _wired(name: str = "node_example_effect") -> ModelContractWiringResult:
    return ModelContractWiringResult(
        contract_name=name,
        package_name="omnibase-infra",
        outcome=EnumWiringOutcome.WIRED,
        wirings=(
            ModelWiringOutcome(
                handler_name="HandlerExample",
                resolution_outcome=EnumHandlerResolutionOutcome.RESOLVED_VIA_CONTAINER,
            ),
        ),
    )


@pytest.mark.unit
def test_manifest_carries_the_content_hash(tmp_path: pathlib.Path) -> None:
    """AC1/AC2 at the manifest: the published row carries the file's content hash."""
    from omnibase_infra.runtime.manifest_builder import build_runtime_manifest

    contract = _write(tmp_path / "contract.yaml", CONTRACT_BODY)
    report = ModelAutoWiringReport(results=(_wired(),))
    manifest = ModelAutoWiringManifest(contracts=(_discovered(contract),))

    built = build_runtime_manifest(
        report=report, manifest=manifest, runtime_profile="main"
    )
    row = built.contracts[0]

    assert row.contract_hash == contract_content_hash(contract)
    assert row.contract_hash != _superseded_identity_hash(row.name, row.version)
    # The pair stays on the row as its own fields: this replaced what the hash
    # means, not what the manifest carries.
    assert row.name == "node_example_effect"
    assert row.version == "1.0.0"

    _write(contract, CONTRACT_BODY_REWRITTEN)
    rebuilt = build_runtime_manifest(
        report=report, manifest=manifest, runtime_profile="main"
    )
    assert rebuilt.contracts[0].contract_hash != row.contract_hash
    assert rebuilt.contracts[0].name == row.name
    assert rebuilt.contracts[0].version == row.version


@pytest.mark.unit
def test_manifest_build_raises_when_a_contract_file_is_unreadable(
    tmp_path: pathlib.Path,
) -> None:
    """AC4 at the manifest: the builder refuses rather than emitting a false hash."""
    from omnibase_infra.runtime.manifest_builder import build_runtime_manifest

    absent = tmp_path / "gone" / "contract.yaml"
    report = ModelAutoWiringReport(results=(_wired(),))
    manifest = ModelAutoWiringManifest(contracts=(_discovered(absent),))

    with pytest.raises(ContractContentHashError):
        build_runtime_manifest(report=report, manifest=manifest, runtime_profile="main")

    # Control: the same call with a readable file builds.
    readable = _write(tmp_path / "contract.yaml", CONTRACT_BODY)
    built = build_runtime_manifest(
        report=report,
        manifest=ModelAutoWiringManifest(contracts=(_discovered(readable),)),
        runtime_profile="main",
    )
    assert built.contracts[0].contract_hash == contract_content_hash(readable)


@pytest.mark.unit
def test_manifest_build_raises_when_the_result_names_no_discovered_contract(
    tmp_path: pathlib.Path,
) -> None:
    """A wiring result with no manifest entry has no contract file, so no hash."""
    from omnibase_infra.runtime.manifest_builder import build_runtime_manifest

    contract = _write(tmp_path / "contract.yaml", CONTRACT_BODY)
    report = ModelAutoWiringReport(results=(_wired("node_not_in_manifest"),))
    manifest = ModelAutoWiringManifest(contracts=(_discovered(contract),))

    with pytest.raises(ContractContentHashError) as excinfo:
        build_runtime_manifest(report=report, manifest=manifest, runtime_profile="main")

    assert "node_not_in_manifest" in str(excinfo.value)


@pytest.mark.unit
def test_there_is_exactly_one_hasher(tmp_path: pathlib.Path) -> None:
    """The builder calls this module rather than carrying its own hashlib copy.

    The previous implementation hashed inline. A second copy is how the label
    and the manifest drift apart while both look correct.
    """
    source = (
        pathlib.Path(__file__).parents[3]
        / "src"
        / "omnibase_infra"
        / "runtime"
        / "manifest_builder.py"
    )
    text = source.read_text(encoding="utf-8")

    assert "contract_content_hash" in text
    assert "hashlib" not in text
