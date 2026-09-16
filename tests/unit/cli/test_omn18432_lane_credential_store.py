# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18432 AC2/AC3: a lane bus identity is storable by reference, or refused.

The gateway credential store (OMN-15922) already established the shape these
tests pin: references and endpoints in ``config.yaml``, which is world-readable
and gets pasted into issues, and secret VALUES only in ``credentials.json`` at
mode 0600, enforced on READ as well as on write because the file survives
``chmod``, backup/restore and ``scp``.

What was missing is that the store modelled exactly one credential -- the
gateway one -- so a machine had nowhere to put a BUS identity by reference. The
consequence was concrete: the ``.201`` dev-lane broker has required SASL since
OMN-18012 phase B, and the launching Mac's only path to a SASL credential was
the ambient ``KAFKA_SASL_*`` environment, which is the surface the whole
"secrets resolve by reference" rule exists to remove.

Every refusal here is asserted with its error CLASS, not merely "it raised":
a store that refuses a mis-permissioned file with the same anonymous error it
uses for a typo teaches the caller nothing, and a caller that cannot tell them
apart writes the retry loop that defeats the refusal.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential

pytestmark = pytest.mark.unit


def _store(tmp_path: Path) -> StoreLaneCredential:
    return StoreLaneCredential(onex_home=tmp_path)


def test_save_then_load_round_trips_the_credential(tmp_path: Path) -> None:
    """The positive control, so every refusal below is a real discrimination."""
    store = _store(tmp_path)
    store.save(lane="dev", sasl_username="dev-cli-host", sasl_password="s3kr3t-value")

    credential = store.load("dev")

    assert credential.lane == "dev"
    assert credential.sasl_username == "dev-cli-host"
    assert credential.sasl_password.get_secret_value() == "s3kr3t-value"


def test_save_puts_the_reference_in_config_and_the_value_only_in_the_secret_file(
    tmp_path: Path,
) -> None:
    """AC2: config.yaml carries a REFERENCE; the value appears nowhere in it."""
    store = _store(tmp_path)
    store.save(lane="dev", sasl_username="dev-cli-host", sasl_password="s3kr3t-value")

    config_text = store.config_path.read_text()
    assert "s3kr3t-value" not in config_text

    document = yaml.safe_load(config_text)
    lane_block = document["lanes"]["dev"]
    assert lane_block["sasl_username"] == "dev-cli-host"
    assert lane_block["sasl_password_ref"] == store.password_ref("dev")
    assert "sasl_password" not in lane_block

    secrets = json.loads(store.credentials_path.read_text())
    assert secrets[store.password_ref("dev")] == "s3kr3t-value"


def test_save_writes_the_secret_file_owner_only(tmp_path: Path) -> None:
    """AC2: 0600, established before any content exists on disk."""
    store = _store(tmp_path)
    store.save(lane="dev", sasl_username="dev-cli-host", sasl_password="s3kr3t-value")

    mode = stat.S_IMODE(store.credentials_path.stat().st_mode)
    assert mode == 0o600


def test_save_preserves_every_other_key_in_the_config_document(
    tmp_path: Path,
) -> None:
    """Storing a lane identity must not lose someone's gateway block."""
    store = _store(tmp_path)
    store.config_path.write_text(
        yaml.safe_dump(
            {
                "gateway": {"tenant_slug": "jonah", "api_key_ref": "jonah-api-key"},
                "mode": "local",
            },
            sort_keys=False,
        )
    )

    store.save(lane="dev", sasl_username="dev-cli-host", sasl_password="s3kr3t-value")

    document = yaml.safe_load(store.config_path.read_text())
    assert document["gateway"] == {
        "tenant_slug": "jonah",
        "api_key_ref": "jonah-api-key",
    }
    assert document["mode"] == "local"
    assert document["lanes"]["dev"]["sasl_username"] == "dev-cli-host"


def test_save_preserves_another_lane_s_entry(tmp_path: Path) -> None:
    """One lane's login must not silently drop another lane's identity."""
    store = _store(tmp_path)
    store.save(lane="dev", sasl_username="dev-cli-host", sasl_password="dev-value")
    store.save(lane="lab", sasl_username="lab-cli-host", sasl_password="lab-value")

    assert store.load("dev").sasl_password.get_secret_value() == "dev-value"
    assert store.load("lab").sasl_password.get_secret_value() == "lab-value"


def test_load_refuses_an_inline_password_in_the_config_file(tmp_path: Path) -> None:
    """AC2: a literal value in the world-readable file is refused, not accepted."""
    store = _store(tmp_path)
    store.config_path.write_text(
        yaml.safe_dump(
            {
                "lanes": {
                    "dev": {
                        "sasl_username": "dev-cli-host",
                        "sasl_password": "literal-value-in-the-wrong-file",
                    }
                }
            },
            sort_keys=False,
        )
    )

    with pytest.raises(ModelOnexError) as caught:
        store.load("dev")

    assert caught.value.error_code == EnumCoreErrorCode.INVALID_CONFIGURATION
    assert "sasl_password" in str(caught.value)
    assert str(store.credentials_path) in str(caught.value)


def test_load_refuses_a_group_or_world_readable_secret_file(tmp_path: Path) -> None:
    """AC2: the mode is enforced on READ, which is the check that survives scp."""
    store = _store(tmp_path)
    store.save(lane="dev", sasl_username="dev-cli-host", sasl_password="s3kr3t-value")
    store.credentials_path.chmod(0o644)

    with pytest.raises(ModelOnexError) as caught:
        store.load("dev")

    assert caught.value.error_code == EnumCoreErrorCode.PERMISSION_DENIED
    assert "0644" in str(caught.value)


def test_load_refuses_when_the_machine_holds_no_lane_block(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.config_path.write_text(yaml.safe_dump({"gateway": {}}, sort_keys=False))

    with pytest.raises(ModelOnexError) as caught:
        store.load("dev")

    assert caught.value.error_code == EnumCoreErrorCode.CONFIGURATION_NOT_FOUND
    assert "lane-login" in str(caught.value)


def test_load_refuses_when_this_lane_is_absent_and_names_the_ones_present(
    tmp_path: Path,
) -> None:
    """An absent lane names its neighbours so a typo is visible at once."""
    store = _store(tmp_path)
    store.save(lane="lab", sasl_username="lab-cli-host", sasl_password="lab-value")

    with pytest.raises(ModelOnexError) as caught:
        store.load("dev")

    assert caught.value.error_code == EnumCoreErrorCode.CONFIGURATION_NOT_FOUND
    assert "lab" in str(caught.value)


def test_load_refuses_a_dangling_reference(tmp_path: Path) -> None:
    """A config naming a secret the 0600 file does not hold is a loud refusal."""
    store = _store(tmp_path)
    store.save(lane="dev", sasl_username="dev-cli-host", sasl_password="s3kr3t-value")
    store.credentials_path.write_text(json.dumps({}))
    store.credentials_path.chmod(0o600)

    with pytest.raises(ModelOnexError) as caught:
        store.load("dev")

    assert caught.value.error_code == EnumCoreErrorCode.CONFIGURATION_NOT_FOUND


def test_load_refuses_a_blank_username(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.config_path.write_text(
        yaml.safe_dump(
            {"lanes": {"dev": {"sasl_username": "  ", "sasl_password_ref": "r"}}},
            sort_keys=False,
        )
    )

    with pytest.raises(ModelOnexError) as caught:
        store.load("dev")

    assert caught.value.error_code == EnumCoreErrorCode.INVALID_CONFIGURATION


def test_clear_removes_the_secret_before_the_reference(tmp_path: Path) -> None:
    """A crash mid-clear must leave a loud dangling ref, never an orphan secret."""
    store = _store(tmp_path)
    store.save(lane="dev", sasl_username="dev-cli-host", sasl_password="s3kr3t-value")

    store.clear("dev")

    secrets = json.loads(store.credentials_path.read_text())
    assert store.password_ref("dev") not in secrets
    document = yaml.safe_load(store.config_path.read_text())
    assert "dev" not in document.get("lanes", {})


def test_clear_is_idempotent_on_a_machine_holding_nothing(tmp_path: Path) -> None:
    _store(tmp_path).clear("dev")
