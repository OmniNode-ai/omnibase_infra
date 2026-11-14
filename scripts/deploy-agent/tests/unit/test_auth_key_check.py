# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the deploy-agent key check value (OMN-24397)."""

from __future__ import annotations

import hashlib
import hmac
import os
from unittest.mock import patch

import pytest
from deploy_agent.auth import KEY_CHECK_LABEL, key_check_value, loaded_key_check_value

pytestmark = pytest.mark.unit

_GOLDEN_SECRET = "fixture-key"
_GOLDEN_VALUE = "6dce9bc99db04417"


def test_golden_vector() -> None:
    assert key_check_value(_GOLDEN_SECRET) == _GOLDEN_VALUE


def test_value_is_16_lowercase_hex() -> None:
    value = key_check_value("some-secret")
    assert len(value) == 16
    assert value == value.lower()
    int(value, 16)


def test_different_keys_give_different_values() -> None:
    assert key_check_value("secret-one") != key_check_value("secret-two")


def test_value_does_not_contain_the_secret_or_a_prefix_of_it() -> None:
    secret = "test-secret-abc123"
    value = key_check_value(secret)
    assert secret not in value
    assert secret[:8] not in value
    assert secret[:4] not in value


def test_failing_control_label_change_changes_the_value() -> None:
    secret = "fixture-key"
    other = hmac.new(secret.encode(), b"other-label"[:16], hashlib.sha256).hexdigest()[
        :16
    ]
    assert other != key_check_value(secret)
    assert other != _GOLDEN_VALUE
    assert KEY_CHECK_LABEL == b"deploy-agent-key-check-v1"


def test_loaded_value_none_when_unset() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DEPLOY_AGENT_HMAC_SECRET", None)
        assert loaded_key_check_value() is None


def test_loaded_value_matches_key_check_value_when_set() -> None:
    secret = "fixture-key"
    with patch.dict(os.environ, {"DEPLOY_AGENT_HMAC_SECRET": secret}):
        assert loaded_key_check_value() == key_check_value(secret)
        assert loaded_key_check_value() == _GOLDEN_VALUE
