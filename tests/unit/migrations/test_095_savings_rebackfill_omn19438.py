# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The savings re-backfill ships as a new, checksum-bound migration."""

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
RELATIVE = (
    "nodes/node_projection_savings/"
    "095_savings_estimates_house_tenant_uuid_rebackfill.sql"
)
FORWARD = ROOT / "docker/migrations/forward"


def test_095_is_the_vendored_forward_only_manifest_entry() -> None:
    """Changing the SQL, omitting its ledger row or misclassifying it fails."""
    digest = hashlib.sha256((FORWARD / RELATIVE).read_bytes()).hexdigest()
    assert digest == "778e9b2a9aa566ebc4a2479079c096992d8c1de0be5949aab777bbf6aa921663"
    rows = [
        line.split("\t")
        for line in (FORWARD / "_ledger/application-migrations.tsv")
        .read_text()
        .splitlines()
        if line.split("\t")[0] == RELATIVE
    ]
    assert rows == [
        [
            RELATIVE,
            "node:node_projection_savings",
            "node:node_projection_savings",
            "tenant",
            f"node:node_projection_savings:{Path(RELATIVE).name}",
            digest,
        ]
    ]
    classes = yaml.safe_load((ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/{RELATIVE}"] == "forward-only"
