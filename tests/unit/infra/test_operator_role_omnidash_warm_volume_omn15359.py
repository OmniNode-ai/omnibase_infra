# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15359: an old operator volume has role_omnidash with NOLOGIN.

Failure observed read only on .201 on 2026-09-29: the role exists but cannot
authenticate. Adding ROLE_OMNIDASH_PASSWORD to Compose without the warm-volume
credential seam would start onex-api with a DSN that fails on every connection.
The cold init script cannot repair a volume that already exists.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = ROOT / "docker/migrations/forward/000_create_multiple_databases.sh"
RUNNER = ROOT / "scripts/run-forward-migrations.sh"
COMPOSE = ROOT / "docker/docker-compose.lakshman.yml"


def test_operator_role_omnidash_has_a_warm_volume_login_path() -> None:
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    compose = COMPOSE.read_text(encoding="utf-8")
    begin = runner.index("# ---- BEGIN service-role database access seam")
    end = runner.index("# ---- END service-role database access seam", begin)
    warm_seam = runner[begin:end]

    assert '"omnidash_analytics:role_omnidash:ROLE_OMNIDASH_PASSWORD"' in bootstrap, (
        "the cold-volume owner and credential contract changed"
    )
    assert "ROLE_OMNIDASH_PASSWORD:" in compose, (
        "the operator forward-migration must receive the credential"
    )
    assert '"role_omnidash:ROLE_OMNIDASH_PASSWORD:omnidash_analytics"' in warm_seam, (
        "a warm operator volume leaves role_omnidash NOLOGIN, blocking onex-api"
    )
