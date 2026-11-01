# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot handover preserves durable store decisions across a role fold."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.idempotency.store_sqlite import StoreIdempotencySqlite
from omnibase_infra.runtime.entrypoint_preflight import run_preflight

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_folded_boot_keeps_completed_work_and_new_writes(tmp_path: Path) -> None:
    effects_root = tmp_path / "effects"
    main_root = tmp_path / "main"
    filename = "delegation_claims.sqlite"
    effects_path = effects_root / "delegation" / filename
    main_path = main_root / "delegation" / filename
    completed, new_work = uuid4(), uuid4()
    domain = "fold-handover"
    effects_store = StoreIdempotencySqlite(effects_path)
    await effects_store.start()
    try:
        assert await effects_store.check_and_record(completed, domain)
    finally:
        await effects_store.close()

    main_root.mkdir()
    unrelated = main_root / "main-owned-state"
    unrelated.write_text("preserve")
    env = {
        "ONEX_RUNTIME_FOLDED_PROFILES": "effects",
        "ONEX_STATE_DIR": str(main_root),
        "ONEX_FOLDED_EFFECTS_STATE_ROOT": str(effects_root),
    }
    assert run_preflight(env) == 0
    main_store = StoreIdempotencySqlite(main_path)
    await main_store.start()
    try:
        assert await main_store.is_processed(completed, domain)
        assert not await main_store.check_and_record(completed, domain)
        assert await main_store.check_and_record(new_work, domain)
    finally:
        await main_store.close()

    assert main_path.samefile(effects_path)
    assert unrelated.read_text() == "preserve"
    assert run_preflight(env) == 0
    restarted = StoreIdempotencySqlite(main_path)
    await restarted.start()
    try:
        assert not await restarted.check_and_record(completed, domain)
        assert not await restarted.check_and_record(new_work, domain)
    finally:
        await restarted.close()

    await effects_store.start()
    try:
        assert await effects_store.is_processed(new_work, domain)
    finally:
        await effects_store.close()
