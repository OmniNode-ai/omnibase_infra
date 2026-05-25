-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Rollback: 083_create_log_entries

\connect omnidash_analytics

DROP TABLE IF EXISTS log_entries;
