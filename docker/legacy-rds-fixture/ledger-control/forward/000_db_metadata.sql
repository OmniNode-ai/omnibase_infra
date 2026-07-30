-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT

CREATE TABLE IF NOT EXISTS public.db_metadata (
  id BOOLEAN PRIMARY KEY,
  migrations_complete BOOLEAN NOT NULL DEFAULT FALSE,
  runner_completed_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO public.db_metadata (id)
VALUES (TRUE)
ON CONFLICT (id) DO NOTHING;
