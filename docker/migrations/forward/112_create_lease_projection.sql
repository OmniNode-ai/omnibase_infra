-- OMN-21424: generic lease projection, one row per (kind, resource).
-- The writer upserts on (kind, resource) and never lowers fence: fence is the
-- etcd revision of the grant and rises with every grant of a resource.
-- Schema-qualified: scripts/ci/check_application_database_sql.py rejects an unqualified target.

CREATE TABLE IF NOT EXISTS public.lease_projection (
    kind        TEXT NOT NULL,
    resource    TEXT NOT NULL,
    lease_id    UUID NOT NULL,
    fence       BIGINT NOT NULL CHECK (fence >= 1),
    holder      TEXT NOT NULL,
    state       TEXT NOT NULL CHECK (state IN ('held', 'released', 'expired')),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kind, resource)
);
