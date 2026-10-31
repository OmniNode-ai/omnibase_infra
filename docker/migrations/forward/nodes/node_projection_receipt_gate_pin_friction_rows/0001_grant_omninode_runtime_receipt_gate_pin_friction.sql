-- OMN-20134: omninode_runtime grants for omninode_internal.receipt_gate_pin_friction.
-- Node: node_projection_receipt_gate_pin_friction
--
-- The writer reads the stored row, folds the observation and upserts it, so it
-- needs SELECT, INSERT and UPDATE; it never deletes (a pin pull request that
-- happened is not removed). projection_cursor is BIGSERIAL, whose standalone
-- sequence carries its own ACL: the table grant does not reach it, so the
-- sequence grant below is a separate, necessary half (OMN-17447's class).
-- GRANT is idempotent; re-running is a no-op.

GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;

GRANT SELECT, INSERT, UPDATE
    ON omninode_internal.receipt_gate_pin_friction
    TO omninode_runtime;

GRANT USAGE
    ON SEQUENCE omninode_internal.receipt_gate_pin_friction_projection_cursor_seq
    TO omninode_runtime;

SELECT 1 / count(*) AS receipt_gate_pin_friction_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'receipt_gate_pin_friction'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS receipt_gate_pin_friction_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'receipt_gate_pin_friction'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS receipt_gate_pin_friction_update_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'omninode_internal'
  AND table_name = 'receipt_gate_pin_friction'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'UPDATE';

SELECT 1 / count(*) AS receipt_gate_pin_friction_cursor_sequence_usage_assertion
WHERE has_sequence_privilege(
          'omninode_runtime',
          pg_get_serial_sequence(
              'omninode_internal.receipt_gate_pin_friction', 'projection_cursor'
          ),
          'USAGE'
      );
