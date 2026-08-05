# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

FROM postgres:16-alpine AS fresh

COPY docker/legacy-rds-fixture/cluster-seed.sql /docker-entrypoint-initdb.d/00-cluster-seed.sql

FROM postgres:16-alpine AS legacy

COPY docker/legacy-rds-fixture/cluster-seed.sql /docker-entrypoint-initdb.d/00-cluster-seed.sql
COPY docker/legacy-rds-fixture/legacy-seed.sql /docker-entrypoint-initdb.d/10-legacy-seed.sql

FROM postgres:16-alpine AS proof

COPY scripts/run-forward-migrations.sh /opt/omn15422/run-forward-migrations.sh
COPY docker/migrations/ /migrations/
COPY docker/legacy-rds-fixture/ledger-control/ /opt/omn15422/ledger-control/
COPY docker/migrations/forward/fenced-node-migrations.yaml /opt/omn15422/ledger-control/forward/fenced-node-migrations.yaml
COPY docker/migrations/forward/_ledger/bootstrap.sql /opt/omn15422/ledger-control/forward/_ledger/bootstrap.sql
COPY docker/legacy-rds-fixture/fixture-manifest.json /opt/omn15422/fixture-manifest.json
COPY docker/legacy-rds-fixture/prove.sh /opt/omn15422/prove.sh
COPY docker/legacy-rds-fixture/cutover-proof/prove.sh /opt/omn15422/cutover-proof/prove.sh
COPY src/omnibase_infra/migration/cutover/sql/bootstrap.sql /opt/omn15422/cutover-proof/bootstrap.sql

ENTRYPOINT ["sh", "/opt/omn15422/prove.sh"]
