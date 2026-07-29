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
COPY docker/legacy-rds-fixture/fixture-manifest.json /opt/omn15422/fixture-manifest.json
COPY docker/legacy-rds-fixture/prove.sh /opt/omn15422/prove.sh

ENTRYPOINT ["sh", "/opt/omn15422/prove.sh"]
