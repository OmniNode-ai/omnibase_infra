#!/bin/bash
# SSL Certificate Setup for OmniNode Bridge
# This script generates self-signed certificates for development/testing
# For production, replace with proper CA-signed certificates

set -e

echo "Setting up SSL certificates for OmniNode Bridge..."

# Create directory structure
mkdir -p ssl/{ca,postgres,client}

# Certificate configuration
COUNTRY="US"
STATE="CA"
CITY="San Francisco"
ORG="OmniNode"
ORG_UNIT="Development"
COMMON_NAME_CA="OmniNode-CA"
COMMON_NAME_SERVER="omninode-bridge-postgres"
COMMON_NAME_CLIENT="omninode-bridge-client"

# Generate CA private key
echo "Generating CA private key..."
openssl genrsa -out ssl/ca/ca.key 4096
chmod 600 ssl/ca/ca.key

# Generate CA certificate
echo "Generating CA certificate..."
openssl req -new -x509 -days 3650 -key ssl/ca/ca.key -out ssl/ca/ca.crt \
  -subj "/C=${COUNTRY}/ST=${STATE}/L=${CITY}/O=${ORG}/OU=${ORG_UNIT}/CN=${COMMON_NAME_CA}"

# Generate server private key
echo "Generating PostgreSQL server private key..."
openssl genrsa -out ssl/postgres/server.key 2048
chmod 600 ssl/postgres/server.key

# Generate server certificate signing request
echo "Generating PostgreSQL server CSR..."
openssl req -new -key ssl/postgres/server.key -out ssl/postgres/server.csr \
  -subj "/C=${COUNTRY}/ST=${STATE}/L=${CITY}/O=${ORG}/OU=${ORG_UNIT}/CN=${COMMON_NAME_SERVER}"

# Create server certificate extensions
cat > ssl/postgres/server_ext.conf << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = omninode-bridge-postgres
DNS.2 = postgres
DNS.3 = localhost
IP.1 = 127.0.0.1
IP.2 = 172.20.0.10
EOF

# Generate server certificate
echo "Generating PostgreSQL server certificate..."
openssl x509 -req -in ssl/postgres/server.csr -CA ssl/ca/ca.crt -CAkey ssl/ca/ca.key \
  -CAcreateserial -out ssl/postgres/server.crt -days 365 \
  -extensions v3_req -extfile ssl/postgres/server_ext.conf

# Generate client private key
echo "Generating client private key..."
openssl genrsa -out ssl/client/client.key 2048
chmod 600 ssl/client/client.key

# Generate client certificate signing request
echo "Generating client CSR..."
openssl req -new -key ssl/client/client.key -out ssl/client/client.csr \
  -subj "/C=${COUNTRY}/ST=${STATE}/L=${CITY}/O=${ORG}/OU=${ORG_UNIT}/CN=${COMMON_NAME_CLIENT}"

# Generate client certificate
echo "Generating client certificate..."
openssl x509 -req -in ssl/client/client.csr -CA ssl/ca/ca.crt -CAkey ssl/ca/ca.key \
  -CAcreateserial -out ssl/client/client.crt -days 365

# Copy CA certificate to client directory for verification
cp ssl/ca/ca.crt ssl/client/ca.crt
cp ssl/ca/ca.crt ssl/postgres/ca.crt

# Set proper permissions
chmod 644 ssl/ca/ca.crt ssl/postgres/server.crt ssl/client/client.crt
chmod 600 ssl/postgres/server.key ssl/client/client.key
chmod 600 ssl/ca/ca.key

# Clean up CSR files
rm -f ssl/postgres/server.csr ssl/client/client.csr ssl/postgres/server_ext.conf

echo "SSL certificates generated successfully!"
echo ""
echo "Certificate files created:"
echo "  CA Certificate: ssl/ca/ca.crt"
echo "  Server Certificate: ssl/postgres/server.crt"
echo "  Server Private Key: ssl/postgres/server.key"
echo "  Client Certificate: ssl/client/client.crt"
echo "  Client Private Key: ssl/client/client.key"
echo ""
echo "To verify the certificates:"
echo "  openssl x509 -in ssl/postgres/server.crt -text -noout"
echo "  openssl x509 -in ssl/client/client.crt -text -noout"
echo ""
echo "IMPORTANT: These are self-signed certificates for development only!"
echo "For production, use certificates signed by a trusted CA."
