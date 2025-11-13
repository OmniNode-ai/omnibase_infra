#!/bin/bash
# setup-kafka-hostname.sh - Configure Kafka/Redpanda hostname resolution
#
# This script adds the Docker container hostname to /etc/hosts to enable
# Kafka/Redpanda broker discovery from the host machine.
#
# Why is this needed?
# Kafka/Redpanda uses a two-step broker discovery protocol:
# 1. Client connects to bootstrap server (localhost:29092)
# 2. Broker returns metadata with advertised address (omninode-bridge-redpanda:9092)
# 3. Client must resolve this hostname to connect
#
# This is Kafka-specific behavior. Other services (PostgreSQL, HTTP) don't require this.

set -e

HOSTNAME="omninode-bridge-redpanda"
IP="127.0.0.1"

echo "🔍 Checking if hostname configuration is needed..."

# Check if hostname already exists in /etc/hosts
if grep -q "$HOSTNAME" /etc/hosts; then
    echo "✅ Hostname '$HOSTNAME' already configured in /etc/hosts"
    grep "$HOSTNAME" /etc/hosts
    exit 0
fi

echo "📝 Adding hostname '$HOSTNAME' to /etc/hosts..."
echo ""
echo "This requires sudo permissions to modify /etc/hosts"
echo "Command to be executed:"
echo "  echo '$IP $HOSTNAME' | sudo tee -a /etc/hosts"
echo ""

# Add hostname to /etc/hosts
if echo "$IP $HOSTNAME" | sudo tee -a /etc/hosts > /dev/null; then
    echo "✅ Successfully added hostname to /etc/hosts"
    echo ""
    echo "Verification:"
    grep "$HOSTNAME" /etc/hosts
    echo ""
    echo "🎉 Kafka/Redpanda hostname configuration complete!"
    echo ""
    echo "You can now connect to Kafka/Redpanda from Python clients:"
    echo "  aiokafka: AIOKafkaProducer(bootstrap_servers='localhost:29092')"
    echo "  confluent-kafka: Producer({'bootstrap.servers': 'localhost:29092'})"
    echo "  kcat: kcat -b localhost:29092 -L"
else
    echo "❌ Failed to add hostname to /etc/hosts"
    echo ""
    echo "Please run this command manually:"
    echo "  echo '$IP $HOSTNAME' | sudo tee -a /etc/hosts"
    exit 1
fi
