# Kafka Advertised Listeners Fix for Remote Deployment

## Issue Summary

**Problem**: Redpanda (Kafka) broker advertises `localhost:29092` instead of the remote host address `192.168.86.200:29102`, blocking external clients from connecting.

**Impact**:
- Local clients on 192.168.86.200 can connect via internal address
- External clients (local development machines, other services) cannot connect
- Integration tests from external systems fail

**Root Cause**: The `--advertise-kafka-addr` parameter in `deployment/docker-compose.yml` line 107 uses hardcoded `localhost:29092`.

## Solution

### Option 1: Environment Variable Override (Recommended)

**Why**: Flexible, works for any deployment without changing base compose files.

1. **Update `remote.env`** to include:
   ```bash
   # Kafka Advertised Listeners for Remote Access
   KAFKA_ADVERTISED_HOST=192.168.86.200
   KAFKA_ADVERTISED_PORT=29102
   ```

2. **Create `docker-compose.remote-override.yml`**:
   ```yaml
   version: '3.8'

   services:
     redpanda:
       command:
         - redpanda
         - start
         - --kafka-addr=internal://0.0.0.0:9092,external://0.0.0.0:29092
         - --advertise-kafka-addr=internal://omninode-bridge-redpanda:9092,external://${KAFKA_ADVERTISED_HOST:-localhost}:${KAFKA_ADVERTISED_PORT:-29092}
         - --pandaproxy-addr=internal://0.0.0.0:8082,external://0.0.0.0:8083
         - --advertise-pandaproxy-addr=internal://omninode-bridge-redpanda:8082,external://${KAFKA_ADVERTISED_HOST:-localhost}:8083
         - --schema-registry-addr=internal://0.0.0.0:8081,external://0.0.0.0:8084
         - --rpc-addr=omninode-bridge-redpanda:33145
         - --advertise-rpc-addr=omninode-bridge-redpanda:33145
         - --mode=dev-container
         - --smp=1
         - --default-log-level=${REDPANDA_LOG_LEVEL:-info}
   ```

3. **Deploy with override**:
   ```bash
   docker compose \
     -f deployment/docker-compose.yml \
     -f docker-compose.remote.yml \
     -f docker-compose.remote-override.yml \
     --env-file remote.env \
     up -d redpanda
   ```

### Option 2: Update Base Configuration

**Why**: Simpler but requires changing base compose file.

**Edit `deployment/docker-compose.yml` line 107**:
```yaml
# Before:
- --advertise-kafka-addr=internal://omninode-bridge-redpanda:9092,external://localhost:29092,external://omninode-bridge-redpanda:29092

# After:
- --advertise-kafka-addr=internal://omninode-bridge-redpanda:9092,external://${KAFKA_ADVERTISED_HOST:-localhost}:${KAFKA_ADVERTISED_PORT:-29092}
```

Then add to `.env` (local) or `remote.env` (remote):
```bash
KAFKA_ADVERTISED_HOST=192.168.86.200
KAFKA_ADVERTISED_PORT=29102
```

### Option 3: Script-Based Dynamic Configuration

**Why**: Works with existing deployment scripts.

**Update `setup-remote.sh`** to include Kafka configuration:
```bash
# Configure Kafka advertised listeners
cat >> ~/omninode_bridge/remote.env << 'EOF'

# Kafka Remote Configuration
KAFKA_ADVERTISED_HOST=192.168.86.200
KAFKA_ADVERTISED_PORT=29102
EOF

# Restart Redpanda with new configuration
cd ~/omninode_bridge
docker compose \
  -f deployment/docker-compose.yml \
  -f docker-compose.remote.yml \
  --env-file remote.env \
  up -d redpanda
```

## Verification

### 1. Check Advertised Address

From any machine, check the advertised address:
```bash
docker exec omninode-bridge-redpanda \
  rpk cluster info --brokers localhost:9092
```

**Expected Output**:
```
CLUSTER
=======
redpanda.8c0d7a90-9f68-4df1-8b9e-2e3c4a5b6c7d

BROKERS
=======
ID    HOST               PORT
0*    192.168.86.200    29102  # ← Should show remote IP, not localhost
```

### 2. Test External Connection

From local development machine:
```bash
# Test Kafka connection
poetry run python -c "
from aiokafka import AIOKafkaProducer
import asyncio

async def test():
    producer = AIOKafkaProducer(bootstrap_servers='192.168.86.200:29102')
    await producer.start()
    print('✅ Connected successfully!')
    await producer.stop()

asyncio.run(test())
"
```

### 3. Verify Topic Access

```bash
# List topics from external client
rpk topic list --brokers 192.168.86.200:29102

# Should show all topics without connection errors
```

## Implementation Steps (Recommended: Option 1)

**Step 1: Create Override File**
```bash
cd /Volumes/PRO-G40/Code/omninode_bridge
cat > docker-compose.remote-override.yml << 'EOF'
version: '3.8'

services:
  redpanda:
    command:
      - redpanda
      - start
      - --kafka-addr=internal://0.0.0.0:9092,external://0.0.0.0:29092
      - --advertise-kafka-addr=internal://omninode-bridge-redpanda:9092,external://${KAFKA_ADVERTISED_HOST:-localhost}:${KAFKA_ADVERTISED_PORT:-29092}
      - --pandaproxy-addr=internal://0.0.0.0:8082,external://0.0.0.0:8083
      - --advertise-pandaproxy-addr=internal://omninode-bridge-redpanda:8082,external://${KAFKA_ADVERTISED_HOST:-localhost}:8083
      - --schema-registry-addr=internal://0.0.0.0:8081,external://0.0.0.0:8084
      - --rpc-addr=omninode-bridge-redpanda:33145
      - --advertise-rpc-addr=omninode-bridge-redpanda:33145
      - --mode=dev-container
      - --smp=1
      - --default-log-level=${REDPANDA_LOG_LEVEL:-info}
EOF
```

**Step 2: Update remote.env**
```bash
cat >> remote.env << 'EOF'

# Kafka Advertised Listeners for Remote Access
KAFKA_ADVERTISED_HOST=192.168.86.200
KAFKA_ADVERTISED_PORT=29102
EOF
```

**Step 3: Update Migration Script**

Add to `migrate-to-remote.sh` after file sync:
```bash
echo "📝 Creating Kafka configuration override..."
cat > docker-compose.remote-override.yml << 'EOFOVERRIDE'
version: '3.8'

services:
  redpanda:
    command:
      - redpanda
      - start
      - --kafka-addr=internal://0.0.0.0:9092,external://0.0.0.0:29092
      - --advertise-kafka-addr=internal://omninode-bridge-redpanda:9092,external://${KAFKA_ADVERTISED_HOST:-localhost}:${KAFKA_ADVERTISED_PORT:-29092}
      - --pandaproxy-addr=internal://0.0.0.0:8082,external://0.0.0.0:8083
      - --advertise-pandaproxy-addr=internal://omninode-bridge-redpanda:8082,external://${KAFKA_ADVERTISED_HOST:-localhost}:8083
      - --schema-registry-addr=internal://0.0.0.0:8081,external://0.0.0.0:8084
      - --rpc-addr=omninode-bridge-redpanda:33145
      - --advertise-rpc-addr=omninode-bridge-redpanda:33145
      - --mode=dev-container
      - --smp=1
      - --default-log-level=\${REDPANDA_LOG_LEVEL:-info}
EOFOVERRIDE

echo "🔄 Syncing override to remote..."
scp docker-compose.remote-override.yml ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/
```

**Step 4: Update Setup Script**

Modify `setup-remote.sh` to use the override:
```bash
# In the deployment section, change:
docker compose \
  -f deployment/docker-compose.yml \
  -f docker-compose.remote.yml \
  --env-file remote.env \
  up -d

# To:
docker compose \
  -f deployment/docker-compose.yml \
  -f docker-compose.remote.yml \
  -f docker-compose.remote-override.yml \
  --env-file remote.env \
  up -d
```

**Step 5: Deploy**
```bash
# On remote system (192.168.86.200)
cd ~/omninode_bridge
docker compose \
  -f deployment/docker-compose.yml \
  -f docker-compose.remote.yml \
  -f docker-compose.remote-override.yml \
  --env-file remote.env \
  up -d redpanda
```

**Step 6: Verify**
```bash
# Check advertised address
docker exec omninode-bridge-redpanda rpk cluster info --brokers localhost:9092

# Test from local machine
rpk topic list --brokers 192.168.86.200:29102
```

## Troubleshooting

### Issue: Still seeing localhost

**Cause**: Environment variables not passed to container

**Fix**:
```bash
# Verify environment variables are set
docker compose \
  -f deployment/docker-compose.yml \
  -f docker-compose.remote.yml \
  -f docker-compose.remote-override.yml \
  --env-file remote.env \
  config | grep advertise-kafka-addr

# Should show: external://192.168.86.200:29102
```

### Issue: Connection refused from external

**Cause**: Firewall blocking port 29102

**Fix**:
```bash
# On remote system
sudo ufw allow 29102/tcp
sudo ufw status
```

### Issue: Topic creation fails

**Cause**: Topic manager using wrong broker address

**Fix**: Topics are created using internal address (omninode-bridge-redpanda:9092), which is correct. External advertised address only affects external clients.

## Performance Impact

- **Negligible**: Advertised listeners configuration has no performance impact
- **Latency**: External connections may have slightly higher latency due to network routing
- **Throughput**: No impact on message throughput

## Security Considerations

**Exposed Port**: Port 29102 is now accessible from external networks
- **Recommendation**: Use firewall rules to restrict access to trusted IPs
- **Production**: Use TLS encryption and SASL authentication

**Firewall Configuration**:
```bash
# Allow only from specific development IPs
sudo ufw allow from 192.168.86.0/24 to any port 29102
sudo ufw allow from 10.0.0.0/8 to any port 29102
```

## References

- Redpanda Advertised Listeners: https://docs.redpanda.com/current/manage/kubernetes/networking/external-access/
- Kafka Network Configuration: https://kafka.apache.org/documentation/#brokerconfigs_advertised.listeners
- Docker Compose Environment Variables: https://docs.docker.com/compose/environment-variables/

## Rollback

If issues occur:
```bash
# Revert to localhost-only configuration
cd ~/omninode_bridge
docker compose \
  -f deployment/docker-compose.yml \
  -f docker-compose.remote.yml \
  --env-file remote.env \
  up -d redpanda
```

## Next Steps

1. ✅ Create `docker-compose.remote-override.yml`
2. ✅ Update `remote.env` with KAFKA_ADVERTISED_* variables
3. ✅ Update `migrate-to-remote.sh` to include override
4. ✅ Update `setup-remote.sh` to use override
5. ✅ Deploy to remote system
6. ✅ Verify external connectivity
7. ✅ Update integration tests to use remote Kafka
8. ✅ Document in deployment guides
