#!/usr/bin/env bash
#
# Меняет Orderer.BatchTimeout у уже работающего channel-а через config update.
#
# Зачем: при больших BatchTimeout (>=30s) сам deployCC падает по
# --waitForEvent таймауту peer-а (он ждёт 30s, что транзакция уйдёт в блок).
# Поэтому сеть/чейнкод поднимаются на дефолтном BatchTimeout=2s, а уже после
# deploy и перед benchmark BatchTimeout меняется этим скриптом.
#
# Использование (из networks/):
#   ./scripts/setBatchTimeout.sh <CHANNEL_NAME> <TIMEOUT>
#   ./scripts/setBatchTimeout.sh bayesian-channel 30s

TEST_NETWORK_HOME=${TEST_NETWORK_HOME:-${PWD}}

# Fabric-бинари и core.yaml — те же пути, что network.sh подставляет своим скриптам
export PATH="${TEST_NETWORK_HOME}/../bin:$PATH"
export FABRIC_CFG_PATH="${TEST_NETWORK_HOME}/../config"

. ${TEST_NETWORK_HOME}/scripts/configUpdate.sh

CHANNEL_NAME=$1
NEW_TIMEOUT=$2

if [ -z "$CHANNEL_NAME" ] || [ -z "$NEW_TIMEOUT" ]; then
  echo "usage: $0 <channel> <timeout, e.g. 30s>" >&2
  exit 1
fi

ARTIFACTS=${TEST_NETWORK_HOME}/channel-artifacts
mkdir -p "$ARTIFACTS"
ORIGINAL=${ARTIFACTS}/batch_timeout_config.json
MODIFIED=${ARTIFACTS}/batch_timeout_modified.json
UPDATE_TX=${ARTIFACTS}/batch_timeout_update.pb

infoln "Fetching current channel config for ${CHANNEL_NAME}"
fetchChannelConfig 1 ${CHANNEL_NAME} ${ORIGINAL}

infoln "Patching Orderer.BatchTimeout -> ${NEW_TIMEOUT}"
jq '.channel_group.groups.Orderer.values.BatchTimeout.value.timeout = "'${NEW_TIMEOUT}'"' \
  ${ORIGINAL} > ${MODIFIED}
verifyResult $? "Failed to patch config JSON, jq required"

createConfigUpdate ${CHANNEL_NAME} ${ORIGINAL} ${MODIFIED} ${UPDATE_TX}

infoln "Signing config update as OrdererMSP admin"
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID=OrdererMSP
export CORE_PEER_TLS_ROOTCERT_FILE=${ORDERER_CA}
export CORE_PEER_MSPCONFIGPATH=${TEST_NETWORK_HOME}/organizations/ordererOrganizations/example.com/users/Admin@example.com/msp
peer channel signconfigtx -f ${UPDATE_TX}
res=$?
verifyResult $res "Failed to sign config update as OrdererMSP admin"

infoln "Submitting config update as Org1 admin"
setGlobals 1
peer channel update -f ${UPDATE_TX} -c ${CHANNEL_NAME} \
  -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "$ORDERER_CA" >&log.txt
res=$?
cat log.txt
verifyResult $res "Failed to submit BatchTimeout config update"
successln "Orderer.BatchTimeout updated to ${NEW_TIMEOUT} on channel '${CHANNEL_NAME}'"
