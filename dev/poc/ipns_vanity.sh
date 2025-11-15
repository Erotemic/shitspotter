VANITY_PREFIX="shit"  # Change this to your desired prefix
MATCH_FOUND=false

while [ "$MATCH_FOUND" = false ]; do
    KEY_NAME="vanity-key-$(date +%s)"
    ipfs key gen --type=rsa --size=2048 $KEY_NAME > /dev/null 2>&1
    IPNS_ADDR=$(ipfs key list -l | grep $KEY_NAME | awk '{print $1}')

    if [[ $IPNS_ADDR == $VANITY_PREFIX* ]]; then
        MATCH_FOUND=true
        echo "Found matching IPNS address: $IPNS_ADDR"
        echo "Using key: $KEY_NAME"
        break
    else
        ipfs key rm $KEY_NAME > /dev/null 2>&1  # Remove non-matching keys
    fi
done
