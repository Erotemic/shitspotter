#!/bin/bash


install_go(){
    __doc__="
    https://golang.org/doc/install

    https://golang.org/dl/
    https://golang.org/dl/go1.17.linux-amd64.tar.gz
    "
    ARCH="$(dpkg --print-architecture)"
    echo "ARCH = $ARCH"
    GO_VERSION="1.17.5"
    GO_KEY=go${GO_VERSION}.linux-${ARCH}
    URL="https://go.dev/dl/${GO_KEY}.tar.gz"

    declare -A GO_KNOWN_HASHES=(
        ["go1.17.5.linux-amd64-sha256"]="bd78114b0d441b029c8fe0341f4910370925a4d270a6a590668840675b0c653e"
        ["go1.17.5.linux-arm64-sha256"]="6f95ce3da40d9ce1355e48f31f4eb6508382415ca4d7413b1e7a3314e6430e7e"
    )
    EXPECTED_HASH="${GO_KNOWN_HASHES[${GO_KEY}-sha256]}"
    echo "EXPECTED_HASH = $EXPECTED_HASH"

    source ~/local/init/utils.sh
    BASENAME=$(basename "$URL")
    curl_verify_hash "$URL" "$BASENAME" "$EXPECTED_HASH" sha256sum "-L"

    mkdir -p "$HOME/.local"
    tar -C "$HOME/.local" -xzf "$BASENAME"
    mkdir -p "$HOME/.local/bin"
    # Add $HOME/.local/go to your path or make symlinks
    ln -s "$HOME/.local/go/bin/go" "$HOME/.local/bin/go"
    ln -s "$HOME/.local/go/bin/gofmt" "$HOME/.local/bin/gofmt"
}



install_ipfs(){
    __doc__="
    https://docs.ipfs.io/how-to/command-line-quick-start/#prerequisites
    https://docs.ipfs.io/install/command-line/
    https://dist.ipfs.io/#go-ipfs
    https://dist.ipfs.io/go-ipfs

    https://developers.cloudflare.com/distributed-web/ipfs-gateway/setting-up-a-server
    "
    source ~/local/init/utils.sh
    mkdir -p "$HOME/temp/setup-ipfs"
    cd "$HOME/temp/setup-ipfs"
    #URL="https://dist.ipfs.io/go-ipfs/v0.9.0/go-ipfs_v0.9.0_linux-amd64.tar.gz"
    #URL=https://dist.ipfs.io/go-ipfs/v0.11.0/go-ipfs_v0.11.0_linux-amd64.tar.gz

    ARCH="$(dpkg --print-architecture)"
    echo "ARCH = $ARCH"
    IPFS_VERSION="v0.12.0-rc1"
    IPFS_KEY=go-ipfs_${IPFS_VERSION}_linux-${ARCH}
    URL="https://dist.ipfs.io/go-ipfs/${IPFS_VERSION}/${IPFS_KEY}.tar.gz"
    declare -A IPFS_KNOWN_HASHES=(
        ["go-ipfs_v0.12.0-rc1_linux-arm64-sha512"]="730c9d7c31f5e10f91ac44e6aa3aff7c3e57ec3b2b571e398342a62d92a0179031c49fc041cd063403147377207e372d005992fee826cd4c4bba9b23df5c4e0c"
        ["go-ipfs_v0.12.0-rc1_linux-amd64-sha512"]="b0f913f88c515eee75f6dbf8b41aedd876d12ef5af22762e04c3d823964207d1bf314cbc4e39a12cf47faad9ca8bbbbc87f3935940795e891b72c4ff940f0d46"
    )
    EXPECTED_HASH="${IPFS_KNOWN_HASHES[${IPFS_KEY}-sha512]}"
    BASENAME=$(basename "$URL")
    curl_verify_hash "$URL" "$BASENAME" "$EXPECTED_HASH" sha512sum

    echo "BASENAME = $BASENAME"
    tar -xvzf "$BASENAME"
    cp go-ipfs/ipfs "$HOME/.local/bin"

    # That should install IPFS now, lets set it up

    mkdir -p "$HOME/data/ipfs"
    cd "$HOME/data/ipfs"

    # https://github.com/lucas-clemente/quic-go/wiki/UDP-Receive-Buffer-Size
    sudo sysctl -w net.core.rmem_max=2500000

    # Maybe server is not the best profile?
    # https://docs.ipfs.io/how-to/command-line-quick-start/#prerequisites
    #ipfs init --profile server
    #ipfs init --profile badgerds
    ipfs init --profile lowpower

    __results__="
    generating ED25519 keypair...done
    peer identity: 12D3KooWQWMkq2gK91xxBEdkKhd8EysLdQ2bUh4MTYyyqXA3bC3J
    initializing IPFS node at /home/joncrall/.ipfs
    to get started, enter:

        ipfs cat /ipfs/QmQPeNsJPyVWPFDVHb77w8G42Fvo15z4bG2X8D2GhfbSXc/readme
        ipfs cat /ipfs/QmQPeNsJPyVWPFDVHb77w8G42Fvo15z4bG2X8D2GhfbSXc/quick-start
        ipfs cat /ipfs/QmQPeNsJPyVWPFDVHb77w8G42Fvo15z4bG2X8D2GhfbSXc/security-notes
        ipfs cat /ipfs/QmQPeNsJPyVWPFDVHb77w8G42Fvo15z4bG2X8D2GhfbSXc/about
    "

    # In a background tmux session? 
    ipfs daemon

    ipfs swarm peers

    ipfs cat /ipfs/QmSgvgwxZGaBLqkGyWemEDqikCqU52XxsYLKtdy3vGZ8uq > spaceship-launch.jpg

    msg_hash=$(echo "Hello Universe! My name is $(whoami) and I'm excited to start using IPFS!" | ipfs add -q)
    echo "msg_hash = $msg_hash"

    msg_hash=QmeZnz1FJRcSebJuEGc1SqDKE9p5EmrJ93mfrJGXNCkYEM

    # We should be able to see our local network
    curl "http://127.0.0.1:8080/ipfs/$msg_hash"
    # We are not exposed to the world by default
    # But if we were this would work: 
    curl "https://ipfs.io/ipfs/$msg_hash"

    IDENTIFIER="Erotemic <erotemic@gmail.com>"
    KEYID=$(gpg --list-keys --keyid-format LONG "$IDENTIFIER" | head -n 2 | tail -n 1 | awk '{print $1}' | tail -c 9)
    codeblock "
    Hello Universe! It's me, 4AC8B478335ED6ED667715F3622BE571405441B4
    It's cool how easy it is to make a unique message.
    " > _tosign.txt
    gpg --local-user "$KEYID" --clearsign --yes -o _signed.txt _tosign.txt
    cat _signed.txt
    gpg --verify _signed.txt
    MSG_HASH=$(cat _signed.txt | ipfs add -q)
    echo "MSG_HASH = $MSG_HASH"

    curl "http://127.0.0.1:8080/ipfs/Qmae6ELLJMCjhayWWstYtxuiLhiHU9YvFwoiLWUZxChtaz"
    # Can view web UI via: http://localhost:5001/ipfs/bafybeid26vjplsejg7t3nrh7mxmiaaxriebbm4xxrxxdunlk7o337m5sqq/#/ipfs/QmSN2YW4zKEfXgxnSiLYuGvzgYBL6Gqz8WXjUcCq9eov43
    # Can view web UI via: http://localhost:5001/ipfs/bafybeid26vjplsejg7t3nrh7mxmiaaxriebbm4xxrxxdunlk7o337m5sqq/#/ipfs/QmXhQGNHnU46mX48w62jpyK6RWCjxBsPdxBkfrji66MWjC
    # https://github.com/ipfs/go-ipfs/blob/master/docs/fuse.md

    # To get the IPFS node online we need to:
    # (1) give the machine a static IP on your local (router) network
    # (2) Forward port 4001 to your machine
}


setup_shitspotter_ipns(){
    load_secrets
    # We assume the previous command exposes IPNS_PRIVATE_KEY_STORE
    mkdir -p "$IPNS_PRIVATE_KEY_STORE"

    # Can list available "root-folders" 
    ipfs key list

    IPNS_KEY=shitspotter_ipfs_key_v0
    
    # Create a public/private key pair for an IPNS "root-folder"
    ipfs key gen $IPNS_KEY

    cd ~/data/dvc-repos
    #shitspotter_dvc
    # Add the current state of the data to IPFS
    ipfs add -r ~/data/dvc-repos/shitspotter_dvc

    added QmWhKBAQ765YH2LKMQapWp7mULkQxExrjQKeRAWNu5mfBK shitspotter_dvc/data.kwcoco.json
    added QmXBnDDB4TX5FPGddYCQaVSqTmc6cMMTdUwsLwZhuAM3Yb shitspotter_dvc/_cache
    added QmbvEN1Ky3MGGBVDwyMBZvdUCFi1WvfdzkTzgtE7sAvW9B shitspotter_dvc/analysis
    added QmXdQzqcFv3pky621txT5Z6k41gZR9bkckG4no6DNh2ods shitspotter_dvc/assets/_poop-unstructured-2021-02-06
    added QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn shitspotter_dvc/assets/_trashed
    added QmZ4vipXwH7f27VSjx3Bz4aLoeigL9T22sFADv5KCBTFW7 shitspotter_dvc/assets/poop-2020-12-28
    added QmTHipghcRCVamWLojWKQy8KgamtRnPv9fL3dxxPv7VVZx shitspotter_dvc/assets/poop-2021-02-06
    added QmZ3W4pXVkbhQKssWBhBgspeAB3U6GRGD85eff7BvAPNri shitspotter_dvc/assets/poop-2021-03-05
    added QmZb6s53W34rmUJ2s5diw4ErhK3aLb5Td9MtML4u5wqMT5 shitspotter_dvc/assets/poop-2021-04-06
    added QmbZrgM4jCJ8ccU9DLGewPkVBDH6pDVs4vdUUk1jeKyfic shitspotter_dvc/assets/poop-2021-04-19
    added QmTexn6vX8vtAYiZYDq2YmHjoUnnJAAxEtyFPwXsqfvpKy shitspotter_dvc/assets/poop-2021-04-25
    added QmXFyYBVqVVcKqcJuGzo3d9WTRxf4U4cZBmRaT6q52mqLp shitspotter_dvc/assets/poop-2021-05-11T000000
    added QmcTkxhsA4QsWb9KJsLKGnWNyhf7SuMNhAmf55DiXqG8iU shitspotter_dvc/assets/poop-2021-05-11T150000
    added QmNVZ6BGbTWd5Tw5s4E3PagzEcvp1ekxxQL6bRSHabEsG3 shitspotter_dvc/assets/poop-2021-06-05
    added QmQAbQTbTquTyMmd27oLunS3Sw2rZvJH5p7zus4h1fvxdz shitspotter_dvc/assets/poop-2021-06-20
    added QmRkCQkAjYFoCS4cEyiDNnk9RbcoQPafmZvoP3GrpVzJ8D shitspotter_dvc/assets/poop-2021-09-20
    added QmYYUdAPYQGTg67cyRWA52yFgDAWhHDsEQX9yqED3tj4ZX shitspotter_dvc/assets/poop-2021-11-11
    added QmYXXjAutQLdq644rsugp6jxPH6GSaP3kKRTC2jsy4FQMp shitspotter_dvc/assets/poop-2021-11-26
    added QmQAufuJGGn7TDeiEE52k5SLPGrcrawjrd8S2AATrSSBvM shitspotter_dvc/assets/poop-2021-12-27
    added QmfZZwoj1gwGPctBQW5Mkye3a8VuajFBCksHVJH7r9Wn3U shitspotter_dvc/assets
    added QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG shitspotter_dvc

    ipfs pin add QmWhKBAQ765YH2LKMQapWp7mULkQxExrjQKeRAWNu5mfBK
    ipfs pin add QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG
    

    ipfs ls QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG
    ipfs ls QmbvEN1Ky3MGGBVDwyMBZvdUCFi1WvfdzkTzgtE7sAvW9B
    ipfs ls QmbvEN1Ky3MGGBVDwyMBZvdUCFi1WvfdzkTzgtE7sAvW9B
    ipfs ls https://ipfs.io/ipfs/QmRGxbcjYb7ndCzZ4fEBBk2ZR7MtU43f4SSDEeZp9vonx9
    ipfs ls /ipfs/QmRGxbcjYb7ndCzZ4fEBBk2ZR7MtU43f4SSDEeZp9vonx9


    ipfs key export -o "$IPNS_PRIVATE_KEY_STORE/IPNS_KEY.key" $IPNS_KEY

    # https://github.com/ipfs/ipfs-ds-convert
    ipfs ls QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG
    ipfs ls QmfZZwoj1gwGPctBQW5Mkye3a8VuajFBCksHVJH7r9Wn3U
    ipfs ls "https://ipfs.io/ipfs/QmRGxbcjYb7ndCzZ4fEBBk2ZR7MtU43f4SSDEeZp9vonx9"
    ipfs ls "https://ipfs.io/ipfs/QmQAufuJGGn7TDeiEE52k5SLPGrcrawjrd8S2AATrSSBvM"
    ipfs ls /ipfs/QmQAufuJGGn7TDeiEE52k5SLPGrcrawjrd8S2AATrSSBvM

    ipfs get /ipfs/QmXpgvXK7grMY8UFMtHiYoQGX5s8zLQFWTLcxLqvu2ZsAD -o foo.jpg

    lowpower
    #ipfs config profile apply lowpower

    #ipfs config profile apply badgerds
    #ipfs-ds-convert convert
}


demo_ipns(){
    __doc__="
    IPNS is a content addressing system so one link can point to the latest
    version of some content.

    References:
        https://docs.ipfs.io/concepts/ipns/
        https://docs.ipfs.io/concepts/dnslink/
        https://stackoverflow.com/questions/39803954/ipfs-how-to-add-a-file-to-an-existing-folder
    "

    mkdir -p "$HOME/tmp/test_ipns"
    cd "$HOME/tmp/test_ipns"
    mkdir -p folder1
    mkdir -p folder1/folder2
    mkdir -p folder1/folder3

    IPNS_KEY=test_v0
    ipfs key gen $IPNS_KEY

    # Create and add a file (hash1 should be QmUVTKsrYJpaxUT7dr9FpKq6AoKHhEM7eG1ZHGL56haKLG)
    echo "Hello IPFS" > folder1/folder2/hello.txt
    CONTENT_HASH1=$(ipfs add -rq folder1 | tail -1)
    echo "CONTENT_HASH1 = $CONTENT_HASH1"
    ipfs ls -s "$CONTENT_HASH1"

    # Publish it to IPNS, which generates a "key"
    echo "IPNS_KEY = $IPNS_KEY"
    ipfs name publish --key "$IPNS_KEY" "/ipfs/$CONTENT_HASH1"
    curl https://gateway.ipfs.io/ipns/k51qzi5uqu5dm46rm0vzciz3s5no8luxhxhhod7xj5p613zinudxn1t2lcgtty

    # Modify the file (hash2 should be QmVR7cMgdzTjiks29tuqcHbUpF96NGYVmHB5ombRctRuZh)
    echo "Hello again IPFS (modified)" > folder1/folder2/hello.txt
    CONTENT_HASH2=$(ipfs add -rq folder1 | tail -1)
    echo "CONTENT_HASH2 = $CONTENT_HASH2"
    ipfs name publish --key "$IPNS_KEY" "/ipfs/$CONTENT_HASH2"

    curl https://gateway.ipfs.io/ipns/k51qzi5uqu5dm46rm0vzciz3s5no8luxhxhhod7xj5p613zinudxn1t2lcgtty

    ipfs ls /ipns/k51qzi5uqu5dm46rm0vzciz3s5no8luxhxhhod7xj5p613zinudxn1t2lcgtty
    ipfs ls Qmf2ko68zRLhSkQD46t2MjYnTbyMf976BdQDXmAWUN21uX

    ipfs name resolve

 
    ipfs key gen test
    ipfs key export -o todo-hide-secret-file-test.key test

    k51qzi5uqu5dhdij66ntfd6bsozesxh82pfkgys54n2qsmck96nwkr6mvlimk1
    ipfs name publish -k test k51qzi5uqu5dhdij66ntfd6bsozesxh82pfkgys54n2qsmck96nwkr6mvlimk1

    

    ipfs cat /ipns/k51qzi5uqu5dkqxbxeulacqmz5ekmopr3nsh9zmgve1dji0dccdy86uqyhq1m0
    ipfs cat /ipns/k51qzi5uqu5dhdij66ntfd6bsozesxh82pfkgys54n2qsmck96nwkr6mvlimk1
}
