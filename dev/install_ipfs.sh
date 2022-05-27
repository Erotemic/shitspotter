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

    INSTALL_PREFIX=$HOME/.local
    _SUDO=""

    # Uncomment for root-level installation
    #_SUDO="sudo"
    #INSTALL_PREFIX=/opt/go

    $_SUDO mkdir -p "$INSTALL_PREFIX"
    $_SUDO tar -C "$INSTALL_PREFIX" -xzf "$BASENAME"
    $_SUDO mkdir -p "$INSTALL_PREFIX/bin"
    # Add $HOME/.local/go to your path or make symlinks
    $_SUDO ln -s "$INSTALL_PREFIX/go/bin/go" "$INSTALL_PREFIX/bin/go"
    $_SUDO ln -s "$INSTALL_PREFIX/go/bin/gofmt" "$INSTALL_PREFIX/bin/gofmt"
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
    IPFS_VERSION="v0.12.2"
    IPFS_KEY=go-ipfs_${IPFS_VERSION}_linux-${ARCH}
    URL="https://dist.ipfs.io/go-ipfs/${IPFS_VERSION}/${IPFS_KEY}.tar.gz"
    declare -A IPFS_KNOWN_HASHES=(
        ["go-ipfs_v0.12.0-rc1_linux-arm64-sha512"]="730c9d7c31f5e10f91ac44e6aa3aff7c3e57ec3b2b571e398342a62d92a0179031c49fc041cd063403147377207e372d005992fee826cd4c4bba9b23df5c4e0c"
        ["go-ipfs_v0.12.0-rc1_linux-amd64-sha512"]="b0f913f88c515eee75f6dbf8b41aedd876d12ef5af22762e04c3d823964207d1bf314cbc4e39a12cf47faad9ca8bbbbc87f3935940795e891b72c4ff940f0d46"
        ["go-ipfs_v0.12.2_linux-arm64-sha512"]="75b71c4a4f7dd888dc8c1995e57a2c67b17c9593f9c4fa3a585a3803f2ac1ae9c2a97c7f7381ca8cf2bc731f0a9eff9b88131b9ba98c15cd41fc68c67e2833b5"
        ["go-ipfs_v0.12.2_linux-amd64-sha512"]="d1b376fe1fb081631af773ea05632090dd79ae5a0057f8b8a0202c28b64a966d14bbfde768ce5a993745761ce56ceed6323a6bd1714f9ae71fa4d68fcbeb1dbb"
    )
    EXPECTED_HASH="${IPFS_KNOWN_HASHES[${IPFS_KEY}-sha512]}"
    BASENAME=$(basename "$URL")
    curl_verify_hash "$URL" "$BASENAME" "$EXPECTED_HASH" sha512sum

    echo "BASENAME = $BASENAME"
    tar -xvzf "$BASENAME"

    _SUDO=""
    INSTALL_PREFIX=$HOME/.local
    export IPFS_PATH=$HOME/.ipfs

    # Uncomment for root-level installation
    #_SUDO="sudo"
    #INSTALL_PREFIX=/opt/ipfs
    #export IPFS_PATH=/data/ipfs
    #$_SUDO mkdir -p $IPFS_PATH

    $_SUDO mkdir -p "$INSTALL_PREFIX/bin"
    $_SUDO cp go-ipfs/ipfs "$INSTALL_PREFIX/bin"

    # That should install IPFS now, lets set it up

    mkdir -p "$HOME/data/ipfs"
    cd "$HOME/data/ipfs"

    # https://github.com/lucas-clemente/quic-go/wiki/UDP-Receive-Buffer-Size
    sudo sysctl -w net.core.rmem_max=2500000

    # Maybe server is not the best profile?
    # https://docs.ipfs.io/how-to/command-line-quick-start/#prerequisites
    #ipfs init --profile server
    #ipfs init --profile badgerds
    # Notes on ipfs profiles:
    #https://github.com/ipfs/go-ipfs-config/blob/0a474258a95d8d9436a49539ad9d7da357b015ab/profile.go


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



sudo_level_hacks(){
    # For setting up on a shared server as sudo
    _SUDO="sudo"
    export PATH=/opt/go/bin:/opt/ipfs/bin:$PATH
    export IPFS_PATH=/data/ipfs

    sudo addgroup ipfs
    sudo usermod -a -G ipfs "$USER"

    sudo chown -R root:ipfs /opt/go
    sudo chown -R root:ipfs /opt/ipfs

    sudo chown -R root:ipfs $IPFS_PATH
    sudo chmod -R g+rw $IPFS_PATH

    ipfs init --profile badgerds

    # RIRE
    ipfs pin add bafybeih23xv6uamx7k27wk4uvzkxdtdryqeok22hpl3ybideggcjhipwme --progress

    DEST_PREFIX=/usr/local
    $_SUDO ln -s "/opt/go/go/bin/go" "$DEST_PREFIX/bin/go"
    $_SUDO ln -s "/opt/go/go/bin/gofmt" "$DEST_PREFIX/bin/gofmt"
    $_SUDO ln -s "/opt/ipfs/bin/ipfs" "$DEST_PREFIX/bin/ipfs"

    /usr/local/bin/ipfs

    # Assume transfer local install to root
    sudo cp -v -- */ipfs-cluster-* "/opt/ipfs/bin"

    DEST_PREFIX=/usr/local
    $_SUDO ln -s "/opt/ipfs/bin/ipfs-cluster-ctl" "$DEST_PREFIX/bin/ipfs-cluster-ctl"
    $_SUDO ln -s "/opt/ipfs/bin/ipfs-cluster-follow" "$DEST_PREFIX/bin/ipfs-cluster-follow"
    $_SUDO ln -s "/opt/ipfs/bin/ipfs-cluster-service" "$DEST_PREFIX/bin/ipfs-cluster-service"
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

    # Update 2022-01-30
    QmaPPoPs7wXXkBgJeffVm49rd63ZtZw5GrhvQQbYrUbrYL

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

ipfs_debug_and_info(){
    SHIT_CID=QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG
    ipfs pin add "${SHIT_CID}" --progress
    
    ipfs pin add QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG --progress
    ipfs refs -r QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG | wc

    ( HASH=$SHIT_CID ; echo $HASH ; ipfs refs -r $HASH ) | xargs -l bash -c 'echo -n "$0 - "; ipfs dht findprovs -n 2 $0 | wc -l'
    

    ipfs dht findprovs -r "${SHIT_CID}"



    # Generate a random file, pin it, and see if it is accessible
    pip install diceware

    diceware -n 100 > rando.txt
    RANDO_HASH=$(ipfs add rando.txt -q)
    echo "RANDO_HASH=$RANDO_HASH"

    # On another machine
    RANDO_HASH=QmYMVEQrW8xd3nWKQRyYbjWef7hDSGrcirz8HX8jKA8osQ
    ipfs get $RANDO_HASH


    #### With bigger data
    # Make big data on PI
    mkdir -p test_bigger_data_pin
    head -c500000000 /dev/urandom > test_bigger_data_pin/rando1_500MB.txt
    head -c50000000 /dev/urandom > test_bigger_data_pin/rando2_50MB.txt
    head -c50000000 /dev/urandom > test_bigger_data_pin/rando3_50MB.txt
    head -c50000000 /dev/urandom > test_bigger_data_pin/rando4_50MB.txt

    ipfs add -r test_bigger_data_pin
    BIG_DATA_HASH=$(ipfs add -r test_bigger_data_pin -q | tail -n 1)
    echo "BIG_DATA_HASH = $BIG_DATA_HASH"

    BIG_DATA_HASH="Qmdf7DHgHHJXwm1moQDkxns3Yupw6nCbjqurHamBr6KL6R"

    ipfs get Qmdf7DHgHHJXwm1moQDkxns3Yupw6nCbjqurHamBr6KL6R


    # Show my peer id
    ipfs config show | jq .Identity

    # Find number of peers with data:
    # https://www.reddit.com/r/ipfs/comments/r86kp6/is_there_a_way_to_see_how_many_peers_are_hosting/
    # This shows nearby "peer-ids" (lower bound) of people hosting the data
    # the
    source ~/local/init/utils.sh
    mapfile -t PEER_ID_ARR <<< "$(ipfs dht findprovs QmNj2MbeL183GtPoGkFv569vMY8nupUVGEVvvvqhjoAATG)"
    bash_array_repr "${PEER_ID_ARR[@]}"

    # Print info about peers
    for PEER_ID in "${PEER_ID_ARR[@]}"; do
        printf "\n\n=====\n\n"
        echo "PEER_ID = $PEER_ID"
        ipfs dht findpeer "$PEER_ID"
    done
}

http_server(){
    #https://reposhub.com/javascript/misc/ipfs-ipfs-webui.html
    # https://discuss.ipfs.io/t/how-can-i-enable-remote-connection-to-webui/698/3
    ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["http://localhost:3000", "https://webui.ipfs.io", "http://127.0.0.1:5001"]'
    ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["POST"]'

    # There isn't a great way to connect to the webUI on another machine, you
    # need to forward the port to your local machine and then connect through
    # that.
    ssh -L 5001:localhost:5001 pazuzu
    # To undo:
    # ipfs config --json API.HTTPHeaders {}
    
}


install_ipfs_service(){
    # https://gist.github.com/pstehlik/9efffa800bd1ddec26f48d37ce67a59f
    # https://www.maxlaumeister.com/u/run-ipfs-on-boot-ubuntu-debian/
    # https://linuxconfig.org/how-to-create-systemd-service-unit-in-linux#:~:text=There%20are%20basically%20two%20places,%2Fetc%2Fsystemd%2Fsystem%20.
    source ~/local/init/utils.sh
    # https://linuxconfig.org/how-to-create-systemd-service-unit-in-linux#:~:text=There%20are%20basically%20two%20places,%2Fetc%2Fsystemd%2Fsystem%20.
    SERVICE_DPATH=/etc/systemd/system
    IPFS_SERVICE_FPATH=$SERVICE_DPATH/ipfs.service
    IPFS_EXE=$(which ipfs)
    echo "IPFS_EXE = $IPFS_EXE"
    echo "IPFS_SERVICE_FPATH = $IPFS_SERVICE_FPATH"
    sudo_writeto $IPFS_SERVICE_FPATH "
        [Unit]
        Description=IPFS daemon
        After=network.target
        [Service]
        Environment=\"IPFS_PATH=/data/ipfs\"
        User=$USER
        ExecStart=${IPFS_EXE} daemon
        [Install]
        WantedBy=multiuser.target
        "
    #sudo systemctl daemon-reload
    sudo systemctl start ipfs
    sudo systemctl status ipfs
        
}


init_ipfs_cluster(){
    # https://cluster.ipfs.io/documentation/deployment/setup/
    source ~/local/init/utils.sh
    mkdir -p "$HOME/temp/setup-ipfs-cluster"
    cd "$HOME/temp/setup-ipfs-cluster"

    ARCH="$(dpkg --print-architecture)"
    echo "ARCH = $ARCH"
    IPFS_CLUSTER_VERSION="v0.14.4"

    EXE_NAME=ipfs-cluster-ctl
    KEY=${EXE_NAME}_${IPFS_CLUSTER_VERSION}_linux-${ARCH}
    URL="https://dist.ipfs.io/${EXE_NAME}/${IPFS_CLUSTER_VERSION}/$KEY.tar.gz"
    declare -A KNOWN_HASHES=(
        ["ipfs-cluster-ctl_v0.14.4_linux-arm64-sha512"]="2c8d2d5023c4528a902889b33a7e52fd71261f34ada62999d6a8fe3910d652093a95320693f581937d8509ccb07ff5b9501985e0262a67c12e64419fa49e4339"
        ["ipfs-cluster-ctl_v0.14.4_linux-amd64-sha512"]="454331518c0d67319c873c69b7fceeab06cbe4bb926cecb16cc46da86be79d56f63b7100b9ccba5a9c6e99722e27446e33623d7191f3b09c6faed4c36c15204a"
    )
    EXPECTED_HASH="${KNOWN_HASHES[${KEY}-sha512]}"
    BASENAME=$(basename "$URL")
    curl_verify_hash "$URL" "$BASENAME" "$EXPECTED_HASH" sha512sum

    EXE_NAME=ipfs-cluster-service
    KEY=${EXE_NAME}_${IPFS_CLUSTER_VERSION}_linux-${ARCH}
    URL="https://dist.ipfs.io/${EXE_NAME}/${IPFS_CLUSTER_VERSION}/$KEY.tar.gz"
    declare -A KNOWN_HASHES=(
        ["ipfs-cluster-service_v0.14.4_linux-arm64-sha512"]="79129b6cc94d36a9921f8e07e207ee13336c89a245a44b075b0ada50b72796b31a7e90bf15171e355e0a1e08cc55e40e67376f813016d678f5a7d007327ffd04"
        ["ipfs-cluster-service_v0.14.4_linux-amd64-sha512"]="430dbbab5c651fcf99ae9b122fc663cdb5785e51e8dc6c2381b0b82e5f963c5945f9c1c10781d50a5aeac675dc3bbf783b2e03b8c3d5fb5e94804cb2c2efcc9f"
    )
    EXPECTED_HASH="${KNOWN_HASHES[${KEY}-sha512]}"
    BASENAME=$(basename "$URL")
    curl_verify_hash "$URL" "$BASENAME" "$EXPECTED_HASH" sha512sum


    EXE_NAME=ipfs-cluster-follow
    KEY=${EXE_NAME}_${IPFS_CLUSTER_VERSION}_linux-${ARCH}
    URL="https://dist.ipfs.io/${EXE_NAME}/${IPFS_CLUSTER_VERSION}/$KEY.tar.gz"
    declare -A KNOWN_HASHES=(
        ["ipfs-cluster-follow_v0.14.4_linux-arm64-sha512"]="136fe71f0df0dd44b5ac3e97db8529399dfa84e18fb7b15f16120503dcb44b339e55263a264d1c3ff4bd693c68bcfe5bc208b4b37aa29402fb545256ab06eb88"
        ["ipfs-cluster-follow_v0.14.4_linux-amd64-sha512"]="22ac2f2a89693c715be5f8a528c89def7c54abc3a3256a85468730c974831dff2e0a21ea489d66c0457f61e7e76d948614c99794333cb8a0dabf3e4a04f74ef8"
    )
    EXPECTED_HASH="${KNOWN_HASHES[${KEY}-sha512]}"
    BASENAME=$(basename "$URL")
    curl_verify_hash "$URL" "$BASENAME" "$EXPECTED_HASH" sha512sum

    tar -xvzf "ipfs-cluster-ctl_v0.14.4_linux-${ARCH}.tar.gz"
    tar -xvzf "ipfs-cluster-follow_v0.14.4_linux-${ARCH}.tar.gz"
    tar -xvzf "ipfs-cluster-service_v0.14.4_linux-${ARCH}.tar.gz"

    INSTALL_PREFIX="$HOME/.local"
    mkdir -p "$INSTALL_PREFIX/bin"
    cp -v -- */ipfs-cluster-* "$INSTALL_PREFIX/bin"

    # Now initialize
    export IPFS_PATH=/data/ipfs
    ipfs-cluster-service init --consensus crdt

    # https://cluster.ipfs.io/documentation/deployment/setup/
}
