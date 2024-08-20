
Machine Setup
-------------

Ensure that the torrent port is open and correctly fowarded.

This may require configuring your router to forward port 51413 to the seeding
machine.

.. code::

    TRANSMISSION_TORRENT_PORT=51413
    TORRENT_TRACKER_PORT=6969

    # If the UFW firewall is enabled, open the ports
    if sudo ufw status | grep "Status: active"; then

        sudo ufw allow $TRANSMISSION_TORRENT_PORT comment "transmission torrent port"
        sudo ufw allow $TORRENT_TRACKER_PORT comment "torrent tracker port"

    fi
    sudo ufw reload
    sudo ufw status


Install the transmission bittorrent client.

.. code::

    # Install Transmission
    sudo apt-get install transmission-daemon transmission-cli


Configure transmission to allow for local peer discovery

.. code::

    # Enable local peer discovery in the settings
    cat ~/.config/transmission/settings.json | grep lpd
    sed -i 's/"lpd-enabled": false/"lpd-enabled": true/' ~/.config/transmission/settings.json
    cat ~/.config/transmission/settings.json | grep lpd

    # For the daemon do this for the global settings
    sudo sed -i 's/"lpd-enabled": false/"lpd-enabled": true/' /etc/transmission-daemon/settings.json
    sudo cat /etc/transmission-daemon/settings.json | grep lpd

    # Restart the transmission daemon
    sudo systemctl restart transmission-daemon.service

    # Verify the status of the daemon
    systemctl status transmission-daemon.service

Optional Install a GUI:

.. code::

    # Install the GUI as well, although this is not needed.
    sudo apt install transmission-gtk


Instructions To Create The Torrent
----------------------------------

..
    https://github.com/qbittorrent/qBittorrent

    Install Instructions Are Modified ChatGPT outputs
    (which was very helpful here).

.. code:: bash

    # Install Transmission CLI
    sudo apt-get install transmission-daemon transmission-cli

    # Create a new torrent
    # DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
    # cd $DVC_DATA_DPATH

    # Hack put the data in transmission downloads folder to avoid perm issues
    time rsync -avprPR --chmod=Du=rwx,Dg=rwx,Do=rwx,Fu=rw,Fg=rw,Fo=rw \
        $HOME/data/dvc-repos/./shitspotter_dvc \
        /var/lib/transmission-daemon/downloads/

    chmod ugo+rw -R $HOME/data/dvc-repos

    # WORKING_DPATH=$HOME/data/dvc-repos
    # TORRENT_FPATH=$HOME/data/dvc-repos/shitspotter_dvc_v1.torrent
    WORKING_DPATH=/var/lib/transmission-daemon/downloads
    TORRENT_NAME=shitspotter_dvc_v3
    TORRENT_FPATH=$WORKING_DPATH/$TORRENT_NAME.torrent
    COMMENT="shitspotter torrent v3"
    #TRACKER_URL=udp://tracker.openbittorrent.com:80
    TRACKER_URL=udp://open.tracker.cl:1337/announce

    cd $WORKING_DPATH

    set -x
    time transmission-create \
        --outfile "$TORRENT_FPATH" \
        --tracker "$TRACKER_URL" \
        --comment "$COMMENT" \
        "$WORKING_DPATH/shitspotter_dvc"
    set +x

    # Start seeding the torrent
    transmission-remote --auth transmission:transmission --add $TORRENT_FPATH --download-dir $WORKING_DPATH

    transmission-remote --auth transmission:transmission --list

    # Enable local peer discovery in the settings
    cat ~/.config/transmission/settings.json | grep lpd
    sed -i 's/"lpd-enabled": false/"lpd-enabled": true/' ~/.config/transmission/settings.json
    cat ~/.config/transmission/settings.json | grep lpd

    transmission-lookup-torrent-id(){
        # Helper function to lookup the id for a torrent by its name
        python3 -c "if 1:
            import subprocess, sys, re
            # This command may need to be modified
            out = subprocess.check_output(
                'transmission-remote --auth transmission:transmission --list',
                shell=True, universal_newlines=True)
            splitpat = re.compile('   *')
            for line in out.split(chr(10)):
                line_ = line.strip()
                if not line_ or line_.startswith(('Sum:', 'ID')):
                    continue
                row_vals = splitpat.split(line_)
                name = row_vals[-1]
                id = row_vals[0].strip('*')
                if name == sys.argv[1]:
                    print(id)
                    sys.exit(0)
            print('error')
            sys.exit(1)
        " "$1"
    }

    transmission-remote --auth transmission:transmission --list

    # Choose a name for the new torrent
    TORRENT_NAME=shitspotter_dvc

    # Lookup a torrent ID by its name
    TORRENT_ID=$(transmission-lookup-torrent-id "$TORRENT_NAME")
    echo $TORRENT_ID

    # Show info about a torrent
    transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --info
    # transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --remove

    transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --reannounce

    # Add a new tracker to the torrent
    TRACKER_URL=udp://open.tracker.cl:1337/announce
    transmission-remote --auth transmission:transmission --torrent $TORRENT_ID \
        --tracker-add "$TRACKER_URL"

    TRACKER_URL=https://academictorrents.com/announce.php
    transmission-remote --auth transmission:transmission --torrent $TORRENT_ID \
        --tracker-add "$TRACKER_URL"

    transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --find $HOME/data/dvc-repos




Testing On Local Network
------------------------


.. code:: bash

    rsync jojo:data/dvc-repos/shitspotter_dvc/shitspotter_dvc.torrent .

    mkdir -p ./tmpdata
    # transmission-remote --auth transmission:transmission --add shitspotter_dvc.torrent --download-dir $PWD/tmpdata
    transmission-remote --auth transmission:transmission --add shitspotter_dvc_v2.torrent --download-dir /var/lib/transmission-daemon/downloads/
    transmission-remote --auth transmission:transmission --list
    #transmission-cli shitspotter.torrent --download-dir tmpdata


Instructions To Download/Seed The Torrent
-----------------------------------------


.. code:: bash


   MAGNENT_LINK="magnet:?xt=urn:btih:040b6645b16518de50278f5d4b2584b3a18438d5&dn=shitspotter%5Fdvc&tr=udp%3A%2F%2Ftracker.openbittorrent.com%3A80"
