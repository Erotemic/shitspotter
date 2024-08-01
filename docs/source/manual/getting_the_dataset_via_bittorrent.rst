Notes
-----

https://antrikshy.com/code/seeding-torrents-using-transmission-cli
https://forum.transmissionbt.com/viewtopic.php?t=9778


.. code::

   apt show transmission-daemon
   apt show transmission-cli
   apt show transmission-remote
   apt show transmission-qt


.. code::

   sudo apt install transmission-gtk
   sudo apt install transmission-qt
   sudo apt-get install transmission-daemon transmission-cli

   # Create a dummy set of data that will be shared
   WORKING_DPATH=$HOME/tmp/create-torrent-demo
   DATA_DPATH=$WORKING_DPATH/shared-demo-data
   TORRENT_FPATH=$WORKING_DPATH/shared-demo-data.torrent

   mkdir -p "$WORKING_DPATH"
   cd $WORKING_DPATH

   mkdir -p "$DATA_DPATH"
   echo "some data" > $DATA_DPATH/data1.txt
   echo "some data" > $DATA_DPATH/data2.txt
   echo "some other data" > $DATA_DPATH/data3.txt

   transmission-create --comment "a demo torrent" --outfile "$TORRENT_FPATH" "$DATA_DPATH"

   cat $TORRENT_FPATH

   # Start seeding the torrent
   # transmission-cli "$TORRENT_FPATH" -w $(dirname $DATA_DPATH)

   # Do we need additional flags to tell transmission we have the data already?
   # --download-dir tmpdata

   # On remote machine
   rsync toothbrush:tmp/create-torrent-demo/shared-demo-data.torrent .
   transmission-cli shared-demo-data.torrent --download-dir DATA_DPATH




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
    DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
    cd $DVC_DATA_DPATH

    TRACKER_URL=udp://tracker.openbittorrent.com:80
    transmission-create \
        --outfile shitspotter_dvc.torrent \
        --tracker "$TRACKER_URL" \
        --comment "first shitspotter torrent" \
        $HOME/data/dvc-repos/shitspotter_dvc

    # Start seeding the torrent
    transmission-remote --auth transmission:transmission --add shitspotter_dvc.torrent --download-dir $HOME/data/dvc-repos

    transmission-remote --auth transmission:transmission --list

    # Enable local peer discovery in the settings
    cat ~/.config/transmission/settings.json | grep lpd
    sed -i 's/"lpd-enabled": false/"lpd-enabled": true/' ~/.config/transmission/settings.json
    cat ~/.config/transmission/settings.json | grep lpd




Testing On Local Network
------------------------


.. code:: bash

    rsync jojo:data/dvc-repos/shitspotter_dvc/shitspotter_dvc.torrent .

    mkdir -p ./tmpdata
    transmission-remote --auth transmission:transmission --add shitspotter_dvc.torrent --download-dir $PWD/tmpdata
    transmission-remote --auth transmission:transmission --list
    #transmission-cli shitspotter.torrent --download-dir tmpdata


Instructions To Download/Seed The Torrent
-----------------------------------------
