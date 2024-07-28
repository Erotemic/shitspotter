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
   transmission-cli "$TORRENT_FPATH" -w $(dirname $DATA_DPATH)

   # Do we need additional flags to tell transmission we have the data already?
   # --download-dir tmpdata

   # On remote machine
   rsync toothbrush:tmp/create-torrent-demo/shared-demo-data.torrent .
   transmission-cli shared-demo-data.torrent --download-dir tmpdata




Machine Setup
-------------

Ensure that the torrent port is open and correctly fowarded.

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

    # Install the GUI as well, although this is not needed.
    sudo apt install transmission-gtk


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


Small Demo to Verify Torrents Download Correctly
------------------------------------------------

(Not Working Yet, FIXME)

On the seeding machine

.. code:: bash

   # Create a dummy set of data that will be shared
   WORKING_DPATH=$HOME/tmp/create-torrent-demo
   DATA_DPATH=$WORKING_DPATH/shared-demo-data-v1
   TORRENT_FPATH=$WORKING_DPATH/shared-demo-data-v1.torrent

   mkdir -p "$WORKING_DPATH"
   cd $WORKING_DPATH

   mkdir -p "$DATA_DPATH"
   echo "some data" > $DATA_DPATH/data1.txt
   echo "some data" > $DATA_DPATH/data2.txt
   echo "some other data" > $DATA_DPATH/data3.txt

   transmission-create --comment "a demo torrent v1" --outfile "$TORRENT_FPATH" "$DATA_DPATH"

   cat "$TORRENT_FPATH"

   # Start seeding the torrent
   # Ensure that the download directory contains the data to be seeded
   transmission-cli --verify --download-dir "$(dirname $DATA_DPATH)" $TORRENT_FPATH

   transmission-remote --auth transmission:transmission --add "$TORRENT_FPATH" --download-dir "$(dirname $DATA_DPATH)"

   # List the torrents registered with the daemon
   transmission-remote --auth transmission:transmission --list

   # Start the torrent
   transmission-remote --auth transmission:transmission --torrent 2 --start
   transmission-remote --auth transmission:transmission --list

   transmission-remote --auth transmission:transmission --torrent 1 --remove

   # Verify it is in a good status? Is idle good?
   transmission-remote --auth transmission:transmission --list
   transmission-remote --auth transmission:transmission -t2 -i
   transmission-remote --auth transmission:transmission -t2 --start
   transmission-remote --auth transmission:transmission -tall --start
   transmission-remote --auth transmission:transmission -tall -i
   transmission-remote --auth transmission:transmission -tall --remove

   transmission-remote --auth transmission:transmission -tall --find "$(dirname $DATA_DPATH)"
   transmission-remote --auth transmission:transmission -tall -i
   transmission-remote --auth transmission:transmission -tall -f
   transmission-remote --auth transmission:transmission -tall --get all



   transmission-remote --auth transmission:transmission --add "$TORRENT_FPATH" --download-dir "$(dirname $DATA_DPATH)"

On the downloading machine, do something to transfer the torrent file itself.

.. code:: bash

   SEEDING_MACHINE_NAME=remote
   SEEDING_MACHINE_NAME=toothbrush

   SEEDING_MACHINE_NAME=remote
   rsync $SEEDING_MACHINE_NAME:tmp/create-torrent-demo/shared-demo-data-v1.torrent .

   transmission-cli shared-demo-data-v1.torrent

   rsync toothbrush:shitspotter.torrent .
   transmission-remote --auth transmission:transmission --add "shitspotter.torrent"
   transmission-remote --auth transmission:transmission --add "shitspotter.torrent" -w "$HOME/data/dvc-repos"
   # transmission-remote --auth transmission:transmission --add "shared-demo-data-v1.torrent"




Instructions To Create The Torrent
----------------------------------

..
    https://github.com/qbittorrent/qBittorrent

    Install Instructions Are Modified ChatGPT outputs
    (which was very helpful here).

.. code::

    # Install Transmission CLI
    sudo apt-get install transmission-daemon transmission-cli

    # Create a new torrent
    DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
    transmission-create -o shitspotter.torrent $HOME/data/dvc-repos/shitspotter_dvc

    # Start seeding the torrent
    transmission-cli shitspotter.torrent --download-dir tmpdata

    # Enable local peer discovery in the settings
    cat ~/.config/transmission/settings.json | grep lpd
    sed -i 's/"lpd-enabled": false/"lpd-enabled": true/' ~/.config/transmission/settings.json
    cat ~/.config/transmission/settings.json | grep lpd




Testing On Local Network
------------------------


.. code::

    rysnc jojo:shitspotter.torrent .
    transmission-cli shitspotter.torrent --download-dir tmpdata


Instructions To Download/Seed The Torrent
-----------------------------------------
