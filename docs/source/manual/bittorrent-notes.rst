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



Misc Notes
----------

Other notes that are not well organized yet

.. code:: bash
   ###################################
   # Work In Progress After This Point
   ###################################

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


   # Query download dir / incomplete dir
   transmission-daemon --dump-settings 2>&1| jq '."download-dir"'
   transmission-daemon --dump-settings 2>&1| jq '."incomplete-dir"'

    # Reload settings
    sudo invoke-rc.d transmission-daemon reload
    sudo service transmission-daemon restart

    # Show transmission daemon logs
    journalctl -u transmission-daemon.service

   # Move the data into a place where the daemon can see it
   # (would be nice if we could tell transmission where the data is instead)
   #rsync -rPR ./shared-demo-data-v001 /var/lib/transmission-daemon/downloads
   ## Move the torrent file there as well
   #cp $TORRENT_FPATH /var/lib/transmission-daemon/downloads

   # Look at remote GUI (need to open firewall?)
   transmission-qt --remote 192.168.222.18 --username transmission --password transmission

    # Check visibility
    # Get your WAN IP Address
    WAN_IP_ADDRESS=$(curl ifconfig.me)
    echo "WAN_IP_ADDRESS = $WAN_IP_ADDRESS"

    # Check if node is visible to WAN
    # https://canyouseeme.org/
    # https://portchecker.co/canyouseeme



    ### Attempt to enable more logs in the service
    # https://askubuntu.com/questions/397589/enable-logging-for-service
    # https://www.cviorel.com/enable-transmission-daemon-logging-to-file/
    SERVICE_FPATH=/lib/systemd/system/transmission-daemon.service
    cat $SERVICE_FPATH

    sudo sed -i 's|ExecStart.*|ExecStart=/usr/bin/transmission-daemon -f --log-debug --logfile /var/log/transmission.log|g' $SERVICE_FPATH
    sudo touch /var/log/transmission.log
    sudo chown debian-transmission /var/log/transmission.log
    sudo chmod 644 /var/log/transmission.log

    sudo systemctl daemon-reload
    sudo service transmission-daemon restart

    watch tail /var/log/transmission.log

    # Maybe try giving the transmission-daemon permission to the user group?
    sudo usermod -a -G $USER debian-transmission




References:

    https://askubuntu.com/questions/397589/enable-logging-for-service
    https://superuser.com/questions/385685/failed-sharing-my-first-file-with-bittorrent


Getting the 2025-04-20 dataset seeding
--------------------------------------

Steps I took to get the data seeding. On my main box, academic torrents didn't
see that I was seeding via deluge. Might be a port issue. Putting it on my seed
box, which should make it visible.

Turns out the issue was that I just wasn't using the right tracker URL. In deluge changing it to: https://academictorrents.com/announce.php worked.
In any case, I need to put the data on my seed box anyway.

.. code:: bash

    rsync /var/lib/transmission-daemon/downloads/shitspotter_dvc-2025-04-20.torrent jojo:.

    ssh jojo

    # On jojo
    transmission-remote --auth transmission:transmission --add shitspotter_dvc-2025-04-20.torrent --download-dir /var/lib/transmission-daemon/downloads/
    transmission-remote --auth transmission:transmission --list

    # On main machine, hack to get the data seeding
    rsync -avPR /var/lib/transmission-daemon/downloads/./shitspotter_dvc-2025-04-20 jojo:/var/lib/transmission-daemon/downloads/

    # Note, once I got deluge to recognize, I stopped rsyncing and am checking that transmission will complete the download.
    python -m shitspotter.transmission list

    python -m shitspotter.transmission lookup_id coco2014 --verbose
    TORRENT_ID=$(python -m shitspotter.transmission lookup_id shitspotter_dvc-2025-04-20)
    transmission-remote --auth transmission:transmission -t$TORRENT_ID --files | head

    python -m shitspotter.transmission info coco2014
    python -m shitspotter.transmission info shitspotter_dvc-2025-04-20 --verbose=3

    python -m shitspotter.transmission add_tracker shitspotter_dvc-2025-04-20 https://academictorrents.com/announce.php --verbose=3
    python -m shitspotter.transmission find shitspotter_dvc-2025-04-20 /var/lib/transmission-daemon/downloads/ --verbose=3
    python -m shitspotter.transmission verify shitspotter_dvc-2025-04-20 --verbose=3


    # Enable DHT, PEX, and LPD
    transmission-remote --auth transmission:transmission --dht
    transmission-remote --auth transmission:transmission --pex
    transmission-remote --auth transmission:transmission --lpd

    # Check session stats:
    transmission-remote --auth transmission:transmission -t4 --info-trackers
    transmission-remote --auth transmission:transmission -t4 --info
    transmission-remote --auth transmission:transmission --session-info
    transmission-remote --auth transmission:transmission --session-stats

    # Move the torrent data
    transmission-remote --auth transmission:transmission -t4 --move /flash/debian-transmission-downloads -w /flash/debian-transmission-downloads
    transmission-remote --auth transmission:transmission -t4 --info

    python -m shitspotter.transmission move shitspotter_dvc /flash/debian-transmission-downloads --verbose=3

    python -m shitspotter.transmission add "magnet:?xt=urn:btih:27a2512ae93298f75544be6d2d629dfb186f86cf" --verbose=3
    python -m shitspotter.transmission info "27a2512ae93298f75544be6d2d629dfb186f86cf"
    python -m shitspotter.transmission remove shitspotter_dvc-2025-04-20

    python -m shitspotter.transmission add "magnet:?xt=urn:btih:27a2512ae93298f75544be6d2d629dfb186f86cf" --verbose=3
    python -m shitspotter.transmission remove 27a2512ae93298f75544be6d2d629dfb186f86cf

    # Using the magnent link was having issues, downloading the torrent file directly
    curl -O https://academictorrents.com/download/27a2512ae93298f75544be6d2d629dfb186f86cf.torrent
    python -m shitspotter.transmission add 27a2512ae93298f75544be6d2d629dfb186f86cf.torrent --verbose=3

    # Maybe it won't download because of the ufw rules on jojo, lets stop it, do an rsync, and then
    # ensure external downloads work.
    python -m shitspotter.transmission stop shitspotter_dvc-2025-04-20 --verbose=3
    rsync -avPR /var/lib/transmission-daemon/downloads/./shitspotter_dvc-2025-04-20 jojo:/flash/debian-transmission-downloads
    python -m shitspotter.transmission start shitspotter_dvc-2025-04-20 --verbose=3
    python -m shitspotter.transmission info shitspotter_dvc-2025-04-20
    python -m shitspotter.transmission verify shitspotter_dvc-2025-04-20
    python -m shitspotter.transmission list
