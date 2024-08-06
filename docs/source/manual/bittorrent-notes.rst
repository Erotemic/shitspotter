




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
