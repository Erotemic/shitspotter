Small Demo to Verify Torrents Download Correctly
------------------------------------------------


Add your user to the debian-transmission group

sudo usermod -a -G debian-transmission $USER


On the seeding machine

.. code:: bash

   transmission-lookup-torrent-id(){
       # Helper function to lookup the id for a torrent by its name
       python3 -c "if 1:
           import subprocess
           import sys
           # This command may need to be modified
           out = subprocess.check_output(
               'transmission-remote --auth transmission:transmission --list',
               shell=True, universal_newlines=True)
           import re
           splitpat = re.compile('   *')
           for line in out.split(chr(10)):
               line_ = line.strip()
               if not line_:
                   continue
               if line_.startswith('Sum:'):
                   continue
               if line_.startswith('ID'):
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

   # Create a dummy set of data that will be shared
   # WORKING_DPATH=$HOME/tmp/create-torrent-demo
   WORKING_DPATH=/var/lib/transmission-daemon/downloads
   TORRENT_NAME=shared-demo-data-v002
   DATA_DPATH=$WORKING_DPATH/$TORRENT_NAME
   TORRENT_FPATH=$WORKING_DPATH/$TORRENT_NAME.torrent

   mkdir -p "$WORKING_DPATH"
   cd $WORKING_DPATH

   mkdir -p "$DATA_DPATH"
   echo "some data" > $DATA_DPATH/data1.txt
   echo "some data" > $DATA_DPATH/data2.txt
   echo "some other data" > $DATA_DPATH/data3.txt

   # A list of open tracker URLS is:
   # https://gist.github.com/mcandre/eab4166938ed4205bef4
   # https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_all.txt
   TRACKER_URL=udp://tracker.openbittorrent.com:80
   TRACKER_URL=udp://open.tracker.cl:1337/announce
   COMMENT="a demo torrent named: $TORRENT_NAME"

   transmission-create \
       --comment "$COMMENT" \
       --tracker "$TRACKER_URL" \
       --outfile "$TORRENT_FPATH" \
       "$DATA_DPATH"

   cat "$TORRENT_FPATH"

   tree -f $DATA_DPATH

   # Start seeding the transmission daemon
   transmission-remote --auth transmission:transmission \
       --add "$TORRENT_FPATH" \
       --download-dir "$WORKING_DPATH"

   # Show Registered Torrents to verify success
   transmission-remote --auth transmission:transmission --list

   # DEBUGGING
   # https://forum.transmissionbt.com/viewtopic.php?t=11830

   # Start the torrent
   transmission-remote --auth transmission:transmission --list

   # Lookup a torrent ID by its name
   TORRENT_ID=$(transmission-lookup-torrent-id "$TORRENT_NAME")
   echo $TORRENT_ID

   # Show info about a torrent
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --info

   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --remove

   # Verify the torrent
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --verify

   # Verify the torrent
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --start

   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --reannounce

   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --find /var/lib/transmission-daemon/downloads

   # Add a new tracker to the torrent
   TRACKER_URL=udp://open.tracker.cl:1337/announce
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID \
       --tracker-add "$TRACKER_URL"

   # Alternative: start seeding the torrent
   # Ensure that the download directory contains the data to be seeded
   # transmission-cli --verify --download-dir "$(dirname $DATA_DPATH)" $TORRENT_FPATH

   # The transmission deamon seems like it needs to have downloads in a special location
   # https://superuser.com/questions/1687624/how-to-create-and-seed-new-torrent-files-for-bittorrent-using-transmisson-client
   # /var/lib/transmission-daemon/downloads


On the downloading machine

.. code:: bash

   SEEDING_MACHINE_NAME=some_remote_name
   # SEEDING_MACHINE_NAME=toothbrush
   SEEDING_MACHINE_NAME=jojo

   rsync $SEEDING_MACHINE_NAME:/var/lib/transmission-daemon/downloads/shared-demo-data-v002.torrent .

   TEST_DOWNLOAD_DPATH="$HOME/tmp/transmission-dl"
   mkdir -p "$TEST_DOWNLOAD_DPATH"
   transmission-remote --auth transmission:transmission --add "shared-demo-data-v002.torrent" -w "$TEST_DOWNLOAD_DPATH"
   transmission-remote --auth transmission:transmission --list

   tree $TEST_DOWNLOAD_DPATH

   # Show Registered Torrents to verify success
   transmission-remote --auth transmission:transmission --list

   # transmission-cli shared-demo-data-v1.torrent
   transmission-remote --auth transmission:transmission -t3 -i


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



On the downloading machine, do something to transfer the torrent file itself.

.. code:: bash

   SEEDING_MACHINE_NAME=some_remote_name
   # SEEDING_MACHINE_NAME=toothbrush
   SEEDING_MACHINE_NAME=jojo

   rsync $SEEDING_MACHINE_NAME:tmp/create-torrent-demo/shared-demo-data-v1.torrent .

   TEST_DOWNLOAD_DPATH="$HOME/tmp/transmission-dl"
   transmission-remote --auth transmission:transmission --add "shared-demo-data-v1.torrent" -w "$TEST_DOWNLOAD_DPATH"

   # transmission-cli shared-demo-data-v1.torrent


   # Shistposter test

   #rsync toothbrush:shitspotter.torrent .
   #transmission-remote --auth transmission:transmission --add "shitspotter.torrent"
   #transmission-remote --auth transmission:transmission --add "shitspotter.torrent" -w "$HOME/data/dvc-repos"
   # transmission-remote --auth transmission:transmission --add "shared-demo-data-v1.torrent"

   transmission-remote --auth transmission:transmission --add "$TORRENT_FPATH" --download-dir "$(dirname $DATA_DPATH)"

   # Show Registered Torrents to verify success
   transmission-remote --auth transmission:transmission --list


   # Look at remote GUI (need to open firewall?)
   transmission-qt --remote 192.168.222.18 --username transmission --password transmission



