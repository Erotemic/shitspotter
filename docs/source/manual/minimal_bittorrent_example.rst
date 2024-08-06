This example considers an environment with two machines on the same network.
This network is behind a simple NAT router.  One machine is the seeding machine
and the other is the download machine.


Machine Setup
-------------
On both the seeding and download machine.

We are going to use transmission as our client on both machines. We will
primarily use the ``transmission-remote`` tool to interface with the
``transmission-daemon``. In one instance we will use the ``transmission-cli``.

.. code::

   sudo apt-get install transmission-remote transmission-daemon transmission-cli
   sudo apt-get install transmission-daemon transmission-cli

   # Add your user to the debian-transmission group
   sudo usermod -a -G debian-transmission $USER

   # Optional Install GUI interfaces
   sudo apt install transmission-gtk
   sudo apt install transmission-qt

   # Ensure you logout and log into a new shell.

Setup Seeding Machine
---------------------

Ensure that the torrent port is open and correctly forwarded.

This may require configuring your router to forward port 51413 (and 6969?) to
the seeding machine. If you have a firewall, be sure the relevant ports are
open. The following example does this for the ``ufw`` firewall.

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


Creating and Seeding Data
-------------------------

On the seeding machine

.. code:: bash

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

   # Choose a name for the new torrent
   TORRENT_NAME=shared-demo-data-v003
   NUM_DATA=1000

   # Create a dummy set of data that will be shared
   # TODO: Should make working_dpath properly configurable.
   # Hacking and putting this in the var/lib works around permission issues
   # that could be overcome with chmod or config editing.
   # WORKING_DPATH=$HOME/tmp/create-torrent-demo
   WORKING_DPATH=/var/lib/transmission-daemon/downloads

   DATA_DPATH=$WORKING_DPATH/$TORRENT_NAME
   TORRENT_FPATH=$WORKING_DPATH/$TORRENT_NAME.torrent

   mkdir -p "$WORKING_DPATH"
   cd $WORKING_DPATH

   # TODO: make the size of the data configurable
   mkdir -p "$DATA_DPATH"
   for index in $(seq 1 $NUM_DATA); do
       echo "some data $index" > ${DATA_DPATH}/data${index}.txt
       # Add random characters to make the file bigger
       dd if=/dev/urandom bs=10000 count=1 | base32 >> ${DATA_DPATH}/data${index}.txt
   done

   # A list of open tracker URLS is:
   # https://gist.github.com/mcandre/eab4166938ed4205bef4
   # https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_all.txt
   # TODO: Make the tracker configurable and setup good defaults
   # TRACKER_URL=udp://tracker.openbittorrent.com:80
   TRACKER_URL=udp://open.tracker.cl:1337/announce
   COMMENT="a demo torrent named: $TORRENT_NAME"

   # Use the transmission-cli to create the torrent.
   # This can take some time if the data is big.
   transmission-create \
       --comment "$COMMENT" \
       --tracker "$TRACKER_URL" \
       --outfile "$TORRENT_FPATH" \
       "$DATA_DPATH"

   cat "$TORRENT_FPATH"

   # Start seeding the transmission daemon
   transmission-remote --auth transmission:transmission \
       --add "$TORRENT_FPATH" \
       --download-dir "$WORKING_DPATH"

   # Show Registered Torrents to verify success
   transmission-remote --auth transmission:transmission --list

   # DEBUGGING
   # ---------
   # https://forum.transmissionbt.com/viewtopic.php?t=11830

   # Lookup a torrent ID by its name
   TORRENT_ID=$(transmission-lookup-torrent-id "$TORRENT_NAME")
   echo $TORRENT_ID

   # Show info about a torrent
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --info
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --info-files
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --info-trackers
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --info-pieces
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --info-peers

   # Verify the torrent
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --verify

   # Reannounce the torrent
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --reannounce

   # Locate the data
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --find /var/lib/transmission-daemon/downloads

   # CONTEXTUAL: start the torrent
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --start

   # CONTEXTUAL: remove the torrent
   # transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --remove

   # Add a new tracker to the torrent
   TRACKER_URL=udp://open.tracker.cl:1337/announce
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID \
       --tracker-add "$TRACKER_URL"

   # The transmission deamon seems like it needs to have downloads in a special location
   # https://superuser.com/questions/1687624/how-to-create-and-seed-new-torrent-files-for-bittorrent-using-transmisson-client
   # /var/lib/transmission-daemon/downloads


Downloading Data
----------------

On the downloading machine

.. code:: bash

   # TODO: set to the name of the seeding machine that can be used to rsync the data
   SEEDING_MACHINE_NAME=seeding_machine_uri
   SEEDING_MACHINE_NAME=jojo

   TORRENT_NAME=shared-demo-data-v003

   # Get the torrent onto the downloading machine
   rsync $SEEDING_MACHINE_NAME:/var/lib/transmission-daemon/downloads/$TORRENT_NAME.torrent .

   # Register the torrent with the transmission-daemon
   # TODO: configure where you will download
   # TEST_DOWNLOAD_DPATH="$HOME/tmp/transmission-dl"
   # mkdir -p "$TEST_DOWNLOAD_DPATH"
   # Download the torrent to the var lib folder to try and avoid permission issues
   transmission-remote --auth transmission:transmission --add "$TORRENT_NAME.torrent" -w "/var/lib/transmission-daemon/downloads"
   transmission-remote --auth transmission:transmission --list

   # Lookup a torrent ID by its name
   TORRENT_ID=$(transmission-lookup-torrent-id "$TORRENT_NAME")
   echo $TORRENT_ID
   transmission-remote --auth transmission:transmission --torrent $TORRENT_ID --info
