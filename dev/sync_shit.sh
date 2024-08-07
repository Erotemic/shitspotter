#!/bin/bash
#rsync -avrpRPL ooo:data/dvc-repos/./shitspotter_dvc "$HOME/data/dvc-repos/"


#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-04-19 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-04-19 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-04-25 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-05-11 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-06-05 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-06-20 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-09-20 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-11-11 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-11-26 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-12-27 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2022-01-27 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2022-03-13-T152627 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2022-04-02-T145512 /data/Pictures
#rsync -avrpRP toothbrush:/data/store/./Pictures /data/
#rsync -avrpRP toothbrush:/data/store/./Pictures /data/


## Dry run to check for differenes
#rsync -avrpRPn toothbrush:/data/store/./Pictures /data/

#ooo:/data/Pictures
#toothbrush:/data/store/Pictures

# Sync from here (local toothbrush) to Ooo
rsync -avrpRPL  "$HOME/data/dvc-repos/./shitspotter_dvc" "ooo:/data/dvc-repos/"

# Sync from toothbrush to here (local ooo).
rsync -avrpRPL toothbrush:data/dvc-repos/./shitspotter_dvc "$HOME/data/dvc-repos/"

# Sync from here (local toothbrush) to jojo and handle transmission permissions
rsync -ogavrpRPL --chmod=Du=rwx,Dg=rwx,Do=rwx,Fu=rw,Fg=rw,Fo=rw "$HOME/data/dvc-repos/./shitspotter_dvc" "jojo:/var/lib/transmission-daemon/downloads"

### Ooo changed to mimic toothbrush paths
rsync -avrpRPn toothbrush:/data/store/./Pictures /data/store

# Real run
rsync -avrpRP toothbrush:/data/store/./Pictures /data/store
