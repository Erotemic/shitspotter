rsync -avrpRPL toothbrush:data/dvc-repos/./shitspotter_dvc "$HOME/data/dvc-repos/"


rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-04-19 /data/Pictures
rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-04-19 /data/Pictures
rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-04-25 /data/Pictures
rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-05-11 /data/Pictures
rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-06-05 /data/Pictures
rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-06-20 /data/Pictures
rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-09-20 /data/Pictures
rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-11-11 /data/Pictures
rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-11-26 /data/Pictures
rsync -avrpRP toothbrush:/data/store/Pictures/./Phone-DCIM-2021-12-27 /data/Pictures

rsync -avrpRP toothbrush:/data/store/./Pictures /data/


# Dry run to check for differenes
rsync -avrpRPn toothbrush:/data/store/./Pictures /data/

ooo:/data/Pictures
toothbrush:/data/store/Pictures
