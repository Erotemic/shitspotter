# YOLO GroundingDino Experiments

In our 2025 paper, the zero shot grounding DINO experiment requires no training and can be run using instructions in

* [run_grounding_dino_experiments_v1.sh](./run_grounding_dino_experiments_v1.sh)


To tune and then evaluate, the procedure we used is documented in 
* [tune_grounding_dino.sh](./tune_grounding_dino.sh)


Reproducing these experiments requires a bit of setup that should be inferable from the information in this repo, but raise an issue if you want to reproduce these and need help.

Our zero shot experiments only rely on HuggingFace infastructure, but for tuning we rely on a fork of
[Erotemic/Open-GroundingDino](https://github.com/Erotemic/Open-GroundingDino)
 which was forked from 
[longzw1997/Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino)
and modified to support our kwcoco format.

