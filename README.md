# Global_DL_SSH
Neural network method for mapping global sea surface height (SSH) from satellite altimetry and sea surface temperature (SST) observations.

This repo contains python code to fully reproduce our training and inference workflow for global SSH mapping from satellite altimetry and gridded SST observations (ADD PREPRINT LINK). 

Disclaimer: this workflow was developed with research/method development in mind, not to be a user-friendly package for production. In future, we hope to refactor the code to streamline the pipeline from observations to maps, make the code more user friendly, and make the workflow cloud-native to avoid having to download large datasets. That said, the code in this repo plus publicly available datasets (linked throughout the repo) is sufficient to fully reproduce our workflow given enough storage space ~15TB and access to multiple CPUs and GPUs.

Steps to reproduce SSH mapping workflow:

1. Download L4 MUR SST product for desired years (2010-2022 in our paper) from NASA PO.DAAC, L3 SSH observations from CMEMS, MDT from CMEMS, bathymetry from GEBCO, and auxilliary data sets from the 2023a Ocean Data Challenge (./data/sad).
2. Create a 'raw' directory with 5615 subdirectories named with integers from 0 to 5614 inclusive.
3. Run generate_global_data.py script. This is the slowest part of the pipeline and could take multiple days depending on number of CPUs available, this is a one time pre-processing step though to subset the observations in the local patches and doesn't need to be repeated if detals of the training/inference procedure is changed at a later date.
4. Run pre_process_training.py to generate ML-ready TFRecord input-output pairs for both training and cross-validation.
5. Run simvp_ddp_training.py to train the neural network (SimVP).
6. Run pre_process_testing.py to generate the input data for creating global SSH maps for 2019, the withheld testing year. NB the input currently withholds Saral/Altika for independent evaluation purposes, the desired altimeter constellation can be specified in the code.
7. Run simvp_predict_ssh.py to predict SSH on all the local patches for 2019
8. Run merge_maps.py to merge the local SSH patch reconstructions and save to NetCDF.

We also provide checkpoint files for the models used in the paper, in which case steps 4-5 can be skipped. These checkpoint files were too large for GitHub but are stored in a Harvard DataVerse along with the SSH maps (ADD REPO LINK ONCE UPLOADED).


