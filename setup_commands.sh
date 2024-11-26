#!/bin/sh

#source /home/s1/rkessler/bin/setup_SNANA-EUPS.sh
#source /home/s1/rmorgan/bin/RM_setup_SNANA-EUPS.sh
source /home/s1/simrankj/bin/RM_setup_SNANA-EUPS.sh
#export CVMFS=/cvmfs/des.opensciencegrid.org/
#export PATH=$CVMFS/fnal/anaconda2/bin:$PATH
#export PATH=/cvmfs/des.opensciencegrid.org/fnal/anaconda3/bin:$PATH

#source activate des18a
#source activate des20a
#source activate /data/des80.a/data/imcmahon/micromamba/envs/mi38/
#source activate /cvmfs/des.opensciencegrid.org/fnal/portconda/des-easyaccess-env

export PATH=/cvmfs/des.opensciencegrid.org/fnal/anaconda3/bin:$PATH
source activate /cvmfs/des.opensciencegrid.org/fnal/portconda/des-y6-imsims-env
#export PATH=/cvmfs/des.opensciencegrid.org/fnal/anaconda3/bin:$PATH
#source activate /cvmfs/des.opensciencegrid.org/fnal/portconda/des-y6-fitvd-plus-env