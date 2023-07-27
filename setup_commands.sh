#!/bin/sh

#source /home/s1/rkessler/bin/setup_SNANA-EUPS.sh
#source /home/s1/rmorgan/bin/RM_setup_SNANA-EUPS.sh
source /home/s1/simrankj/bin/RM_setup_SNANA-EUPS.sh
export CVMFS=/cvmfs/des.opensciencegrid.org/
export PATH=$CVMFS/fnal/anaconda2/bin:$PATH
source activate des18a
#source activate des20a
