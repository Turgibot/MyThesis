#!/bin/bash

# Author : Guy Tordjman
# Copyright (c) NBEL OPU Israel
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mujoco_py
cd ~/MyThesis
python /home/turgibot/MyThesis/runner.py &
# python /home/turgibot/MyThesis/server.py &