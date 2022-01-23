#!/bin/sh

# Author : Guy Tordjman
# Copyright (c) NBEL OPU Israel

conda activate mujoco_py
cd ~/MyThesis
python /home/turgibot/MyThesis/runner.py &
python /home/turgibot/MyThesis/server.py &