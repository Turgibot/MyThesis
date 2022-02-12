import os, time, sys
import multiprocessing as mp
import IK_multiple_targets_facing_table as simulate
import server

#parameters
with_unity = True
# with_unity = False
# dev_mode = True
dev_mode = False

lab_scene = 'unitybot.x86_64'

def start_unity():
    if with_unity:
        print("unity is loading")
        if not dev_mode:
            os.system("./Robot/"+lab_scene+" &")
          

if __name__== "__main__":

    """
    --------------------------------------------------------------------------------
    1. Bridge to start a server that listens to unity data
    2. Unity to start a unity app
    3. Simulate to start a mujoco script 
    --------------------------------------------------------------------------------
    """
    bridge = mp.Process(target=server.run)
    unity = mp.Process(target=start_unity)
    mujoco = mp.Process(target=simulate.run, args=(with_unity,))

    """
    --------------------------------------------------------------------------------
    Start processes in order
    --------------------------------------------------------------------------------
    """

    bridge.start()
    unity.start()
    if not dev_mode:
        time.sleep(3)
    mujoco.start()

    """
    --------------------------------------------------------------------------------
    joinn processes
    --------------------------------------------------------------------------------
    """
 
    bridge.join()
    unity.join()
    mujoco.join()
   