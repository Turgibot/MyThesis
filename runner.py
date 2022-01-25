import os, time, sys
import multiprocessing as mp
import IK_multiple_targets as simulate
import server

#parameters
with_unity = True
dev_mode = True

def start_unity():
    if with_unity:
        print("unity is loading")
        if not dev_mode:
            os.system("./Robot/unitybot.x86_64 &")
            time.sleep(3)

if __name__== "__main__":

    """
    --------------------------------------------------------------------------------
    Unity and mujoco are up and running first
    --------------------------------------------------------------------------------
    """
    bridge = mp.Process(target=server.run)
    unity = mp.Process(target=start_unity)
    mujoco = mp.Process(target=simulate.run, args=(with_unity,))

    bridge.start()
    unity.start()
    mujoco.start()
   

    """
    --------------------------------------------------------------------------------
    Create a pipe from unity to target script to transfer ZED camera input
    --------------------------------------------------------------------------------
    """
    
    bridge.join()
    unity.join()
    mujoco.join()
   