import asyncio
from cgitb import handler
import logging
import time
from turtle import update
from typing import AsyncIterable, Iterable

import grpc
from scipy.__config__ import show
import ZedStreamer_pb2 as ZedStreamer_pb2
import ZedStreamer_pb2_grpc as ZedStreamer_pb2_grpc
import numpy as np
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

#TODO implement singleton
class SimHandler():
    __instance = None
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if SimHandler.__instance == None:
            SimHandler()
        return SimHandler.__instance
    def __init__(self):
        """ Virtually private constructor. """
        if SimHandler.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            SimHandler.__instance = self
        
        self.params = [50, 50]

    async def set_params(self, conn):
        while True:
            data = conn.recv()
            self.params[0] = data[0]
            self.params[1] = data[1]
            print("received", data)

    def getSpikesFrom2Frames(self, prev_frame, frame):
        
        pos_th = self.params[0]
        neg_th = self.params[1]

        deltas = np.array(np.array(prev_frame)-np.array(frame), dtype=np.int8)
        deltas = np.where(deltas == 1, 0, deltas)
        deltas = np.where(deltas == -1, 0, deltas)
        deltas = np.where(deltas >= pos_th, 1, deltas)
        deltas = np.where(deltas < -neg_th, -1, deltas)
        deltas = np.where(deltas > 1, 0, deltas)
        deltas = np.where(deltas < -1 , 0, deltas)

        print(self.params)
        
        return deltas

    async def show_images(self, conn):
        name = "ZED Stereo Camera Simulator"
        cv2.startWindowThread()
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        time_before = time.time()
        fps = "FPS: 30"
        while True:
            image = conn.recv()
            frame = np.array(list(image.image_data), dtype = np.uint8)
            frame = frame.reshape((image.height*2, image.width, 3))
            left = frame[:image.height]
            right = frame[image.height:]
            frame = np.concatenate([left, right], axis=1)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)
            cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(name, frame)
            cv2.waitKey(1)
            time_after = time.time()
            fps = "FPS: "+str(1//(time_after-time_before))
            time_before = time_after

    async def show_spikes(self, conn):
        name = "DVS Stereo Camera Simulator"
        prev_frame = None
        colored_frame = None
        cv2.startWindowThread()
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        while True:
            image = conn.recv()
            frame = np.array(list(image.image_data), dtype = np.uint8)
            frame = frame.reshape((image.height*2, image.width, 3))
            left = frame[:image.height]
            right = frame[image.height:]
            frame = np.concatenate([left, right], axis=1)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.flip(frame, 0)
            src_shape = (frame.shape[0], frame.shape[1], 3) 
            if prev_frame is not None:
                spikes_frame = self.getSpikesFrom2Frames(prev_frame, frame).flatten()
                shape = [int(x) for x in spikes_frame.shape]
                colored_frame = np.zeros(shape=shape+[3], dtype="uint8")
                colored_frame[spikes_frame==1] = [255, 0, 0] 
                colored_frame[spikes_frame==-1] = [0, 0, 255] 
                colored_frame = colored_frame.reshape(src_shape)
                cv2.imshow(name, colored_frame)
                cv2.waitKey(1)

            prev_frame = frame.copy()
        


class ZedStreamerServicer(ZedStreamer_pb2_grpc.ZedStreamerServicer):
    def __init__(self, sim_params, images=None, conn1=None, conn2=None,  conn3=None) -> None:
        self.images = images
        self.conn1 = conn1
        self.conn2 = conn2
        self.conn3 = conn3
        self.sim_params = sim_params

    

    async def SendVideo(self, request_iterator: AsyncIterable[
        ZedStreamer_pb2.Image], unused_context) -> ZedStreamer_pb2.Received:
        count = 0
        async for image in request_iterator:
            if self.conn1 is not None:
                self.conn1.send(image)
            if self.conn2 is not None:
                self.conn2.send(image)

        return ZedStreamer_pb2.Received(ack=count)

    async def SendImage(self, request, context) -> ZedStreamer_pb2.Received:
        frame = np.array(list(request.data), dtype = np.uint8)
        frame = frame.reshape((request.height, request.width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.conn1.send(frame)

        return ZedStreamer_pb2.Received(ack=True)

    async def SendParams(self, request, context) -> ZedStreamer_pb2.Received:
        
        data = list(request.data)
        if self.conn3 is not None:
                self.conn3.send(data)


        return ZedStreamer_pb2.Received(ack=True)


    
    

async def serve(servicer) -> None:
    server = grpc.aio.server()
    ZedStreamer_pb2_grpc.add_ZedStreamerServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()
    cv2.destroyAllWindows()

def startServer(images, sim_params, zed_conn, spikes_conn, params_parent_con):
    servicer = ZedStreamerServicer(images, sim_params, zed_conn, spikes_conn, params_parent_con)
    logging.basicConfig(level=logging.INFO)
    asyncio.get_event_loop().run_until_complete(serve(servicer))


async def run_the_show(zed_child_con,spikes_child_con,params_child_con):
    handler = SimHandler()
    zed_task =  asyncio.create_task(handler.show_images(zed_child_con))
    # spikes_task =  asyncio.create_task(handler.show_spikes(spikes_child_con))
    # params_task =  asyncio.create_task(handler.set_params(params_child_con))
    await asyncio.sleep(1)


def start_the_show(zed_child_con,spikes_child_con,params_child_con):
    asyncio.get_event_loop().run_forever(run_the_show(zed_child_con,spikes_child_con,params_child_con))

# use a pipe to transfer image stream
def run():
    zed_images = []
    sim_params = []
 
    zed_parent_con, zed_child_con = mp.Pipe()
    spikes_parent_con, spikes_child_con = mp.Pipe()
    params_parent_con, params_child_con = mp.Pipe()

    p0 = mp.Process(target=startServer, args=(zed_images, sim_params, zed_parent_con, spikes_parent_con, params_parent_con))
    p1 = mp.Process(target=start_the_show, args=(zed_child_con,spikes_child_con,params_child_con))


    p0.start()
    p1.start()

   
    p0.join()
    p1.join()


if __name__ == "__main__":
    run()