import asyncio
from cgitb import handler
from collections import defaultdict
from datetime import date
from email.policy import default
import logging
import time
from turtle import update
from typing import AsyncIterable, Iterable

import grpc
from sympy import im
import ZedStreamer_pb2 as ZedStreamer_pb2
import ZedStreamer_pb2_grpc as ZedStreamer_pb2_grpc
import numpy as np
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

import os
from datetime import datetime
import pickle as pkl
import pathlib

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
        

    def set_params(self, conn, sim_params):
        while True:
            data = conn.recv()
            sim_params[0] = data[0]
            sim_params[1] = data[1]
            sim_params[2] = data[2]
            sim_params[3] = data[3]
            

    def getSpikesFrom2Frames(self, prev_frame, frame, pos_th, neg_th):

        deltas = np.array(np.array(prev_frame, dtype=np.int16)-np.array(frame, dtype=np.int16), dtype=np.int16)
        deltas = np.where(deltas == 1, 0, deltas)
        deltas = np.where(deltas == -1, 0, deltas)
        deltas = np.where(deltas >= pos_th, 1, deltas)
        deltas = np.where(deltas < -neg_th, -1, deltas)
        deltas = np.where(deltas > 1, 0, deltas)
        deltas = np.where(deltas < -1 , 0, deltas)
        
        return deltas

    def show_images(self, conn):
        name = "Stereo Camera Simulator"
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
            # cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(name, frame)
            cv2.waitKey(1)
            time_after = time.time()
            fps = "FPS: "+str(1//(time_after-time_before))
            time_before = time_after
            
    
    def show_depths(self, conn):
        name = "Stereo Depth Map"
        cv2.startWindowThread()
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
       
        while True:
            
            image = conn.recv()
            frame = np.array(list(image.image_data), dtype = np.uint8)
            frame = frame.reshape((image.height*2, image.width, 3))
            left = frame[:image.height]
            right = frame[image.height:]
            frame = np.concatenate([left, right], axis=1)
            frame *= 2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.flip(frame, 0)

            cv2.imshow(name, frame)
            cv2.waitKey(1)
           

    def show_spikes(self, conn, sim_params):
        name = "Stereo Event Camera Simulator"
        prev_frame = None
        frame = None
        colored_frame = None
        cv2.startWindowThread()
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        while True:
                    
            pos_th = sim_params[0]
            neg_th = sim_params[1]
            is_running = sim_params[2]
        
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
                spikes_frame = self.getSpikesFrom2Frames(prev_frame, frame, pos_th, neg_th).flatten()
                shape = [int(x) for x in spikes_frame.shape]
                colored_frame = np.zeros(shape=shape+[3], dtype="uint8")
                colored_frame[spikes_frame==1] = [255, 0, 0] 
                colored_frame[spikes_frame==-1] = [0, 0, 255] 
                colored_frame = colored_frame.reshape(src_shape)
                cv2.imshow(name, colored_frame)
                cv2.waitKey(1)
                        
            prev_frame = frame.copy()
            del image
    def save_images_as_pkls(self, conn, sim_params):
        while sim_params[0]==-1:
            pass
           
        dir_name = "pickle_output"
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open("{}/{}_{}_".format(dir_name, sim_params[0],sim_params[1])+datetime.now().strftime("%d%m%Y_%H%M%S"), 'wb') as f:        
            while True:
                image = conn.recv()
                if sim_params[3]==1: 
                    pkl.dump(image, f, pkl.HIGHEST_PROTOCOL)
                del image
        
    
    def save_depths_as_pkls(self, conn, sim_params):
        while sim_params[0]==-1:
            pass
        dir_name = "pickle_depth_output"
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open("{}/{}_{}_".format(dir_name, sim_params[0],sim_params[1])+datetime.now().strftime("%d%m%Y_%H%M%S"), 'wb') as f:        
            while True:
                image = conn.recv()
                if sim_params[3]==1:  
                    pkl.dump(image, f, pkl.HIGHEST_PROTOCOL)
                del image

        

class ZedStreamerServicer(ZedStreamer_pb2_grpc.ZedStreamerServicer):
    def __init__(self, sim_params, images=None, conn1=None, conn2=None,  conn3=None, conn4=None, conn5=None, conn6=None) -> None:
        self.images = images
        self.conn1 = conn1
        self.conn2 = conn2
        self.conn3 = conn3
        self.conn4 = conn4
        self.conn5 = conn5
        self.conn6 = conn6
        self.sim_params = sim_params

    

    async def SendVideo(self, request_iterator: AsyncIterable[
        ZedStreamer_pb2.Image], unused_context) -> ZedStreamer_pb2.Received:
        count = 0
        async for image in request_iterator:
            if self.conn1 is not None:
                self.conn1.send(image)
            if self.conn2 is not None:
                self.conn2.send(image)
            if self.conn5 is not None:
                self.conn5.send(image)

        return ZedStreamer_pb2.Received(ack=count)
    
    async def SendDepth(self, request_iterator: AsyncIterable[
        ZedStreamer_pb2.Depth], unused_context) -> ZedStreamer_pb2.Received:
        count = 0
        async for depth in request_iterator:
            if self.conn4 is not None:
                self.conn4.send(depth)
            if self.conn6 is not None:
                self.conn6.send(depth)
           

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

def startServer(images, sim_params, zed_conn, spikes_conn, params_parent_con, depths_parent_con, save_parent_con, save_depth_parent_con):
    servicer = ZedStreamerServicer(images, sim_params, zed_conn, spikes_conn, params_parent_con, depths_parent_con, save_parent_con, save_depth_parent_con)
    logging.basicConfig(level=logging.INFO)
    asyncio.get_event_loop().run_until_complete(serve(servicer))





#global variables shared memory among all processes

manager = mp.Manager()
zed_images = manager.list()
sim_params = manager.list()


def run():
    # list of global params
    sim_params.append(-1)
    sim_params.append(-1)
    sim_params.append(-1)
    sim_params.append(-1)
   
    # handles data from server
    handler = SimHandler()

    #pass data from server to processes in pipes
    zed_parent_con, zed_child_con = mp.Pipe()
    spikes_parent_con, spikes_child_con = mp.Pipe()
    params_parent_con, params_child_con = mp.Pipe()
    depths_parent_con, depths_child_con = mp.Pipe()
    save_parent_con, save_child_con = mp.Pipe()
    save_depth_parent_con, save_depth_child_con = mp.Pipe()

    # this process is the actual server that listens to unity
    
    p0 = mp.Process(target=startServer, args=(zed_images, sim_params, zed_parent_con, spikes_parent_con, params_parent_con, depths_parent_con, save_parent_con, save_depth_parent_con))
    
    # data from unity passed by the server to handler functions
    
    p1 = mp.Process(target=handler.show_images, args=(zed_child_con,))
    p2 = mp.Process(target=handler.show_spikes, args=(spikes_child_con, sim_params))
    p3 = mp.Process(target=handler.set_params, args=(params_child_con,sim_params))
    p4 = mp.Process(target=handler.show_depths, args=(depths_child_con,))
    p5 = mp.Process(target=handler.save_images_as_pkls, args=(save_child_con, sim_params))
    p6 = mp.Process(target=handler.save_images_as_pkls, args=(save_depth_child_con, sim_params))
    
    #start processes
    
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
   
    #join all
    
    p0.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
   

if __name__ == "__main__":
    run()