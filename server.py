import asyncio
import logging
import time
from typing import AsyncIterable, Iterable

import grpc
import ZedStreamer_pb2 as ZedStreamer_pb2
import ZedStreamer_pb2_grpc as ZedStreamer_pb2_grpc
import numpy as np
import cv2
import multiprocessing as mp

class ZedStreamerServicer(ZedStreamer_pb2_grpc.ZedStreamerServicer):
    def __init__(self, images=None, conn1=None, conn2=None) -> None:
        self.images = images
        self.conn1 = conn1
        self.conn2 = conn2

    

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
        frame = np.array(list(request.image_data), dtype = np.uint8)
        frame = frame.reshape((request.height, request.width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.conn1.send(frame)

        return ZedStreamer_pb2.Received(ack=True)



async def serve(servicer) -> None:
    server = grpc.aio.server()
    ZedStreamer_pb2_grpc.add_ZedStreamerServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()
    cv2.destroyAllWindows()

def startServer(images, zed_conn, spikes_conn):
    servicer = ZedStreamerServicer(images, zed_conn, spikes_conn)
    logging.basicConfig(level=logging.INFO)
    asyncio.get_event_loop().run_until_complete(serve(servicer))

def show_images(conn):
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

def show_spikes(conn):
    name = "DVS Stereo Camera Simulator"
    prev_frame = None
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
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.flip(frame, 0)
        if prev_frame is not None:
            spikes_frame = getSpikesFrom2Frames(prev_frame, frame)
            cv2.putText(spikes_frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(name, spikes_frame)
            cv2.waitKey(1)
        prev_frame = frame.copy()
        time_after = time.time()
        fps = "FPS: "+str(1//(time_after-time_before))
        time_before = time_after

def getSpikesFrom2Frames(prev_frame, frame):
    th=60
    deltas = np.array(prev_frame)-np.array(frame)
    deltas = np.where(deltas>=th, 255, deltas)
    deltas = np.where(deltas<th, 0, deltas)

    return deltas

# use a pipe to transfer image stream
def run():
    zed_images = []
    zed_parent_con, zed_child_con = mp.Pipe()
    spikes_parent_con, spikes_child_con = mp.Pipe()

    p0 = mp.Process(target=startServer, args=(zed_images, zed_parent_con, spikes_parent_con))
    p1 = mp.Process(target=show_images, args=(zed_child_con,))
    p2 = mp.Process(target=show_spikes, args=(spikes_child_con,))

    p0.start()
    p1.start()
    p2.start()
   
    p0.join()
    p1.join()
    p2.join()
   

if __name__ == "__main__":
    run()