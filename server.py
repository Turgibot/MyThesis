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
    def __init__(self, images=None, conn=None) -> None:
        self.images = images
        self.conn = conn

    

    async def SendVideo(self, request_iterator: AsyncIterable[
        ZedStreamer_pb2.Image], unused_context) -> ZedStreamer_pb2.Received:
        count = 0
        async for image in request_iterator:
            self.conn.send(image)

        return ZedStreamer_pb2.Received(ack=count)

    async def SendImage(self, request, context) -> ZedStreamer_pb2.Received:
        frame = np.array(list(request.image_data), dtype = np.uint8)
        frame = frame.reshape((request.height, request.width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.conn.send(frame)

        return ZedStreamer_pb2.Received(ack=True)



async def serve(servicer) -> None:
    server = grpc.aio.server()
    ZedStreamer_pb2_grpc.add_ZedStreamerServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()
    cv2.destroyAllWindows()

def startServer(images, parent_con):
    servicer = ZedStreamerServicer(images, parent_con)
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

# use a pipe to transfer image stream
def run():
    images = []
    parent_con, child_con = mp.Pipe()
    p1 = mp.Process(target=startServer, args=(images, parent_con))
    p2 = mp.Process(target=show_images, args=(child_con,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == "__main__":
    run()