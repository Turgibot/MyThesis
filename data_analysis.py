import pickle as pkl
from turtle import shape
import cv2
import time
from matplotlib.pyplot import title
import numpy as np
import pathlib
from datetime import datetime
from collections import defaultdict, OrderedDict
import glob
import os

class Analyzer:
    def __init__(self, in_folder=None, out_folder=None, file=None, unity=None, stereo=False) -> None:
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.file = file
        self.unity = unity
        self.stereo = stereo
    
    def play(self):
        name = "Stereo Camera Simulator"
        cv2.startWindowThread()
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        pathlib.Path(self.unity).mkdir(parents=True, exist_ok=True)
        path = self.in_folder+"/"+self.file
        prev_ts = None
        ts = 0
        counter = 0
        with open(path, 'rb') as f:
            while True:
                try:
                    image = pkl.load(f)
                except:
                    print("End of recordings")
                    break
                counter+=1
                if counter <= 3:
                    continue
                if prev_ts is None:
                    prev_ts = image.timestamp
                    start_ts = prev_ts
                frame = np.array(list(image.image_data), dtype = np.uint8)
                frame = frame.reshape((image.height*2, image.width, 3))
                left = frame[:image.height]
                right = frame[image.height:]
                frame = np.concatenate([left, right], axis=1)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 0)
                # cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
                if not self.stereo:
                    frame = frame[:, :image.width]
                cv2.imshow(name, frame)
                if cv2.waitKey(25) == 27:
                    break
                curr_ts = image.timestamp
                dt = curr_ts - prev_ts
                prev_ts = curr_ts
                ts += dt
                cv2.imwrite(self.unity+"/"+str(ts/10**10)+'.jpg', frame)
            cv2.destroyAllWindows()

    def create_spikes(self):
        pathlib.Path(self.out_folder).mkdir(parents=True, exist_ok=True)
        input_path = self.in_folder+"/"+self.file
        output_path = self.out_folder+"/"+self.file
        pos_th = int(self.file.split('_')[0])
        neg_th = int(self.file.split('_')[1])

        with open(output_path+"-spikes", 'wb') as write_file:        
            with open(input_path, 'rb') as read_file:
                
                spikes = defaultdict(list)
                image = None
                curr_frame = None
                prev_frame = None
                start_time = 0
                prev_time = 0
                title = None
                counter = 0
                while True:
                    try:
                        image = pkl.load(read_file)
                    except:
                        print("Done creating spikes")
                        break
                    counter+=1
                    if counter <= 3:
                        continue    
                    curr_frame = np.array(list(image.image_data), dtype = np.uint8)
                    curr_frame = curr_frame.reshape((image.height*2, image.width, 3))
                    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                    left = curr_frame[:image.height]
                    right = curr_frame[image.height:]
                    curr_frame = np.concatenate([left, right], axis=1)
                    curr_frame = cv2.flip(curr_frame, 0)
                    if not self.stereo:
                        curr_frame = curr_frame[:, :image.width]
                    if prev_frame is None:
                        prev_frame = curr_frame
                        start_time = image.timestamp
                        prev_time = start_time
                        continue

                    curr_time = image.timestamp
                    width = image.width
                    height = image.height     
                    delta_time = curr_time - prev_time
                    delta_frame = np.array(np.array(prev_frame, dtype=np.int16)-np.array(curr_frame, dtype=np.int16), dtype=np.int16)

                    prev_frame = curr_frame
                    
  
                    if delta_frame.max() == delta_frame.min():
                        prev_time = curr_time
                        continue

                    for x in range(delta_frame.shape[0]):
                        for y in range(delta_frame.shape[1]):
                            if title is None:
                                if self.stereo:
                                    title = (width*2, height)
                                else:
                                    title = (width, height)
                                spikes[-1].append(title)
                            p = 1 if delta_frame[x,y]>=pos_th else -1 if delta_frame[x,y]<= -neg_th else 0
                            if p == 0: continue
                            th = pos_th if p ==1 else neg_th 
                            res = abs(delta_frame[x,y])//th 
                           
                            for i in range(res):
                                addition = (i/res)*delta_time
                                ts = prev_time+addition-start_time
                                # ts = ts%10**11
                                ts /= 10**10
                                spike = (ts, x,y,p)
                                # spike = "%9.6f %d %d %d\n"%(ts, x,y,p)
                                spikes[ts].append(spike)

                    prev_time = curr_time
                    sorted_spikes = OrderedDict(sorted(spikes.items()))

                    pkl.dump(sorted_spikes, write_file, pkl.HIGHEST_PROTOCOL)
                    del image
                    del sorted_spikes
                    del spikes
                    spikes = defaultdict(list)

                    print("Done ", counter)

    def show_text(self):
        path = self.out_folder+"/"+self.file
        counter = 0
        with open(path, 'rb') as f:
            while True:
                title = None
                try:
                    spikes = pkl.load(f)
                except:
                    print("End of recordings or unsuccessful read")
                    break
                keys = list(spikes.keys())
                if title is None:
                    title = spikes[keys[0]]
                    print(title)
                for i in range(1, len(keys)):
                    frame = keys[i]
                    for spike in spikes[frame]:
                        print(spike)
                # for text in T.txt:
                #     print(text)
                counter += 1
        print(counter)
            
    def create_txt(self):
        try:
            read_path = self.out_folder+"/"+self.file+"-spikes"
            write_path = read_path+".txt"
            counter = 0
            with open(read_path, 'rb') as f:
                with open(write_path, 'w') as w:
                    while True:
                        title = None
                        try:
                            spikes = pkl.load(f)
                        except:
                            print("Done exporting spikes in txt format")
                            break
                        keys = list(spikes.keys())
                        if title is None:
                            title = spikes[keys[0]]
                            w.write(str(title[0][0])+" "+str(title[0][1])+"\n")
                        for i in range(1, len(keys)):
                            frame = keys[i]
                            for spike in spikes[frame]:
                                str_txt = str(spike[0])+" "+str(spike[2])+" "+str(spike[1])+" "+str(spike[3])+"\n"
                                w.write(str_txt)
                                
            with open(write_path, "r") as f:
                lines = f.readlines()
            with open(write_path, "w") as f:
                f.write(lines[0])
                for line in lines:
                    if len(line.split())==4:
                        f.write(line)            
        except:
            print("Failed creating text for "+write_path)
       
    def show_spikes(self):
        try:
            read_path = self.out_folder+"/"+self.file+"-spikes.txt"
            name = "Event camera simulator"
            with open(read_path, 'r') as f:
                lines = f.readlines()
            params = lines[0].split()
            width = int(params[0])
            height = int(params[1])
            period = 0.02
            increment = period
            mat = np.zeros(shape=(height,width, 3), dtype=np.uint8)
            for i in range(1, len(lines)):
                try:
                    data = lines[i].split(" ")
                    if len(data) == 4:
                        ts = float(data[0])
                        x = int(data[2])
                        y = int(data[1])
                        p = int(data[3])
                        color = [255, 0, 0] if p==1 else [0, 0, 255] if p ==-1 else [0, 0, 0] 
                        mat[x][y] = color
                    else:
                        print(i)
                except Exception as e:
                    print(x, y)
                    print("shape",mat.shape)
                    print(e)
                if ts >= period:
                    cv2.imshow(name, mat)
                    if cv2.waitKey(20) == 27:
                        break
                    mat[:,:] = [0,0,0]
                    period += increment
                
            cv2.destroyAllWindows()
            print("End of the fireworks show")
        except:
            print("Failed reading fireworks text for "+read_path)
       
    def wrap_it_up_as_h5(self):
        list_of_files = glob.glob(self.out_folder+'/*.txt') 
        txt_file = max(list_of_files, key=os.path.getctime).split('/')[1]
        list_of_images = glob.glob(self.unity+'/*.jpg') 
        pass


if __name__ == "__main__":
    in_folder = "pickle_output"
    out_folder =  "spikes_output"
    unity = "unity_images"
    list_of_files = glob.glob(in_folder+'/*') 
    counter = 0
    # file = max(list_of_files, key=os.path.getctime).split('/')[1]
    for path in list_of_files:
        counter+=1
        if counter < 13:
            continue
        file = path.split('/')[1]
        analizer = Analyzer(in_folder=in_folder,out_folder=out_folder, unity=unity+"/"+str(counter), file=file, stereo=False)
        analizer.play()
        analizer.create_spikes()
        analizer.create_txt()
        analizer.show_spikes()
        
        