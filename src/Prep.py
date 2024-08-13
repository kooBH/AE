import os,glob
import numpy as np
import torch
from tqdm.auto import tqdm
from lip.extract import LipExtractor
from moviepy.editor import *
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--dir_in','-i',type=str,required=True)
parser.add_argument('--dir_out','-o',type=str,required=True)
args = parser.parse_args()

dir_LRS = args.dir_in
dir_out = args.dir_out

# list items
list_data = glob.glob(os.path.join(dir_LRS,"**","*.mp4"),recursive=True)
#print(len(list_data))


extractor = LipExtractor("cuda:1")
extractor.face_detector.net.share_memory()
extractor.face_detector.net.eval()

def process_data(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vid = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vid.append(frame)
    vid = np.array(vid)
    cap.release()

    seq = extractor(vid)
    return seq

from PIL import Image as im 




def process(idx) : 
    path = list_data[idx]
    print(path)
    dir_item = path.split("/")[-2]
    name_item = (path.split("/")[-1]).split(".")[0]

    try :
        crop = process_data(path)
    except OverflowError : 
        print("OverflowError")
        return
    #data = im.fromarray(crop[0]) 
    #data.save('gfg_dummy_pic.png') 

    crop = torch.from_numpy(crop)
    crop=crop.float

    os.makedirs(os.path.join(dir_out,dir_item),exist_ok=True)
    path_out = os.path.join(dir_out,dir_item,name_item+".pt")
    torch.save(crop,path_out)

def process_mp(batch) : 
    for idx in batch : 
        with torch.no_grad():
            process(idx)

"""
https://pytorch.org/docs/stable/notes/multiprocessing.html
#from multiprocessing import Pool, cpu_count
"""
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method

if __name__=='__main__': 
    set_start_method('spawn')
    num_process = 18
    processes = []
    batch_for_each_process = np.array_split(range(len(list_data)),num_process)

    for worker in range(num_process):
        p = mp.Process(target=process_mp, args=(batch_for_each_process[worker][:],) )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()






