import os,glob
import numpy as np
import torch
from tqdm.auto import tqdm
from lip.extract import LipExtractor
from moviepy.editor import *

dir_LRS = "/home/data/kbh/LRS3/trainval"
dir_out = "/home/data/kbh/lip/LRS3/trainval"

dir_LRS = "/home/data/kbh/LRS3/test"
dir_out = "/home/data/kbh/lip/LRS3/test"

# list items
list_data = glob.glob(os.path.join(dir_LRS,"**","*.mp4"),recursive=True)
print(len(list_data))

os.makedirs(dir_out,exist_ok=True)

extractor = LipExtractor("cuda:1")

def process_data(path):
    video = VideoFileClip(path)
    vid = []
    for frame in video.iter_frames() :
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vid.append(frame)
    vid = np.array(vid)
    seq = extractor(vid)
    return seq

from PIL import Image as im 

for path in tqdm(list_data):
    dir_item = path.split("/")[-2]
    name_item = (path.split("/")[-1]).split(".")[0]

    try :
        crop = process_data(path)
    except OverflowError : 
        print("OverflowError")
        continue
    #data = im.fromarray(crop[0]) 
    #data.save('gfg_dummy_pic.png') 

    crop = torch.from_numpy(crop)
    crop=crop.float

    os.makedirs(os.path.join(dir_out,dir_item),exist_ok=True)
    path_out = os.path.join(dir_out,dir_item,name_item+".pt")
    torch.save(crop,path_out)






