#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 02:08:25 2020

@author: ubadmin
"""
import random
import skvideo.io
import sys
import numpy as np
import os

val_size=100    
train_size=10000   
_SAMPLE_VIDEO_FRAMES=90            
                       
list_train = [x.strip() for x in open(sys.argv[1])]
random.shuffle(list_train)


for ll in list_train:
    
    try:
        frames = skvideo.io.vread(ll)
        frames = frames.astype('float32') / 128. - 1
    except:
        os.remove(ll)
        continue
        
    if frames.shape[0] < _SAMPLE_VIDEO_FRAMES:
        frames = np.pad(frames, ((0, _SAMPLE_VIDEO_FRAMES-frames.shape[0]),(0,0),(0,0),(0,0)),'wrap')
    else:
        frames=frames[-_SAMPLE_VIDEO_FRAMES:]
    
    dst_path = ll.replace('.mp4','.npy')
    np.save(dst_path,frames)