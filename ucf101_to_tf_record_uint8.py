import cv2
import numpy as np
import imageio
import tensorflow as tf
import i3d

import random
import skvideo.io
import sys
import os

import setGPU
from os import listdir
import glob
sys.path.insert(1, os.path.realpath(os.path.pardir))

import utils.pre_process_rgb_flow as img_tool
import utils.kinetics_i3d_utils as ki3du

#%%
def main(argv, arc):

    videos_base_path = argv[1]
    class_name = argv[2]
    tf_dst_folder = argv[3]
    
    # import pdb
    # pdb.set_trace()
    # videos_base_path = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/'
    # class_name ='hula hooping'
    # tf_dst_folder = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/val/'
    
    
    
    if class_name=='all':
        classes_list =listdir(videos_base_path)
    else:
        classes_list =[class_name]
    
    
    if not os.path.exists(tf_dst_folder):
        os.makedirs(tf_dst_folder)
    
    n_frames = ki3du._SAMPLE_VIDEO_FRAMES
    
    kinetics_classes =ki3du.load_kinetics_classes()
    
    
    for c in classes_list:
        videos_list = os.path.join(videos_base_path,c)
        if not os.path.exists(videos_list):
            print('{} not exist'.format(videos_list))
            continue
        
        video_list_path=glob.glob(videos_list + '*/*.mp4')
        k=0
        for i,v in enumerate(video_list_path):
            if i%100 ==0:
                if i>0:
                    writer.close()
                    k+=1
                target_folder =os.path.join(tf_dst_folder,c)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    
                train_filename = os.path.join(target_folder,'kinetics_{}_{:04}.tfrecords'.format(c, k) ) # address to save the TFRecords file
                # open the TFRecords file
                writer = tf.python_io.TFRecordWriter(train_filename)
        
            cls =  c
            cls_id = kinetics_classes.index(cls)
            vid_path = v
            
            
            if os.path.exists(v)==False:
                continue
            try:
                
                frames = skvideo.io.vread(vid_path)
                # frames = frames.astype('float32') / 128. - 1
            except:
                os.remove(vid_path)
                continue
                
            if frames.shape[0] < ki3du._SAMPLE_VIDEO_FRAMES:
                continue
                #frames = np.pad(frames, ((0, _SAMPLE_VIDEO_FRAMES-frames.shape[0]),(0,0),(0,0),(0,0)),'wrap')
            else:
                frames=frames[-ki3du._SAMPLE_VIDEO_FRAMES:]
                
            # prob = sess.run(softmax,feed_dict={rgb_input:frames})
            # top_id = prob.argmax()
            # if cls_id!=top_id:
            #     continue
            feature = {'train/label': img_tool._int64_feature(cls_id),
                       'train/video': img_tool._bytes_feature(frames.tobytes())}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    
        writer.close()

#%%
def main_random_videos(database_path,video_list,class_name, tf_dst_folder):

    # videos_base_path = argv[1]
    # class_name = argv[2]
    # tf_dst_folder = argv[3]
    # class_name = 'all'
    # database_path='/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/'
    # tf_dst_folder='/home/ubadmin/pony/database/Kinetics/tfrecord_uint8/val/all_cls_shuffle/'
    database_path='/data/DL/Adversarial/database/UCF-101/video/'
    tf_dst_folder='/home/ubadmin/pony/database/UCF101/tfrecord_uint8/train/all_cls_shuffle/'
    class_name='all'
    video_list='/data/DL/Adversarial/database/UCF-101/ucfTrainTestlist/trainlist01.txt'

    num_videos_in_single_tfrecord=50
    
    # import pdb
    # pdb.set_trace()
    # videos_base_path = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/'
    # class_name ='hula hooping'
    # tf_dst_folder = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/val/'
    
    
    
    # if class_name=='all':
    #     classes_list =listdir(videos_base_path)
    # else:
    #     classes_list =[class_name]
    
    
    if not os.path.exists(tf_dst_folder):
        os.makedirs(tf_dst_folder)
    
    n_frames = 40
    
    ucf_classes = [x.strip() for x in open('/data/DL/Adversarial/kinetics-i3d/data/label_map_ucf_101.txt')]

    # ucf =ki3du.load_kinetics_classes()
    videos_list_path=[x.strip() for x in open(video_list)]
    
    # videos_list_path=glob.glob(database_path + '*/*.avi')
    random.shuffle(videos_list_path)
           
        
    k=0
    i=0
#%%
    for v in videos_list_path:
        if i%num_videos_in_single_tfrecord ==0:
            if k>0:
                writer.close()
                
            target_folder =os.path.join(tf_dst_folder)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                
            train_filename = os.path.join(target_folder,'UCF101_N_{}_{:04}.tfrecords'.format(num_videos_in_single_tfrecord, k) ) # address to save the TFRecords file
            # open the TFRecords file
            writer = tf.python_io.TFRecordWriter(train_filename)
            k+=1
            
        gt_cls_id = v.split(' ')[-1]
        vid_path=v.split(' ')[0]
        
        cls =  vid_path.split('/')[0]
        cls_id = ucf_classes.index(cls)
        
        vid_name=vid_path.split('/')[1]
        assert(cls_id ==int(gt_cls_id)-1)
        # vid_path = v
        video_full_path = os.path.join(database_path,vid_name)
        
        if os.path.exists(video_full_path)==False:
            continue
        # try:
        frames,_ = img_tool.video_to_image_and_of(video_full_path,target_fps=25,
                                            n_steps=40)
            # frames = skvideo.io.vread(vid_path)
            # frames = frames.astype('float32') / 128. - 1
        # except:
        #     os.remove(vid_path)
        #     continue
            
        # if frames.shape[0] < 40:
        #     continue
        #     #frames = np.pad(frames, ((0, _SAMPLE_VIDEO_FRAMES-frames.shape[0]),(0,0),(0,0),(0,0)),'wrap')
        # else:
        #     frames=frames[-40:]
            
        # prob = sess.run(softmax,feed_dict={rgb_input:frames})
        # top_id = prob.argmax()
        # if cls_id!=top_id:
        #     continue
        feature = {'train/label': img_tool._int64_feature(cls_id),
                   'train/video': img_tool._bytes_feature(frames.tobytes())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        i+=1

    writer.close()
#%%

if __name__ == '__main__':
    # main(sys.argv, len(sys.argv))
    
    
    database_path='/data/DL/Adversarial/database/UCF-101/video/'
    tfrecord_dist_path='/home/ubadmin/pony/database/UCF101/tfrecord_uint8/test/all_cls_shuffle/'
    video_list='/data/DL/Adversarial/database/UCF-101/ucfTrainTestlist/trainlist01.txt'
    main_random_videos(database_path,video_list,'all',tfrecord_dist_path)