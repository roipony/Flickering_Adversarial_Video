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

import pre_process_rgb_flow as img_tool
import kinetics_i3d_utils as ki3du



scope ='RGB'

# sess = tf.Session(config  = tf_config)
# with tf.variable_scope(scope):
#     rgb_input = tf.placeholder(tf.float32,
#     shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
# rgb_model = load_i3d_model(num_classes=400)
# init_model(model=rgb_model,sess=sess, ckpt_path='data/checkpoints/rgb_imagenet/model.ckpt',eval_type='rgb')

# model_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
# softmax = tf.nn.softmax(logits = model_logits)

# f, of, = video_to_image_and_of('/home/ROIPO/Downloads/y2mate.com - utsa_track_devon_bond_triple_jump_67R1t4-b1nw_360p.mp4',n_steps=90)

# videos_base_path = sys.argv[1]
# class_name = sys.argv[2]
# tf_dst_folder = sys.argv[3]



videos_base_path = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/'
class_name ='all'
tf_dst_folder = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/val/'



if class_name=='all':
    classes_list =listdir(videos_base_path)
else:
    classes_list =[class_name]


if not os.path.exists(tf_dst_folder):
    os.makedirs(tf_dst_folder)
# video_list_path = [x.strip() for x in open('/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val_path_list.txt')]#'/media/ROIPO/Data/projects/Adversarial/kinetics-downloader/dataset/train/triple_jump/'
# # video_list_path = listdir(video_base_path)
# label_map_path='/data/DL/Adversarial/kinetics-i3d/data/label_map.txt'
# random.shuffle(video_list_path)
# class_target = ['juggling balls']
# class_target = ['juggling balls']
n_frames = ki3du._SAMPLE_VIDEO_FRAMES

kinetics_classes =ki3du.load_kinetics_classes()

#
# video_list = [x.strip() for x in open(vl)]


for c in classes_list:
    videos_list = os.path.join(videos_base_path,c)
    video_list_path=glob.glob(pp + '*/*.mp4')
    k=0
    for i,v in enumerate(video_list_path):
        if i%100 ==0:
            if i>0:
                writer.close()
                k+=1
            train_filename = os.path.join(tf_dst_folder,c,'kinetics_{}_{:04}.tfrecords'.format(c, k) ) # address to save the TFRecords file
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
        feature = {'train/label': _int64_feature(cls_id),
                   'train/video': _bytes_feature(frames.tobytes())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

writer.close()


        # video_path= ['/media/ROIPO/Data/projects/Adversarial/database/UCF-101/video/v_BreastStroke_g01_c01.avi',
        # '/media/ROIPO/Data/projects/Adversarial/database/UCF-101/video/v_BreastStroke_g01_c02.avi']
        #



    # for i, vp in enumerate(video_path):
    #     frames, flow_frames = video_to_image_and_of(video_path=vp,n_steps=80)
    #     feature = {'train/label': _int64_feature(1),
    #                'train/video': _float_list_feature(frames.reshape([-1]))}
    #     example = tf.train.Example(features=tf.train.Features(feature=feature))
    #     writer.write(example.SerializeToString())
    

    # sys.stdout.flush()
    
    #     frames_list.append(frames)
    # np.save('data/db.npy', frames_list)
