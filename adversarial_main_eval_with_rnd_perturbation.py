# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=''
import numpy as np
import setGPU
import matplotlib
# matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import imageio
import pickle
import glob
import tensorflow as tf
import time
import re
# tf.enable_eager_execution()
# sys.path.insert(1, '../Adversarial/kinetics-i3d/')

sys.path.insert(1, os.path.realpath(os.path.pardir))


# import skvideo
from utils import pre_process_rgb_flow as img_tool
from utils import kinetics_i3d_utils as ki3du


_IMAGE_SIZE = 224
_BATCH_SIZE = 8

_SAMPLE_VIDEO_FRAMES =90 #90 #79 # 79 #90 #90 #250 #90 #79
_BASE_PATCH_FRAMES = _SAMPLE_VIDEO_FRAMES #_SAMPLE_VIDEO_FRAMES #_SAMPLE_VIDEO_FRAMES # 1# _SAMPLE_VIDEO_FRAMES # 1:for sticker _SAMPLE_VIDEO_FRAMES # 1
_IND_START = 0  # 0 #50
_IND_END =_SAMPLE_VIDEO_FRAMES



#%% model loader
ckpt_path = '/data/DL/Adversarial/kinetics-i3d/result/generalization/universal/val_test/all_cls_shuffle_t15000_v2000_linf_0.15_lambda_1.0_beta1_0.5'
ckpt_last = tf.train.latest_checkpoint(checkpoint_dir=ckpt_path)
linf=float(re.search('linf_(.+?)_', ckpt_path).group(1))
model = ki3du.kinetics_i3d(ckpt_path=ckpt_last,
                           batch_size=None,
                           init_pert_from_ckpt=True,
                           inf_norm=linf)

sess=model.sess
inputs = model.rgb_input
adv_flag = model.adv_flag
cyclic_flag = model.cyclic_flag
softmax=model.softmax

perturbation_clip = model.eps_rgb_clip
perturbation = model.eps_rgb

pert = sess.run(perturbation)
pert_clip = sess.run(perturbation_clip)
#%% dataset loader

tf_records_list_path_val =  '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/test/all_cls_shuffle/'
tf_record_list_val = sorted(glob.glob(tf_records_list_path_val+ '*.tfrecords'))[-30:-1]

# tf_record_list_train = [x.strip() for x in open(tf_records_list_path_train)]
dataset_val = tf.data.TFRecordDataset(filenames=tf_record_list_val, num_parallel_reads=os.cpu_count())
dataset_val = dataset_val.prefetch(buffer_size=_BATCH_SIZE*5)
dataset_val = dataset_val.batch(_BATCH_SIZE,drop_remainder=True)
dataset_val = dataset_val.map(img_tool.parse_example_uint8, num_parallel_calls=os.cpu_count())
# dataset_train = dataset_train.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
    # dataset = dataset.shuffle(buffer_size=buffer_size)

# Transformation
# dataset_train = dataset_train.apply(tf.contrib.data.map_and_batch(img_tool.parse_example, batch_size=_BATCH_SIZE,drop_remainder=True,num_parallel_calls=20))
dataset_val =dataset_val.prefetch(1)
iterator_val = dataset_val.make_initializable_iterator()
next_element_val = iterator_val.get_next()


#%%
# int random perturbation -uniform
def get_random_uniform_pert(pert):
    pert_rnd = np.random.uniform(pert.min(), pert.max(), pert.shape).astype(np.float32)
    return pert_rnd

# int random perturbation - not uniforn only min/max
def get_random_min_max_pert(pert):
    pert_rnd = np.random.choice([pert.min(), pert.max()], size=pert.shape, p=[1. / 2, 1. / 2]).astype(np.float32)
    return pert_rnd


# init random perturbation - shuffle the original (images axis)
def get_random_shuffle_along_images_pert(pert):
    pert_rnd = pert.copy()
    np.apply_along_axis(np.random.shuffle,0,pert_rnd)
    return  pert_rnd

# init random perturbation - shuffle the original (all axis)
def get_random_shuffle_along_all_pert(pert):
    pert_rnd = pert.copy()
    pert_rnd =pert_rnd.reshape([-1])
    np.random.shuffle(pert_rnd)
    pert_rnd= pert_rnd.reshape(pert.shape)
    return pert_rnd

random_pert_handler = {'random_shuffle_along_images': get_random_shuffle_along_images_pert,
                    'random_uniform_pert': get_random_uniform_pert,
                    'random_min_max_pert': get_random_min_max_pert,
                    'random_shuffle_along_all_pert':get_random_shuffle_along_all_pert}


#%%
num_of_sampes_each_pert = 12
rnd_pert_res_dict = {}
rnd_pert_res_dict['model_name'] = 'I3D'
rnd_pert_res_dict['original_perturbation'] = pert_clip


miss_rate=0
total_val_vid=0
sess.run(iterator_val.initializer)
valid_videos_l=[]
ii=0
try: 
    # Keep running next_batch till the Dataset is exhausted
    while True:
        
        start = time.perf_counter()
        rgb_sample, sample_label = sess.run(next_element_val)
        end=time.perf_counter()
        # print("load_data_time: {:.5f}".format(end-start))
        
        # p_rgb_sample = rgb_sample +pert_clip
                
        # rgb_sample=rgb_sample[:,-_SAMPLE_VIDEO_FRAMES:,...]
        feed_dict_for_adv_eval = {inputs: rgb_sample, adv_flag:1, cyclic_flag:1}
        prob = sess.run(feed_dict=feed_dict_for_adv_eval, fetches=softmax)
        prob_clean = sess.run(feed_dict= {inputs: rgb_sample, adv_flag: 0, cyclic_flag:0}, fetches=softmax)
        valid_videos = prob_clean.argmax(axis=-1)==sample_label
        valid_videos_l.append(valid_videos)
        miss_rate+=np.logical_and(valid_videos,(prob.argmax(axis=-1)!=sample_label)).sum()
        total_val_vid+=valid_videos.sum()
        
except tf.errors.OutOfRangeError:
    
    print("fool_rate: {:.5f}".format(miss_rate/total_val_vid) )
    pass


print('universal perturbation fooling ratio: {:.4f}'.format(miss_rate/total_val_vid))
universal_miss_ratio= miss_rate/total_val_vid
rnd_pert_res_dict['original_perturbation_fr']=universal_miss_ratio

#%%
dst_path ='/home/ubadmin/DL_shared/Adversarial/computervision-recipes/results/universal_vs_rnd/norm_inf_{0:g}/rnd_perturbation_eval_I3D.npy'.format(linf)


rnd_pert_type_list = ['random_min_max_pert', 'random_shuffle_along_images', 'random_shuffle_along_all_pert', 'random_uniform_pert']
for rnd_pert_type in rnd_pert_type_list:
    
    print("random pert type: {}".format(rnd_pert_type))
    fooling_ratio_l =[]
    rnd_pert_l=[]
    rnd_pert_thick_l = []
    rnd_pert_rough_l =[]
    for jj in range(num_of_sampes_each_pert):
        tt = time.time()
        print("iter {} out of {} ..".format(jj,num_of_sampes_each_pert))
        rnd_pert = random_pert_handler[rnd_pert_type](pert_clip)
        perturbation_assign_op = perturbation.assign(rnd_pert)
        sess.run(perturbation_assign_op)
    
        rnd_pert_l.append(rnd_pert)
        rnd_pert_thick_l.append(np.mean(np.abs(rnd_pert)))
    
        rnd_pert_rough_l.append(np.mean(np.abs(rnd_pert - np.roll(rnd_pert, shift=1, axis=1))))
    
        miss_rate=0
        total_val_vid=0
        sess.run(iterator_val.initializer)
        ii=0
        try: 
            # Keep running next_batch till the Dataset is exhausted
            while True:
                
                start = time.perf_counter()
                rgb_sample, sample_label = sess.run(next_element_val)
                rgb_sample =    rgb_sample[valid_videos_l[ii]]
                sample_label= sample_label[valid_videos_l[ii]]
                ii+=1
                if not len(sample_label)>0:
                    continue
                
                end=time.perf_counter()
                # print("load_data_time: {:.5f}".format(end-start))
                
                # p_rgb_sample = rgb_sample +pert_clip
                        
                # rgb_sample=rgb_sample[:,-_SAMPLE_VIDEO_FRAMES:,...]
                feed_dict_for_adv_eval = {inputs: rgb_sample, adv_flag:1, cyclic_flag:1}
                prob = sess.run(feed_dict=feed_dict_for_adv_eval, fetches=softmax)
                prob_clean = sess.run(feed_dict= {inputs: rgb_sample, adv_flag: 0, cyclic_flag:0}, fetches=softmax)
                valid_videos = prob_clean.argmax(axis=-1)==sample_label
                miss_rate+=np.logical_and(valid_videos,(prob.argmax(axis=-1)!=sample_label)).sum()
                total_val_vid+=valid_videos.sum()
                
        except tf.errors.OutOfRangeError:
            print("fool_rate: {:.5f}".format(miss_rate/total_val_vid) )
            pass
        
        fooling_ratio_l.append(miss_rate/total_val_vid)
        
        print("fooling ratio for iter {}: {:.4f} total time for iter:{:.4f} Sec ".format(ii,fooling_ratio_l[-1],time.time()-tt))


    print("mean fooling ration: {:.2f} ({:.2f})".format(np.mean(fooling_ratio_l),np.std(fooling_ratio_l)))
    
    rnd_pert_res_dict[rnd_pert_type]={'fooling_ratio_l':fooling_ratio_l,
                                      'perturbations': rnd_pert_l,
                                      'perturbations/thickness':  rnd_pert_thick_l,
                                      'perturbations/roughness': rnd_pert_rough_l
                                      }
    
np.save(dst_path, rnd_pert_res_dict)



#%%
