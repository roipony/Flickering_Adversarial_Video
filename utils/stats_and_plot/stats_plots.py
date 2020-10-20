#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:59:22 2020

@author: ubadmin
"""
# %%
import os
import sys
import numpy as np
import glob
import pickle
from collections import namedtuple

sys.path.insert(0,'../../)')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.pre_process_rgb_flow import run_npy
from utils.pre_process_rgb_flow import run_mp4
import utils.pre_process_rgb_flow as img_tool

import seaborn as sbn
import tensorflow as tf

kinetics_classes = [x.strip() for x in open('data/label_map.txt')]
# runcell(9, '/data/DL/Adversarial/kinetics-i3d/stats_plots.py')

#%%
##
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# from tensorflow.python import pywrap_tensorflow

mp4_list = glob.glob('/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/'+'*/*.mp4')
np.random.shuffle(mp4_list)
mp4_list = mp4_list[:100]
for i, ff_path in enumerate(mp4_list):
    ff=img_tool.load_mp4(ff_path)
# ff=img_tool.load_mp4('/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/playing drums/D6IYvlf5t0U_000044_000054.mp4')
    
    imm=ff[int(0.85*ff.shape[0])]
    imm-=imm.min()
    imm/=imm.max()

    imm*=255.
    imm=imm.astype(np.uint8)
    plt.imsave('/home/ubadmin/Downloads/random_images/{:05d}.png'.format(i),imm)

# latest_ckpt_path = tf.train.latest_checkpoint('/data/DL/Adversarial/kinetics-i3d/result/generalization/universal/L12_loss/all_cls_shuffle_t5000_v2000_')
# reader = pywrap_tensorflow.NewCheckpointReader(latest_ckpt_path)
# pert_L12 = reader.get_tensor('RGB/eps')

# adv_vid_L12=np.clip(ff.copy()+pert_L12,-1,1)

# latest_ckpt_path = tf.train.latest_checkpoint('/data/DL/Adversarial/kinetics-i3d/result/generalization/universal/val_test/all_cls_shuffle_t1050_v2000_')
# reader = pywrap_tensorflow.NewCheckpointReader(latest_ckpt_path)
# pert_ours = reader.get_tensor('RGB/eps')

# ones =np.ones_like(ff)
# pert_ours_full =ones*pert_ours
# adv_vid_ours=np.clip(ff.copy()+pert_ours_full,-1,1)


# for i,imm in enumerate(adv_vid_L12):
#     imm-=imm.min()
#     imm/=imm.max()
#     imm*=255.
#     imm=imm.astype(np.uint8)
#     plt.imsave('/home/ubadmin/Downloads/pull_ups_L12_adv/{:05d}.png'.format(i),imm)

# for i,imm in enumerate(adv_vid_ours):
#     imm-=imm.min()
#     imm/=imm.max()

#     imm*=255.
#     imm=imm.astype(np.uint8)
#     plt.imsave('/home/ubadmin/Downloads/pull_ups_ours_adv/{:05d}.png'.format(i),imm)
    
# for i,imm in enumerate(ff):
#     imm-=imm.min()
#     imm/=imm.max()

#     imm*=255.
#     imm=imm.astype(np.uint8)
#     plt.imsave('/home/ubadmin/Downloads/pull_ups_clean/{:05d}.png'.format(i),imm)

#%%
##
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python import pywrap_tensorflow


vid_path = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/baby waking up/LEDQdL23lwg_000085_000095.mp4'
ff=img_tool.load_mp4(vid_path)
# ff=img_tool.load_mp4('/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val/playing drums/D6IYvlf5t0U_000044_000054.mp4')

ff=ff[-90:]

latest_ckpt_path = tf.train.latest_checkpoint('/data/DL/Adversarial/kinetics-i3d/result/generalization/universal/L12_loss/all_cls_shuffle_t5000_v2000_')
reader = pywrap_tensorflow.NewCheckpointReader(latest_ckpt_path)
pert_L12 = reader.get_tensor('RGB/eps')

adv_vid_L12=np.clip(ff.copy()+pert_L12,-1,1)

latest_ckpt_path = tf.train.latest_checkpoint('/data/DL/Adversarial/kinetics-i3d/result/generalization/universal/val_test/all_cls_shuffle_t1050_v2000_')
reader = pywrap_tensorflow.NewCheckpointReader(latest_ckpt_path)
pert_ours = reader.get_tensor('RGB/eps')

ones =np.ones_like(ff)
pert_ours_full =ones*pert_ours
adv_vid_ours=np.clip(ff.copy()+pert_ours_full,-1,1)

dst_path ='/home/ubadmin/ROIPO_PC/media/ROIPO/Data1/projects/Adversarial/EvadeML-Zoo/datasets/kinetics/adversarial/'

np.save(os.path.join(dst_path,'' ,adv_vid_L12)


for i,imm in enumerate(adv_vid_L12):
    imm-=imm.min()
    imm/=imm.max()
    imm*=255.
    imm=imm.astype(np.uint8)
    plt.imsave('/home/ubadmin/Downloads/pull_ups_L12_adv/{:05d}.png'.format(i),imm)

for i,imm in enumerate(adv_vid_ours):
    imm-=imm.min()
    imm/=imm.max()

    imm*=255.
    imm=imm.astype(np.uint8)
    plt.imsave('/home/ubadmin/Downloads/pull_ups_ours_adv/{:05d}.png'.format(i),imm)
    
for i,imm in enumerate(ff):
    imm-=imm.min()
    imm/=imm.max()

    imm*=255.
    imm=imm.astype(np.uint8)
    plt.imsave('/home/ubadmin/Downloads/pull_ups_clean/{:05d}.png'.format(i),imm)
#%%
##

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python import pywrap_tensorflow

results_path = '/data/DL/Adversarial/kinetics-i3d/result/generalization//universal/val_test/'

experiments = [ff for ff in os.listdir(results_path) if os.path.isdir(os.path.join(results_path,ff))]

fooling_ratio_l=[]
roughness_l=[]
thickness_l= []
num_of_train_vid=[]
pert_ours=[]

for exp in experiments:
    num_of_train_vid.append(int(exp.split('_')[-3].strip('t')))
    eval_summery_path =os.path.join(results_path, exp,'eval')
    event_acc = EventAccumulator(eval_summery_path)
    event_acc.Reload()
    fooling_ratio = 1.0 - event_acc.Scalars('ACC')[-1][-1]
    fooling_ratio_l.append(fooling_ratio)
    train_summery_path =os.path.join(results_path, exp)
    event_train = EventAccumulator(train_summery_path)
    event_train.Reload()
    roughness = event_train.Scalars('Perturbation/roughness___')[-1][-1]
    thickness = event_train.Scalars('Perturbation/thickness___')[-1][-1]
    
    thickness_l.append(thickness)
    roughness_l.append(roughness)
    
    latest_ckp = tf.train.latest_checkpoint(os.path.join(results_path, exp))
    reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
    pert = reader.get_tensor('RGB/eps')
    pert_ours.append(pert[np.newaxis])


pert_ours=np.concatenate(pert_ours,axis=0)
num_of_train_vid = np.array(num_of_train_vid) 
fooling_ratio_l = np.array(fooling_ratio_l)
num_of_train_vid_ours = np.append(num_of_train_vid,5000)
fooling_ratio_ours_l= np.append(fooling_ratio_l,0.925)
thickness_l.append(15.5)
roughness_l.append(15.7)
sorted_idx = np.argsort(fooling_ratio_ours_l)

fooling_ratio_ours_l=fooling_ratio_ours_l[sorted_idx]
num_of_train_vid_ours = num_of_train_vid_ours[sorted_idx]

thickness_l_ours =np.array(thickness_l)
thickness_l_ours = thickness_l_ours[sorted_idx]

roughness_l_ours =np.array(roughness_l)
roughness_l_ours = roughness_l_ours[sorted_idx]

results_path = '/data/DL/Adversarial/kinetics-i3d/result/generalization//universal/L12_loss/'

experiments = [ff for ff in os.listdir(results_path) if os.path.isdir(os.path.join(results_path,ff))]

fooling_ratio_l=[]
roughness_l=[]
thickness_l= []
num_of_train_vid=[]
pert_L12=[]


for exp in experiments:
    num_of_train_vid.append(int(exp.split('_')[-3].strip('t')))
    eval_summery_path =os.path.join(results_path, exp,'eval')
    event_acc = EventAccumulator(eval_summery_path)
    event_acc.Reload()
    fooling_ratio = 1.0 - event_acc.Scalars('ACC')[-1][-1]
    fooling_ratio_l.append(fooling_ratio)
    train_summery_path =os.path.join(results_path, exp)
    event_train = EventAccumulator(train_summery_path)
    event_train.Reload()
    roughness = event_train.Scalars('Perturbation/roughness___')[-1][-1]
    thickness = event_train.Scalars('Perturbation/thickness___')[-1][-1]
    
    thickness_l.append(thickness)
    roughness_l.append(roughness)
    
    latest_ckp = tf.train.latest_checkpoint(os.path.join(results_path, exp))
    reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
    pert = reader.get_tensor('RGB/eps')
    pert_L12.append(pert[np.newaxis])


pert_L12=np.concatenate(pert_L12,axis=0)
num_of_train_vid = np.array(num_of_train_vid) 
fooling_ratio_l = np.array(fooling_ratio_l)

sorted_idx = np.argsort(num_of_train_vid)

num_of_train_vid_L12=num_of_train_vid[sorted_idx]
fooling_ratio_L12_l = fooling_ratio_l[sorted_idx]

thickness_l_L12 =np.array(thickness_l)
thickness_l_L12 = thickness_l_L12[sorted_idx]

roughness_l_L12 =np.array(roughness_l)
roughness_l_L12 = roughness_l_L12[sorted_idx]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Num of videos (#)')
ax1.set_ylabel('Fooling ratio')
lns1 =ax1.semilogx(num_of_train_vid_ours,fooling_ratio_ours_l,'b',label='Flickering attack')

ax1.set_xlim(10)
lns2 =ax1.semilogx(num_of_train_vid_L12,fooling_ratio_L12_l,'g',label='SUP')

ax2 = ax1.twinx()

ax2.set_ylabel('[%]')
# ax1.tick_params(axis='y', labelcolor='g')
# ax2.tick_params(axis='y', labelcolor='b')


lns4= ax2.semilogx(num_of_train_vid_ours,thickness_l_ours, '--b', label='thickness_Flickering')
lns5= ax2.semilogx(num_of_train_vid_ours,roughness_l_ours, '-.b', label='roughness_Flickering')
lns6= ax2.semilogx(num_of_train_vid_L12,thickness_l_L12*100, '--g', label='thickness_SUP')
lns7= ax2.semilogx(num_of_train_vid_L12,roughness_l_L12*100, '-.g', label='roughness_SUP')


lns = lns1+lns2+lns4+lns5 +lns6 +lns7
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=4)
ax1.set_yticks(np.linspace(0,1,11))
ax1.grid(True)


# plt.plot(num_of_train_vid_ours,fooling_ratio_ours_l)
# plt.plot(num_of_train_vid_L12,fooling_ratio_L12_l)
# plt.xlabel('Num of videos')
# plt.ylabel('Fooling ratio')

    
#     ax1.tick_params(axis='y', labelcolor='g')
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
#     ax2.set_ylabel('MAP[%]', color='b') 
#     lns3 =ax2.plot(roughness, 'b',label='roughness' )
#     lns4=ax2.plot(thickness, '--b',label='thickness')
#     ax2.tick_params(axis='y', labelcolor='b')
    
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.show()
    
#     lns = lns1+lns2+lns3+lns4
#     labs = [l.get_label() for l in lns]
#     ax1.legend(lns, labs, loc=6)
#     ax2.grid(True)



#%%
##
roughness_np = np.array(roughness_l)[sorted_idx]
roughness_mu = roughness_np.mean(axis=0)
sigma_roughness = roughness_np.std(axis=0)

thickness_np = np.array(thickness_l)[sorted_idx]
thickness_mu = thickness_np.mean(axis=0)
sigma_thickness = thickness_np.std(axis=0)

MADP_cln_np = np.array(clean_vid_MADP_l)
MADP_cln_mu = MADP_cln_np.mean(axis=0)
sigma_MADP_cln = MADP_cln_np.std(axis=0)

MADP_adv_np = np.array(adv_vid_MADP_l)
MADP_adv_mu = MADP_adv_np.mean(axis=0)
sigma_MADP_adv = MADP_adv_np.std(axis=0)


#%%
##

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python import pywrap_tensorflow

results_path = '/data/DL/Adversarial/kinetics-i3d/result/generalization/single_class/'
allcls = [ff for ff in os.listdir(results_path) if os.path.isdir(os.path.join(results_path,ff))]

fooling_ratio_l=[]
roughness_l=[]
thickness_l= []

for cls in allcls:
    eval_summery_path =os.path.join(results_path, cls,'eval')
    event_acc = EventAccumulator(eval_summery_path)
    event_acc.Reload()
    fooling_ratio = 1.0 - event_acc.Scalars('ACC')[-1][-1]
    fooling_ratio_l.append(fooling_ratio)
    train_summery_path =os.path.join(results_path, cls)
    event_train = EventAccumulator(train_summery_path)
    event_train.Reload()
    roughness = event_train.Scalars('Perturbation/roughness__')[-1][-1]
    thickness = event_train.Scalars('Perturbation/thickness__')[-1][-1]
    
    thickness_l.append(thickness)
    roughness_l.append(roughness)
    # latest_ckp = tf.train.latest_checkpoint(os.path.join(results_path, cls))
    # reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
    # pert = reader.get_tensor('RGB/eps')



# leg=[]
# allfiles = os.listdir(results_path)
# files = [fname for fname in allfiles if fname.endswith('.pkl')]

# roughness_l=[]
# thickness_l=[]
# clean_vid_MADP_l=[]
# adv_vid_MADP_l=[]
# for i,res in enumerate(files):
    
#     with open(os.path.join(results_path+files[i]), 'rb') as handle:
#         tmp_dict = pickle.load(handle)
        
#     tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
#     tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())
#     roughness_l.append(tmp_struct.smoothness)
#     thickness_l.append(tmp_struct.fatness)
#     clean_vid_MADP = np.mean(np.abs(np.array(tmp_struct.rgb_sample)-np.roll(np.array(tmp_struct.rgb_sample),1,axis=1)))
#     adv_vid_MADP =   np.mean(np.abs(np.array(tmp_struct.adv_video)-np.roll(np.array(tmp_struct.adv_video),1,axis=1)))
#     clean_vid_MADP_l.append(clean_vid_MADP)
#     adv_vid_MADP_l.append(adv_vid_MADP)
    
roughness_np = np.array(roughness_l)
roughness_mu = roughness_np.mean(axis=0)
sigma_roughness = roughness_np.std(axis=0)

thickness_np = np.array(thickness_l)
thickness_mu = thickness_np.mean(axis=0)
sigma_thickness = thickness_np.std(axis=0)

MADP_cln_np = np.array(clean_vid_MADP_l)
MADP_cln_mu = MADP_cln_np.mean(axis=0)
sigma_MADP_cln = MADP_cln_np.std(axis=0)

MADP_adv_np = np.array(adv_vid_MADP_l)
MADP_adv_mu = MADP_adv_np.mean(axis=0)
sigma_MADP_adv = MADP_adv_np.std(axis=0)

# t = np.arange(roughness_np.shape[1])

# # plot it!
# fig, ax = plt.subplots(1)
# ax.plot(roughness_mu, lw=1, label='mean population 1', color='blue')
# ax.fill_between(t,roughness_mu+sigma_roughness, roughness_mu-sigma_roughness, facecolor='gray', alpha=0.3)

# # ax2 = ax.twinx() 
# ax.plot(thickness_mu, lw=1, label='mean population 2', color='red')
# ax.fill_between(t, thickness_mu+sigma_thickness, thickness_mu-sigma_thickness, facecolor='red', alpha=0.3)    
    


#%%
##  stats private attack ##
results_path = '/data/DL/Adversarial/kinetics-i3d/result/batch/'
results_path = '/home/origan/WW_DL/home/ROIPO/DL_server/data/DL/Adversarial/kinetics-i3d/result/single_video_attack/targeted/'

leg=[]
allfiles = os.listdir(results_path)
files = [fname for fname in allfiles if fname.endswith('.pkl')]

roughness_l=[]
thickness_l=[]
clean_vid_MADP_l=[]
adv_vid_MADP_l=[]
for i,res in enumerate(files):
    
    with open(os.path.join(results_path+files[i]), 'rb') as handle:
        tmp_dict = pickle.load(handle)
        
    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())
    ids_flag =(tmp_struct.softmax.argmax(axis=-1)==tmp_struct.softmax[-1].argmax())
    roughness_l(np.array(tmp_struct.smoothness)[ids_flag].min())
    thickness_l.append(np.array(tmp_struct.fatness)[ids_flag].min())

       
roughness_np = np.array(roughness_l)
roughness_mu = roughness_np.mean(axis=0)
sigma_roughness = roughness_np.std(axis=0)

thickness_np = np.array(thickness_l)
thickness_mu = thickness_np.mean(axis=0)
sigma_thickness = thickness_np.std(axis=0)

MADP_cln_np = np.array(clean_vid_MADP_l)
MADP_cln_mu = MADP_cln_np.mean(axis=0)
sigma_MADP_cln = MADP_cln_np.std(axis=0)


MADP_adv_np = np.array(adv_vid_MADP_l)
MADP_adv_mu = MADP_adv_np.mean(axis=0)
sigma_MADP_adv = MADP_adv_np.std(axis=0)




#%%
##  stats private attack ##
results_path = '/data/DL/Adversarial/kinetics-i3d/result/batch/'
results_path = '/data/DL/Adversarial/kinetics-i3d/result/single_video_attack/targeted/'

leg=[]
allfiles = os.listdir(results_path)
files = [fname for fname in allfiles if fname.endswith('.pkl')]

roughness_l=[]
thickness_l=[]
clean_vid_MADP_l=[]
adv_vid_MADP_l=[]
for i,res in enumerate(files):
    
    with open(os.path.join(results_path+files[i]), 'rb') as handle:
        tmp_dict = pickle.load(handle)
        
    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())
    roughness_l.append(tmp_struct.smoothness)
    thickness_l.append(tmp_struct.fatness)
    clean_vid_MADP = np.mean(np.abs(np.array(tmp_struct.rgb_sample)-np.roll(np.array(tmp_struct.rgb_sample),1,axis=1)))
    adv_vid_MADP =   np.mean(np.abs(np.array(tmp_struct.adv_video)-np.roll(np.array(tmp_struct.adv_video),1,axis=1)))
    clean_vid_MADP_l.append(clean_vid_MADP)
    adv_vid_MADP_l.append(adv_vid_MADP)
    
    
roughness_np = np.array(roughness_l)
roughness_mu = roughness_np.mean(axis=0)
sigma_roughness = roughness_np.std(axis=0)

thickness_np = np.array(thickness_l)
thickness_mu = thickness_np.mean(axis=0)
sigma_thickness = thickness_np.std(axis=0)

MADP_cln_np = np.array(clean_vid_MADP_l)
MADP_cln_mu = MADP_cln_np.mean(axis=0)
sigma_MADP_cln = MADP_cln_np.std(axis=0)


MADP_adv_np = np.array(adv_vid_MADP_l)
MADP_adv_mu = MADP_adv_np.mean(axis=0)
sigma_MADP_adv = MADP_adv_np.std(axis=0)

# t = np.arange(roughness_np.shape[1])

# # plot it!
# fig, ax = plt.subplots(1)
# ax.plot(roughness_mu, lw=1, label='mean population 1', color='blue')
# ax.fill_between(t,roughness_mu+sigma_roughness, roughness_mu-sigma_roughness, facecolor='gray', alpha=0.3)

# # ax2 = ax.twinx() 
# ax.plot(thickness_mu, lw=1, label='mean population 2', color='red')
# ax.fill_between(t, thickness_mu+sigma_thickness, thickness_mu-sigma_thickness, facecolor='red', alpha=0.3)    
    


#%%
##


results_path = '/data/DL/Adversarial/kinetics-i3d/result/batch/'

leg=[]
allfiles = os.listdir(results_path)
files = [fname for fname in allfiles if fname.endswith('.pkl')]
beta_1 = [float(f.split('_')[-1].strip('.pkl')) for f in files]
idx = np.argsort(np.array(beta_1)).astype(np.uint8)
col = np.linspace(0.1,1,files.__len__())

for i,res in enumerate(files):
    
    with open(os.path.join(results_path+files[1]), 'rb') as handle:
        tmp_dict = pickle.load(handle)

    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())
    
    max_prob = np.max(tmp_struct.softmax, axis=-1)
    correct_cls_prob = tmp_struct.softmax[:,tmp_struct.correct_cls_id]
    roughness =tmp_struct.smoothness
    thickness =tmp_struct.fatness
    
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('iter (#)')
    ax1.set_ylabel('probability', color='g')
    lns2 =ax1.plot(correct_cls_prob, 'r',label='original class')
    lns1 = ax1.semilogx(max_prob, '--g',label='max class')
    ax1.tick_params(axis='y', labelcolor='g')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    ax2.set_ylabel('MAP[%]', color='b') 
    lns3 =ax2.plot(roughness, 'b',label='roughness' )
    lns4=ax2.plot(thickness, '--b',label='thickness')
    ax2.tick_params(axis='y', labelcolor='b')
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=6)
    ax2.grid(True)





#%%
## ########################### 3D ###################################
ax = fig.add_subplot(111, projection='3d')
z = np.max(tmp_struct.softmax, axis=-1)

fig = plt.figure()
ax = fig.gca(projection='3d')
results_path = '/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/pkl_final/rgb_sIn7Te48YL4@shooting_goal_(soccer)/'

leg=[]
allfiles = os.listdir(results_path)
files = [fname for fname in allfiles if fname.endswith('.pkl')]
beta_1 = [float(f.split('_')[-1].strip('.pkl')) for f in files]
idx = np.argsort(np.array(beta_1)).astype(np.uint8)
col = np.linspace(0.1,1,files.__len__())

for i,res in enumerate(files):
    fid = 'rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_0.' + str(i) + '0.pkl'
    if i ==10: fid = 'rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_1.00.pkl'
    
    with open(os.path.join(results_path,fid), 'rb') as handle:
        tmp_dict = pickle.load(handle)

    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())

    z = np.max(tmp_struct.softmax, axis=-1)
    smoothness =np.mean(np.mean(np.abs(np.array(tmp_struct.perturbation)-np.roll(np.array(tmp_struct.perturbation),1,axis=1)),axis=1),-1)
    fatness =np.mean(np.mean(np.abs(np.array(tmp_struct.perturbation)),axis=1),-1)
    
    smoothness_ = smoothness.squeeze()/ 2.0 * 100
    fatness_ = fatness.squeeze()/ 2.0 * 100

    ax.plot(smoothness_, fatness_, z,color=[col[i],1-col[i],0])
    leg.append(r'$\beta_1$: {:.1f} $\beta_2$: {:.1f}' .format(i/10,1-i/10))
   
ax.legend(leg)
plt.xlabel('Roughness')
plt.ylabel('Thickness')
ax.set_zlabel('probability')


#%%
##  old pkl version ##

path ='/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/pkl_final/_rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_0.50.pkl'
with open(path, 'rb') as handle:
    tmp_dict = pickle.load(handle)

tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())

max_prob = np.max(tmp_struct.softmax, axis=-1)
correct_cls_prob = tmp_struct.softmax[:,tmp_struct.correct_cls_id]
# roughness =tmp_struct.smoothness
# thickness =tmp_struct.fatness

roughness =np.mean(np.mean(np.abs(np.array(tmp_struct.perturbation)-np.roll(np.array(tmp_struct.perturbation),1,axis=1)),axis=1),-1)
thickness =np.mean(np.mean(np.abs(np.array(tmp_struct.perturbation)),axis=1),-1)

roughness = roughness.squeeze()/ 2.0 * 100
thickness = thickness.squeeze()/ 2.0 * 100
fig, ax1 = plt.subplots()

ax1.set_xlabel('iter (#)')
ax1.set_ylabel('probability', color='g')
lns2 =ax1.plot(correct_cls_prob, 'r',label='original class')

lns1 = ax1.semilogx(max_prob, '--g',label='max class')
ax1.tick_params(axis='y', labelcolor='g')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('MAP[%]', color='b') 
lns3 =ax2.plot(roughness, 'b',label='roughness' )
lns4=ax2.plot(thickness, '--b',label='thickness')
ax2.tick_params(axis='y', labelcolor='b')
ax2.grid(True)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=6)


#%%
## video MADP #######

# path ='/data/DL/Adversarial/kinetics-i3d/result/batch/res_playing_drums_00021.pkl'

path ='/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/pkl_final/kinetics@triple_jumpbeta_1_0.50.pkl'
with open(path, 'rb') as handle:
    tmp_dict = pickle.load(handle)

tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())

max_prob = np.max(tmp_struct.softmax, axis=-1)
correct_cls_prob = tmp_struct.softmax[:,tmp_struct.correct_cls_id]
# roughness =tmp_struct.smoothness
# thickness =tmp_struct.fatness
clean_vid_MADP = np.mean(np.abs(np.array(tmp_struct.rgb_sample)-np.roll(np.array(tmp_struct.rgb_sample),1,axis=1)))
adv_vid_MADP =   np.mean(np.abs(np.array(tmp_struct.adv_video)-np.roll(np.array(tmp_struct.adv_video),1,axis=1)))
roughness =np.mean(np.mean(np.abs(np.array(tmp_struct.perturbation)-np.roll(np.array(tmp_struct.perturbation),1,axis=1)),axis=1),-1)
thickness =np.mean(np.mean(np.abs(np.array(tmp_struct.perturbation)),axis=1),-1)

roughness = roughness.squeeze()/ 2.0 * 100
thickness = thickness.squeeze()/ 2.0 * 100
fig, ax1 = plt.subplots()

ax1.set_xlabel('iter (#)')
ax1.set_ylabel('probability', color='g')
lns2 =ax1.plot(correct_cls_prob, 'r',label='original class')

lns1 = ax1.semilogx(max_prob, '--g',label='max class')
ax1.tick_params(axis='y', labelcolor='g')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('[%]', color='b') 
lns3 =ax2.plot(roughness, 'b',label='roughness' )
lns4=ax2.plot(thickness, '--b',label='thickness')
ax2.tick_params(axis='y', labelcolor='b')
ax2.grid(True)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=6)


if 1:

    num_of_frame = 5
    start_frame = 62
    frame_gap = 1
    end_frmae = start_frame +frame_gap*num_of_frame
    adv_frames_bank = tmp_struct.adv_video.squeeze()
    cln_frames_bank =  tmp_struct.rgb_sample.squeeze()
    
    diff_frames_bank=adv_frames_bank- cln_frames_bank
    
    concat_cln = cln_frames_bank[end_frmae]
    concat_adv = adv_frames_bank[end_frmae]

    
    vert_pad = np.ones(shape=[concat_adv.shape[0],8,3])
    
    
    # diff_frames_bank-=diff_frames_bank.min() 
    # diff_frames_bank-=1.
    concat_diff = diff_frames_bank[end_frmae]

    for i in np.arange(end_frmae-1,start_frame,-frame_gap):
        
        concat_adv = np.concatenate([concat_adv,vert_pad,adv_frames_bank[i]],axis=1)
        
        concat_cln = np.concatenate([concat_cln,vert_pad,cln_frames_bank[i]],axis=1)
        
        
        # dif=]
        
        # plt.bar('R', 4, color = 'r', width = 0.5)
        
        concat_diff = np.concatenate([concat_diff,vert_pad,diff_frames_bank[i]],axis=1)

    
    
    # concat_diff-=concat_diff.min()
    # concat_diff-=1.
    # concat_diff/=concat_diff.max()/2.
    # concat_diff-=1.0
    
    horiz_pad =np.ones(shape=[8,concat_diff.shape[1],3])
        
    full_concat = np.concatenate([concat_cln,horiz_pad,concat_adv,horiz_pad,concat_diff],axis=0)  
    
    fig,ax =plt.subplots()
    # plt.subplot(2,1,1)
    full_concat = ((full_concat+1.0)*127.5).astype(np.uint8)
    im = plt.imshow(full_concat)
    
    plt.axis('OFF')
    # plt.subplot(2,1,2)
    # plt.plot(ll)

#%%
##


import cv2
import kinetics_i3d_utils as i3d_utils
from scipy.special import kl_div

def warp_flow(img, flow):
   h, w = flow.shape[:2]
   flow = -flow
   flow[:,:,0] += np.arange(w)
   flow[:,:,1] += np.arange(h)[:,np.newaxis]
   res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
   return res

def of_calc(prev, frame):

    image_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)

    # flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    optical_flow = cv2.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(prev_gray, image_gray, None)
    return flow


model =i3d_utils.kinetics_i3d_inference()
path ='/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/npy/rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_0.50_cyc.pkl'
with open(path, 'rb') as handle:
    tmp_dict = pickle.load(handle)

tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())




pp=tmp_struct.perturbation[-1].squeeze()
aa=np.concatenate([pp[39:47]]*12)[:90]
aa=np.expand_dims(aa,axis=1)
aa=np.expand_dims(aa,axis=1)


#%%
##
clean_video = tmp_struct.rgb_sample
adv_video = tmp_struct.adv_video

k=5
num_frame =90
target_frameet_frame =clean_video[0,target_frame_idx]

orig_prob = model(adv_video,adv_flag=0)

adv_to_adv_l=[]
cln_to_cln_l=[]

for i in np.arange(k,num_frame):
    orig_prob = model(adv_video,adv_flag=0)
    target_frame =adv_video[0,i]
    cs=np.zeros_like(orig_prob)
    for j in np.arange(i-k,i):
        
        adv_video_tmp = adv_video.copy()
        fref = adv_video_tmp[0,j]
        of = of_calc(fref,target_frame)
        eps = np.random.normal(0,1,of.shape)
        
        of_eps = of+eps
        
        target_frame_recon = warp_flow(fref,of)
        
        adv_video_tmp[0,j]=target_frame_recon
    
        prob = model(adv_video_tmp,adv_flag=0)
    
        kl_p_q = kl_div(orig_prob, prob)
        kl_q_qp= kl_div(prob,orig_prob)
        
        cs+= (kl_p_q+kl_q_qp)/2.
    
    adv_to_adv_l.append(cs.mean())

for i in np.arange(k,num_frame):
    orig_prob = model(clean_video,adv_flag=0)
    target_frame =clean_video[0,i]
    cs=np.zeros_like(orig_prob)
    for j in np.arange(i-k,i):
        
        clean_video_tmp = clean_video.copy()
        fref = clean_video_tmp[0,j]
        of = of_calc(fref,target_frame)
        eps = np.random.normal(0,1,of.shape)
        
        of_eps = of+eps
        
        target_frame_recon = warp_flow(fref,of)
        
        clean_video_tmp[0,j]=target_frame_recon
    
        prob = model(clean_video_tmp,adv_flag=0)
    
        kl_p_q = kl_div(orig_prob, prob)
        kl_q_qp= kl_div(prob,orig_prob)
        
        cs+= (kl_p_q+kl_q_qp)/2.
    
    cln_to_cln_l.append(cs.mean())


#%%
##
    
    
    
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


# path ='/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/npy/kinetics@triple_jumpbeta_1_0.50.pkl'
path=os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_high_jump.pkl'
with open(path, 'rb') as handle:
    tmp_dict = pickle.load(handle)

tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(facecolor='black')#,constrained_layout=True)

ax_pert_graph =fig.add_subplot(2,3,5,facecolor='k')
ax_pert_graph.set_xlim((0, 90))


ax_adv_vid=fig.add_subplot(2,3,3)
ax_pert_vid=fig.add_subplot(2,3,2)
ax_cln_vid=fig.add_subplot(2,3,1)
# ax1 = ax.twinx()

# ax = plt.axes(xlim=(0, 90), ylim=(-0.1, 0.1))
ax_adv_vid.axis('OFF')
ax_pert_vid.axis('OFF')
ax_cln_vid.axis('OFF')


line, = ax_pert_graph.plot([],[] ,lw=2)

adv_video = ((tmp_struct.adv_video[0] +1.0)*127.5).astype(np.uint8)
dummy_img = adv_video[0]
cln_video = ((tmp_struct.rgb_sample[0] +1.0)*127.5).astype(np.uint8)

pert_raw=tmp_struct.perturbation[-1].copy()-tmp_struct.perturbation[-1].min()

scale_factor = int(2/pert_raw.max())
pert_raw/=pert_raw.max()
pert_raw*=255
pert_raw=pert_raw.astype(np.uint8)


pert_video=np.repeat(pert_raw,224,axis=1)
pert_video=np.repeat(pert_video,224,axis=2)


pert = tmp_struct.perturbation[-1].squeeze()/2.0*100

font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 16,
        }

# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

ax_cln_vid.set_title('Clean video\n top-1 class: {}'.format(kinetics_classes[tmp_struct.correct_cls_id]) ,font)
ax_pert_vid.set_title('Perturbation\n'+r'(amplified $\times${} for visualization)'.format(scale_factor),font)
ax_adv_vid.set_title('Adversarial video\n top-1 class: {}'.format(kinetics_classes[tmp_struct.softmax[-1].argmax()]),
                          font)

ax_pert_graph.set_title('RGB Perturbation\n percents from the full scale of the image',font)
ax_pert_graph.set_ylabel('Amplitude from full scale[%]', font) 

font2 = {'family': 'serif',
        'color':  'y',
        'weight': 'normal',
        'size': 16,
        }
ax_pert_graph.set_xlabel('Current\nperturbation', font2) 


y_top=1.2*np.abs(pert).max()
ax_pert_graph.set_ylim(-y_top,y_top)

# ax_pert_graph.yaxis.label.set_color('white')
ax_pert_graph.tick_params(axis='y', labelcolor='w')
ax_pert_graph.tick_params(axis='x', colors='k')
ax_pert_graph.grid(True)


pp=y_top-np.abs(pert).max()

ax_pert_graph.arrow(45,-y_top, 0, 0.5*pp, head_width=2, head_length=0.5*pp, fc='y', ec='y')

ax_pert_graph.arrow(45,y_top, 0, -0.5*pp, head_width=2, head_length=0.5*pp, fc='y', ec='y')


# plt.tight_layout()
# ax_pert_graph.annotate('a polar annotation',
#             xy=(45, -1),# theta, radius
#             xytext=(0.5, 1),    # fraction, fraction
#             textcoords='figure fraction',
#             arrowprops=dict(facecolor='white', shrink=0.05),
#             horizontalalignment='left',
#             verticalalignment='bottom')
# ax_pert_graph.spines['left'].set_color('w')
fig.set_size_inches(19, 11)
# ax_pert_graph.spines['left'].set_color('white')
# plt.rc('axes',edgecolor='white')


img_adv=ax_adv_vid.imshow(np.zeros_like(dummy_img,dtype=np.uint8),zorder=1)
img_cln=ax_cln_vid.imshow(np.zeros_like(dummy_img,dtype=np.uint8),zorder=1)
img_pert=ax_pert_vid.imshow(np.zeros_like(dummy_img,dtype=np.uint8),zorder=1)

plus_pos = [(ax_cln_vid.get_position().x1 + ax_pert_vid.get_position().x0)/2,
            (ax_cln_vid.get_position().y1+ax_cln_vid.get_position().y0)/2]

fig.text(plus_pos[0],plus_pos[1],'$+$',horizontalalignment='center',verticalalignment='center',fontsize=18,color='white')

equal_pos = [(ax_pert_vid.get_position().x1 + ax_adv_vid.get_position().x0)/2,
            (ax_pert_vid.get_position().y1+ax_pert_vid.get_position().y0)/2]

fig.text(equal_pos[0],equal_pos[1],'$=$', horizontalalignment='center',verticalalignment='center',fontsize=18,color='white')

lines = []
plotlays, plotcols = [3], ["red","green","blue"]

if hasattr(tmp_struct,'smoothness'):
    roughness=tmp_struct.smoothness[-1]
else:
    roughness=np.mean(np.abs(np.roll(tmp_struct.perturbation[-1].squeeze(),axis=0,shift=1)-tmp_struct.perturbation[-1].squeeze()))/2*100

if hasattr(tmp_struct,'fatness'):
    thickness=tmp_struct.fatness[-1]
else:
    thickness=np.mean(np.abs(tmp_struct.perturbation[-1].squeeze()))/2*100



beta1=tmp_struct.beta_1

if hasattr(tmp_struct, 'beta_3'):
    beta2 = tmp_struct.beta_2 +tmp_struct.beta_3
else:
     beta2 = tmp_struct.beta_2*2

fig.suptitle('Adversarial example: '+r'$\beta_1$={},$\beta_2$={},'.format(beta1,beta2)
              +' Thickness={:.2f}%, Roughness={:.2f}%'.format(thickness,roughness),color='w',fontsize=16)
fig.subplots_adjust(hspace=0.22)

# plt.text(10,550 , 'I. Naeh, R. Pony, S. Mannor \"Patternless Adversarial Attacks on Video Recognition Networks\" arXiv',
#         verticalalignment='bottom', horizontalalignment='right',
#         color='green', fontsize=15)

for index in range(3):
    lobj = ax_pert_graph.plot([],[],lw=2,color=plotcols[index])[0]
    lines.append(lobj)

# initialization function: plot the background of each frame
def init():
    
    for i in lines:
        line.set_data([],[])
        
    img_adv.set_data(np.zeros_like(dummy_img,dtype=np.uint8))
    img_cln.set_data(np.zeros_like(dummy_img,dtype=np.uint8))
    img_pert.set_data(np.zeros_like(dummy_img,dtype=np.uint8))

    
    return lines

# animation function.  This is called sequentially
def animate(i):
    ii=i %90
    
    x = np.linspace(0, 89, 90)
    y=np.roll(pert,-ii-45,0)

    img_adv.set_data(adv_video[ii])
    img_cln.set_data(cln_video[ii])
    img_pert.set_data(pert_video[ii])

    y_mean=y.mean(axis=-1)
    y_std = y.std(axis=-1)
    
    
    for lnum,line in enumerate(lines):
        line.set_data(x,y[...,lnum]) # set data for each line separately. 

    # p1 = ax_pert_graph.fill_between(np.arange(50,91) , -y_top,y_top, facecolor = 'gray', alpha = 0.2)
    # p2 = ax_pert_graph.fill_between(np.arange(0,41) , -y_top,y_top, facecolor = 'gray', alpha = 0.2)
    # p3 = ax_pert_graph.fill_between(np.arange(45,47) , -y_top,y_top, facecolor = 'y', alpha = 0.5)

    return lines[0],lines[1],lines[2],img_adv,img_cln,img_pert#,p1,p2#img, #lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, save_count=900,
                               frames=90, interval=100, blit=True,repeat=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You ma need to adjust this for
# your system: for more information, see
# http://ma{{tplotlib.sourceforge.net/api/animation_api.html
if save_to_vid:
    anim.save(os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/{}_beta1_{}_th_{:.2f}%_rg_{:.2f}%.mp4'.format(kinetics_classes[tmp_struct.correct_cls_id].replace(' ','_'),
                                                    tmp_struct.beta_1,thickness,roughness), fps=25,dpi=100, extra_args=['-vcodec', 'libx264','-crf', '5'],savefig_kwargs={'bbox_inches':'tight','quality':100,'facecolor':'black'}) #'-filter_complex','loop=loop=3:size=270:start=0'])
plt.show()


#%%
## artist animation

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


# path =path='/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/npy/kinetics@triple_jumpbeta_1_0.50.pkl'
# path='/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/pkl_final/rgb_sIn7Te48YL4@shooting_goal_(soccer)/rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_1.00.pkl'
with open(path, 'rb') as handle:
    tmp_dict = pickle.load(handle)

tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())

# First set up the figure, the axis, and the plot element we want to animate

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)


fig = plt.figure()
# ax = plt.axes(xlim=(0, 90), ylim=(-0.1, 0.1))
# line, = ax.plot([],[] ,lw=2)
# lines = []
# plotlays, plotcols = [3], ["red","green","blue"]


x = np.linspace(0, 90, 90)






lines=[]
for i in range(2000):
    
    
    y = np.roll(tmp_struct.perturbation[-1].squeeze(),i,0)
    line1,=ax1.plot(y[...,0])
    line2=ax2.imshow(tmp_struct.rgb_sample[0,0])
    # line2,=ax2.plot(y[...,1])
    lines.append([line1,line2])

ani = animation.ArtistAnimation(fig,lines,interval=40,blit=True)

plt.show()
    
# for index in range(3):
#     lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
#     lines.append(lobj)
# # initialization function: plot the background of each frame
# def init():
    
#     for line in lines:
#         line.set_data([],[])
#     return lines
#     # line.set_data([])
#     # return line,

# # animation function.  This is called sequentially
# def animate(i):
#     x = np.linspace(0, 90, 90)
#     # y = np.sin(2 * np.pi * (x - 0.01 * i))
#     y = np.roll(tmp_struct.perturbation[-1].squeeze(),i,0)
    
    
#     # line.set_data(y)
#     # return line,
    
#     for lnum,line in enumerate(lines):
#         line.set_data(x,y[...,lnum]) # set data for each line separately. 

#     return lines

# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=90, interval=40, blit=True)

# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.mp4', fps=25, extra_args=['-vcodec', 'libx264'])

plt.show()


#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


# path ='/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/npy/kinetics@triple_jumpbeta_1_0.50.pkl'
paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_high_jump.pkl',
       os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_high_jump.pkl']

tmp_dicts=[]
for path in paths:
    with open(path, 'rb') as handle:
        tmp_dicts.append(pickle.load(handle))


tmp_structs=[]
for tmp_dict in tmp_dicts:
    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_structs.append( namedtuple("dict", tmp_dict.keys())(*tmp_dict.values()))

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(facecolor='black')#,constrained_layout=True)

ax_pert_graph =fig.add_subplot(2,3,5,facecolor='k')
ax_pert_graph.set_xlim((0, 90))


ax_adv_vid=fig.add_subplot(2,3,3)
ax_pert_vid=fig.add_subplot(2,3,2)
ax_cln_vid=fig.add_subplot(2,3,1)
# ax1 = ax.twinx()

# ax = plt.axes(xlim=(0, 90), ylim=(-0.1, 0.1))
ax_adv_vid.axis('OFF')
ax_pert_vid.axis('OFF')
ax_cln_vid.axis('OFF')


line, = ax_pert_graph.plot([],[] ,lw=2)


adv_videos=[]
dummy_imgs=[]
cln_video=[]
for tmp_struct in tmp_structs:
    adv_videos.append(((tmp_struct.adv_video[0] +1.0)*127.5).astype(np.uint8))
    dummy_imgs.append( adv_video[0])
    cln_videos.append(((tmp_struct.rgb_sample[0] +1.0)*127.5).astype(np.uint8))


pert_raw=tmp_struct.perturbation[-1].copy()-tmp_struct.perturbation[-1].min()

scale_factor = int(2/pert_raw.max())
pert_raw/=pert_raw.max()
pert_raw*=255
pert_raw=pert_raw.astype(np.uint8)


pert_video=np.repeat(pert_raw,224,axis=1)
pert_video=np.repeat(pert_video,224,axis=2)


pert = tmp_struct.perturbation[-1].squeeze()/2.0*100

font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 16,
        }

# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()

ax_cln_vid.set_title('Clean video\n top-1 class: {}'.format(kinetics_classes[tmp_struct.correct_cls_id]) ,font)
ax_pert_vid.set_title('Perturbation\n'+r'(amplified $\times${} for visualization)'.format(scale_factor),font)
ax_adv_vid.set_title('Adversarial video\n top-1 class: {}'.format(kinetics_classes[tmp_struct.softmax[-1].argmax()]),
                          font)

ax_pert_graph.set_title('RGB Perturbation\n percents from the full scale of the image',font)
ax_pert_graph.set_ylabel('Amplitude from full scale[%]', font) 

font2 = {'family': 'serif',
        'color':  'y',
        'weight': 'normal',
        'size': 16,
        }
ax_pert_graph.set_xlabel('Current\nperturbation', font2) 


y_top=1.2*np.abs(pert).max()
ax_pert_graph.set_ylim(-y_top,y_top)

# ax_pert_graph.yaxis.label.set_color('white')
ax_pert_graph.tick_params(axis='y', labelcolor='w')
ax_pert_graph.tick_params(axis='x', colors='k')
ax_pert_graph.grid(True)


pp=y_top-np.abs(pert).max()

ax_pert_graph.arrow(45,-y_top, 0, 0.5*pp, head_width=2, head_length=0.5*pp, fc='y', ec='y')

ax_pert_graph.arrow(45,y_top, 0, -0.5*pp, head_width=2, head_length=0.5*pp, fc='y', ec='y')


# plt.tight_layout()
# ax_pert_graph.annotate('a polar annotation',
#             xy=(45, -1),# theta, radius
#             xytext=(0.5, 1),    # fraction, fraction
#             textcoords='figure fraction',
#             arrowprops=dict(facecolor='white', shrink=0.05),
#             horizontalalignment='left',
#             verticalalignment='bottom')
# ax_pert_graph.spines['left'].set_color('w')
fig.set_size_inches(19, 11)
# ax_pert_graph.spines['left'].set_color('white')
# plt.rc('axes',edgecolor='white')


img_adv=ax_adv_vid.imshow(np.zeros_like(dummy_img,dtype=np.uint8),zorder=1)
img_cln=ax_cln_vid.imshow(np.zeros_like(dummy_img,dtype=np.uint8),zorder=1)
img_pert=ax_pert_vid.imshow(np.zeros_like(dummy_img,dtype=np.uint8),zorder=1)

plus_pos = [(ax_cln_vid.get_position().x1 + ax_pert_vid.get_position().x0)/2,
            (ax_cln_vid.get_position().y1+ax_cln_vid.get_position().y0)/2]

fig.text(plus_pos[0],plus_pos[1],'$+$',horizontalalignment='center',verticalalignment='center',fontsize=18,color='white')

equal_pos = [(ax_pert_vid.get_position().x1 + ax_adv_vid.get_position().x0)/2,
            (ax_pert_vid.get_position().y1+ax_pert_vid.get_position().y0)/2]

fig.text(equal_pos[0],equal_pos[1],'$=$', horizontalalignment='center',verticalalignment='center',fontsize=18,color='white')

lines = []
plotlays, plotcols = [3], ["red","green","blue"]

if hasattr(tmp_struct,'smoothness'):
    roughness=tmp_struct.smoothness[-1]
else:
    roughness=np.mean(np.abs(np.roll(tmp_struct.perturbation[-1].squeeze(),axis=0,shift=1)-tmp_struct.perturbation[-1].squeeze()))/2*100

if hasattr(tmp_struct,'fatness'):
    thickness=tmp_struct.fatness[-1]
else:
    thickness=np.mean(np.abs(tmp_struct.perturbation[-1].squeeze()))/2*100



beta1=tmp_struct.beta_1

if hasattr(tmp_struct, 'beta_3'):
    beta2 = tmp_struct.beta_2 +tmp_struct.beta_3
else:
     beta2 = tmp_struct.beta_2*2

fig.suptitle('Adversarial example: '+r'$\beta_1$={},$\beta_2$={},'.format(beta1,beta2)
              +' Thickness={:.2f}%, Roughness={:.2f}%'.format(thickness,roughness),color='w',fontsize=16)
fig.subplots_adjust(hspace=0.22)

# plt.text(10,550 , 'I. Naeh, R. Pony, S. Mannor \"Patternless Adversarial Attacks on Video Recognition Networks\" arXiv',
#         verticalalignment='bottom', horizontalalignment='right',
#         color='green', fontsize=15)

for index in range(3):
    lobj = ax_pert_graph.plot([],[],lw=2,color=plotcols[index])[0]
    lines.append(lobj)

# initialization function: plot the background of each frame
def init():
    
    for i in lines:
        line.set_data([],[])
        
    img_adv.set_data(np.zeros_like(dummy_img,dtype=np.uint8))
    img_cln.set_data(np.zeros_like(dummy_img,dtype=np.uint8))
    img_pert.set_data(np.zeros_like(dummy_img,dtype=np.uint8))

    
    return lines

# animation function.  This is called sequentially
def animate(i):
    ii=i %90
    
    x = np.linspace(0, 89, 90)
    y=np.roll(pert,-ii-45,0)

    img_adv.set_data(adv_video[ii])
    img_cln.set_data(cln_video[ii])
    img_pert.set_data(pert_video[ii])

    y_mean=y.mean(axis=-1)
    y_std = y.std(axis=-1)
    
    
    for lnum,line in enumerate(lines):
        line.set_data(x,y[...,lnum]) # set data for each line separately. 

    # p1 = ax_pert_graph.fill_between(np.arange(50,91) , -y_top,y_top, facecolor = 'gray', alpha = 0.2)
    # p2 = ax_pert_graph.fill_between(np.arange(0,41) , -y_top,y_top, facecolor = 'gray', alpha = 0.2)
    # p3 = ax_pert_graph.fill_between(np.arange(45,47) , -y_top,y_top, facecolor = 'y', alpha = 0.5)

    return lines[0],lines[1],lines[2],img_adv,img_cln,img_pert#,p1,p2#img, #lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, save_count=900,
                               frames=90, interval=100, blit=True,repeat=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You ma need to adjust this for
# your system: for more information, see
# http://ma{{tplotlib.sourceforge.net/api/animation_api.html
if save_to_vid:
    anim.save(os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/{}_beta1_{}_th_{:.2f}%_rg_{:.2f}%.mp4'.format(kinetics_classes[tmp_struct.correct_cls_id].replace(' ','_'),
                                                    tmp_struct.beta_1,thickness,roughness), fps=25,dpi=100, extra_args=['-vcodec', 'libx264','-crf', '5'],savefig_kwargs={'bbox_inches':'tight','quality':100,'facecolor':'black'}) #'-filter_complex','loop=loop=3:size=270:start=0'])
plt.show()


#%%

paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_high_jump.pkl',
       os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_high_jump.pkl']


paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_juggling_balls.pkl',
       
os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_juggling_balls.pkl']


# paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_tai_chi.pkl',
#  os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_tai_chi.pkl']
concat_diff_l=[]
concat_adv_l=[]
for path in paths:
    
    with open(path, 'rb') as handle:
        tmp_dict = pickle.load(handle)
    
    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())
    
    
    
    num_of_frame = 8
    start_frame = 15 # 50 #juggling 25 # high_jump 49
    frame_gap = 1
    end_frmae = start_frame +frame_gap*num_of_frame
    adv_frames_bank = tmp_struct.adv_video.squeeze()
    cln_frames_bank =  tmp_struct.rgb_sample.squeeze()
    
    diff_frames_bank=adv_frames_bank- cln_frames_bank
    
    concat_cln = cln_frames_bank[end_frmae]
    concat_adv = adv_frames_bank[end_frmae]
    
    
    vert_pad = np.ones(shape=[concat_adv.shape[0],8,3])
    diff_frames_bank =(tmp_struct.perturbation[-1]*np.ones([90,224,224,3])).astype(np.float32)
    
    # diff_frames_bank-=diff_frames_bank.min() 
    # diff_frames_bank-=1.
    concat_diff = diff_frames_bank[end_frmae]
    
    print("{}".format(concat_diff.min()))
    concat_diff = concat_diff-(concat_diff.min())
    # concat_diff = concat_diff-(-0.01689697988331318)
    
    
    for i in np.arange(end_frmae-1,start_frame,-frame_gap):
        
        concat_adv = np.concatenate([concat_adv,vert_pad,adv_frames_bank[i]],axis=1)
        
        concat_cln = np.concatenate([concat_cln,vert_pad,cln_frames_bank[i]],axis=1)
        
        
        # dif=]
        
        # plt.bar('R', 4, color = 'r', width = 0.5)
        
        concat_diff = np.concatenate([concat_diff,vert_pad,diff_frames_bank[i]],axis=1)
    
    
    
    concat_diff_l.append(concat_diff)
    concat_adv_l.append(concat_adv)
    # concat_diff-=concat_diff.min()
    # concat_diff-=1.
    # concat_diff/=concat_diff.max()/2.
    # concat_diff-=1.0
    
    
horiz_pad =np.ones(shape=[8,concat_diff.shape[1],3])
        
full_concat = np.concatenate([concat_cln,horiz_pad,concat_adv_l[0],horiz_pad,concat_adv_l[1],horiz_pad,concat_diff_l[0],horiz_pad,concat_diff_l[1]],axis=0)  
    
fig,ax =plt.subplots()
# plt.subplot(2,1,1)
full_concat = ((full_concat+1.0)*127.5).astype(np.uint8)
im = plt.imshow(full_concat)

plt.axis('OFF')
# plt.subplot(2,1,2)
# plt.plot(ll)