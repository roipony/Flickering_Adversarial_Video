#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:59:22 2020

@author: ubadmin
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
from matplotlib.ticker import FormatStrFormatter

# fig, ax = plt.subplots()
#
# ax = fig.add_subplot(111, projection='3d')
# z = np.max(tmp_struct.softmax, axis=-1)

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
##


# fig = plt.figure()
# ax = fig.gca(projection='3d')
results_path = '/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/pkl_final/rgb_sIn7Te48YL4@shooting_goal_(soccer)/'

leg = []
allfiles = os.listdir(results_path)
files = [fname for fname in allfiles if fname.endswith('.pkl')]
beta_1 = [float(f.split('_')[-1].strip('.pkl')) for f in files]
files = np.array(files)[np.argsort(beta_1)]
col = np.linspace(0.1, 1, files.__len__())
thickness_l=[]
for i, res in enumerate(files):
    fid = 'rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_0.' + str(i) + '0.pkl'
    if i == 10: fid = 'rgb_sIn7Te48YL4@shooting_goal_(soccer)beta_1_1.00.pkl'

    with open(os.path.join(results_path, fid), 'rb') as handle:
        tmp_dict = pickle.load(handle)

    # tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())

    # z = np.max(tmp_struct.softmax, axis=-1)

    thickness = np.array(tmp_struct.perturbation[-1]*100.).squeeze()/2
    thickness_l.append(thickness[np.newaxis])

thickness_l.reverse()
thickness_l_np = np.concatenate(thickness_l,0)
thickness_l_np-=thickness_l_np.min()
thickness_l_np =thickness_l_np/thickness_l_np.max()
thickness_l_np*=255
thickness_l_np=thickness_l_np.astype(np.uint8)

plt.figure()
ax1=plt.subplot(3,1,1)
ax1.plot(thickness_l[0][...,0].squeeze(),color='r')
ax1.plot(thickness_l[0][...,1].squeeze(),color='g')
ax1.plot(thickness_l[0][...,2].squeeze(),color='b')
ax1.set_xlim(0,90)
ax1.set_ylim(-5,5)
ax1.set_yticks(np.arange(-4,5,step=2))
plt.ylabel('Amplitude [%] \n $\\beta_1=1$, $\\beta_2$=0')
ax1.grid(True)

plt.subplot(3,1,2)
plt.imshow(thickness_l_np)
ax2= plt.gca()
ax2.yaxis.set_ticks(ticks=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
ax2.yaxis.set_ticklabels(['{:.1f}'.format(e) for e in np.arange(0,1.05,step=0.1)])
plt.ylabel('$\\beta_2$(= 1- $\\beta_1$)')

ax3=plt.subplot(3,1,3)
ax3.plot(thickness_l[-1][...,0].squeeze(),color='r')
ax3.plot(thickness_l[-1][...,1].squeeze(),color='g')
ax3.plot(thickness_l[-1][...,2].squeeze(),color='b')
ax3.set_xlim(0,90)
ax3.set_ylim(-5,5)
ax3.set_yticks(np.arange(-4,5,step=2))
plt.ylabel('Amplitude [%] \n $\\beta_1=0$, $\\beta_2$=1')
ax3.grid(True)
# plt.yticks(ticks=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
           # labels = ['{:.1f}'.format(e) for e in np.arange(0,1.05,step=0.1)] )#np.arange(0,1.05,step=0.1))
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.show()
#%%



paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_tai_chi.pkl',
 os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_tai_chi.pkl']


paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_high_jump.pkl',
       os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_high_jump.pkl']

paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_juggling_balls.pkl',
       
os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_juggling_balls.pkl']

concat_diff_l=[]
concat_adv_l=[]
# for path in paths:

# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# results_path = '/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/pkl_final/rgb_sIn7Te48YL4@shooting_goal_(soccer)/'

# leg = []
# # allfiles = os.listdir(results_path)
# files = [fname for fname in allfiles if fname.endswith('.pkl')]
# beta_1 = [float(f.split('_')[-1].strip('.pkl')) for f in files]
# files = np.array(files)[np.argsort(beta_1)]
# col = np.linspace(0.1, 1, files.__len__())
thickness_l=[]


for path in paths:
    
    with open(path, 'rb') as handle:
        tmp_dict = pickle.load(handle)
    
    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())
    
    # z = np.max(tmp_struct.softmax, axis=-1)

    thickness = np.array(tmp_struct.perturbation[-1]*100.).squeeze()/2
    thickness_l.append(thickness[np.newaxis])

thickness_l.reverse()
thickness_l_np = np.concatenate(thickness_l,0)
thickness_l_np-=thickness_l_np.min()
thickness_l_np =thickness_l_np/thickness_l_np.max()
thickness_l_np*=255
thickness_l_np=thickness_l_np.astype(np.uint8)



max_lim = np.max([thickness_l[0].__abs__().max(),thickness_l[1].__abs__().max()])

plt.figure()
ax1=plt.subplot(2,1,1)
ax1.plot(thickness_l[0][...,0].squeeze(),color='r')
ax1.plot(thickness_l[0][...,1].squeeze(),color='g')
ax1.plot(thickness_l[0][...,2].squeeze(),color='b')
ax1.set_xlim(0,90)
ax1.set_ylim(-1.*max_lim-0.2 ,max_lim +0.2)
ax1.set_yticks(np.arange(-1.*max_lim,max_lim,step=2))
plt.ylabel('Amplitude [%] \n $\\beta_1=1$, $\\beta_2$=0')
ax1.grid(True)

# plt.subplot(3,1,2)
# plt.imshow(thickness_l_np)
# ax2= plt.gca()
# ax2.yaxis.set_ticks(ticks=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
# ax2.yaxis.set_ticklabels(['{:.1f}'.format(e) for e in np.arange(0,1.05,step=0.1)])
# plt.ylabel('$\\beta_2$(= 1- $\\beta_1$)')

ax3=plt.subplot(2,1,2)
ax3.plot(thickness_l[-1][...,0].squeeze(),color='r')
ax3.plot(thickness_l[-1][...,1].squeeze(),color='g')
ax3.plot(thickness_l[-1][...,2].squeeze(),color='b')
ax3.set_xlim(0,90)
ax3.set_ylim(-1.*max_lim-0.2 ,max_lim +0.2)
ax3.set_yticks(np.arange(-1.*max_lim,max_lim,step=2))
plt.ylabel('Amplitude [%] \n $\\beta_1=0$, $\\beta_2$=1')
ax3.grid(True)
# plt.yticks(ticks=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
           # labels = ['{:.1f}'.format(e) for e in np.arange(0,1.05,step=0.1)] )#np.arange(0,1.05,step=0.1))
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.show()
#%%
from matplotlib.lines import Line2D
cmap = plt.cm.coolwarm

paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_tai_chi.pkl',
 os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_tai_chi.pkl']


paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_high_jump.pkl',
       os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_high_jump.pkl']

paths=[os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_0.0_juggling_balls.pkl',
       
os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/res_beta_1_1.0_juggling_balls.pkl']

concat_diff_l=[]
concat_adv_l=[]
# for path in paths:

# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# results_path = '/data/DL/Adversarial/kinetics-i3d/result/videos_for_tests/pkl_final/rgb_sIn7Te48YL4@shooting_goal_(soccer)/'

# leg = []
# # allfiles = os.listdir(results_path)
# files = [fname for fname in allfiles if fname.endswith('.pkl')]
# beta_1 = [float(f.split('_')[-1].strip('.pkl')) for f in files]
# files = np.array(files)[np.argsort(beta_1)]
# col = np.linspace(0.1, 1, files.__len__())
thickness_l=[]


for path in paths:
    
    with open(path, 'rb') as handle:
        tmp_dict = pickle.load(handle)
    
    tmp_dict['softmax'] = np.concatenate(tmp_dict['softmax'], axis=0)
    tmp_struct = namedtuple("dict", tmp_dict.keys())(*tmp_dict.values())
    
    # z = np.max(tmp_struct.softmax, axis=-1)

    thickness = np.array(tmp_struct.perturbation[-1]*100.).squeeze()/2
    thickness_l.append(thickness[np.newaxis])

thickness_l.reverse()
thickness_l_np = np.concatenate(thickness_l,0)
thickness_l_np-=thickness_l_np.min()
thickness_l_np =thickness_l_np/thickness_l_np.max()
thickness_l_np*=255
thickness_l_np=thickness_l_np.astype(np.uint8)




plt.figure()
ax1=plt.subplot(1,1,1)
thickness_l[0]=thickness_l[0][0,15:15+8]
thickness_l[1]=thickness_l[1][0,15:15+8]
# thickness_l[0]=thickness_l[0][::-1]
# thickness_l[1]=thickness_l[1][::-1]

max_lim = np.max([thickness_l[0].__abs__().max(),thickness_l[1].__abs__().max()])

ax1.plot(thickness_l[0][...,0].squeeze(),color='r')
ax1.plot(thickness_l[0][...,1].squeeze(),color='g')
ax1.plot(thickness_l[0][...,2].squeeze(),color='b')
ax1.set_xlim(0,7)
ax1.set_ylim(-1.*max_lim-0.2 ,max_lim +0.2)
ax1.set_yticks(np.arange(-1.*max_lim,max_lim,step=2))
plt.ylabel('Amplitude [%]')
ax1.grid(True)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
# plt.subplot(3,1,2)
# plt.imshow(thickness_l_np)
# ax2= plt.gca()
# ax2.yaxis.set_ticks(ticks=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
# ax2.yaxis.set_ticklabels(['{:.1f}'.format(e) for e in np.arange(0,1.05,step=0.1)])
# plt.ylabel('$\\beta_2$(= 1- $\\beta_1$)')

# ax3=plt.subplot(2,1,2)
ax1.plot(thickness_l[-1][...,0].squeeze(),color='r',ls='--')
ax1.plot(thickness_l[-1][...,1].squeeze(),color='g',ls='--')
ax1.plot(thickness_l[-1][...,2].squeeze(),color='b',ls='--')
# ax1.set_xlim(0,8)
# ax1.set_ylim(-1.*max_lim-0.2 ,max_lim +0.2)
# ax1.set_yticks(np.arange(-1.*max_lim,max_lim,step=2))
# ax1.ylabel('Amplitude [%] \n $\\beta_1=0$, $\\beta_2$=1')

custom_lines = [Line2D([0], [0], color='k', lw=1),
                Line2D([0], [0], color='k', lw=1,ls='--')]


ax1.legend(custom_lines, [r'$\beta_1$: {:.1f} $\beta_2$: {:.1f}' .format(1,0), 
                          r'$\beta_1$: {:.1f} $\beta_2$: {:.1f}' .format(0,1), ])

   

ax1.grid(True)
# plt.yticks(ticks=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
           # labels = ['{:.1f}'.format(e) for e in np.arange(0,1.05,step=0.1)] )#np.arange(0,1.05,step=0.1))
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.show()

#%%


#%%


#%%


#%%


#%%

