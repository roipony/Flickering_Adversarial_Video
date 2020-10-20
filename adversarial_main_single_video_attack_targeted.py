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
# tf.enable_eager_execution()
# sys.path.insert(1, '../Adversarial/kinetics-i3d/')

sys.path.insert(1, os.path.realpath(os.path.pardir))


# import skvideo
from utils import pre_process_rgb_flow as img_tool
from utils import kinetics_i3d_utils as ki3du


_IMAGE_SIZE = 224
_BATCH_SIZE = 1

_SAMPLE_VIDEO_FRAMES =90 #90 #79 # 79 #90 #90 #250 #90 #79
_BASE_PATCH_FRAMES = _SAMPLE_VIDEO_FRAMES #_SAMPLE_VIDEO_FRAMES #_SAMPLE_VIDEO_FRAMES # 1# _SAMPLE_VIDEO_FRAMES # 1:for sticker _SAMPLE_VIDEO_FRAMES # 1
_IND_START = 0  # 0 #50
_IND_END =_SAMPLE_VIDEO_FRAMES

kinetics_classes=ki3du.load_kinetics_classes()

#%% model loader
ckpt_path = '/data/DL/Adversarial/kinetics-i3d/result/generalization/universal/val_test/all_cls_shuffle_t15000_v2000_/'
ckpt_last = tf.train.latest_checkpoint(checkpoint_dir=ckpt_path)
ckpt_last = '/data/DL/Adversarial/kinetics-i3d/result/generalization/model_gen_untargeted_ce_loss_reg/model_step_00000'

model = ki3du.kinetics_i3d(ckpt_path=ckpt_last,batch_size=_BATCH_SIZE,init_pert_from_ckpt=True)

inputs = model.rgb_input
labels =model.labels

perturbation = model.eps_rgb
adversarial_inputs_rgb = model.adversarial_inputs_rgb
eps_rgb = model.eps_rgb
adv_flag = model.adv_flag
softmax = model.softmax
softmax_clean = model.softmax_clean

max_non_label_prob = model.max_non_label_prob

model_logits = model.model_logits
labels = model.labels
cyclic_flag = model.cyclic_pert_flag
norm_reg = model.norm_reg
diff_norm_reg = model.diff_norm_reg
laplacian_norm_reg = model.laplacian_norm_reg

thickness = model.thickness
roughness = model.roughness
thickness_relative= model.thickness_relative
roughness_relative = model.roughness_relative

sess=model.sess

pert = sess.run(perturbation)


#%% configs
result_path =os.environ['DL_SHARED'] +'/Adversarial/Flickering_Adversarial_paper/results/i3d/single_video_attack/'
save_result =True

IMPROVE_ADV_LOSS =True
PROB_MARGIN =0.05
TARGETED_ATTACK=True
USE_LOGITS =False

_cyclic_flag =0.0 # 1.0
_adv_flag =1.0

_lr=0.01

# regularization loss:
_beta_0 =1.# 1.0 #0.1# 1 #1 #1.0 #1.0
if sys.argv.__len__()>1:
    _beta_1 = float(sys.argv[1])
else:
    _beta_1 =1.0 #1  #1 #1 #0.05# 0.5 #0.5 #0.5 #0.1 #0.1 #0.1 #0.1 #0.1 #100000 #0.01  # 100000 #0.0001 # 0.001 #0.01 #0.1 #0.0001 #1
_beta_2 = (1.0 -_beta_1)/2
_beta_3 = (1.0 -_beta_1)/2


#%% load the video to attack
video_path ='/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/test/tai chi/xaI5OgKb6YM_000032_000042.mp4'
# video_path = os.environ['DL_SHARED'] + '/Adversarial/ActivityNet/Crawler/Kinetics/database/test/juggling balls/zP9LFfCor_M_000015_000025.mp4'
# video_path = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/test/high jump/t85lCcJlaIs_000000_000010.mp4'

rgb_sample, _ = img_tool.video_to_image_and_of(video_path,n_steps=_SAMPLE_VIDEO_FRAMES)
gt_label_name = video_path.split('/')[-2]
sample_label=kinetics_classes.index(gt_label_name)

correct_cls_id= sample_label
correct_cls = kinetics_classes[correct_cls_id]

if TARGETED_ATTACK:
    target_class = 'javelin throw' #correct_cls
else:
    target_class = correct_cls #'javelin throw' #correct_cls

#%%
res_dict={}


feed_dict_for_clean_eval = {inputs: rgb_sample, adv_flag: 0}

model_softmax = sess.run(softmax, feed_dict=feed_dict_for_clean_eval)

top_id = model_softmax.argmax()
top_id_prob = model_softmax.max()
predict_label_name = kinetics_classes[top_id]


if save_result:
    dict_result_path = os.path.join(result_path, 'res_beta_1_{}_{}'.format(_beta_1,gt_label_name.replace(' ','_')) + '.pkl')
    
 # 'javelin throw'
target_class_id = kinetics_classes.index(target_class)



if IMPROVE_ADV_LOSS:
    adversarial_loss = model.improve_adversarial_loss(margin=PROB_MARGIN,
                                                      targeted=TARGETED_ATTACK,
                                                      logits=USE_LOGITS)
else:
    adversarial_loss = model.ce_adversarial_loss(targeted=TARGETED_ATTACK)


regularizer_loss = _beta_1 * norm_reg + _beta_2 * diff_norm_reg + _beta_2 * laplacian_norm_reg  # +lab_reg

weighted_regularizer_loss = _beta_0 * regularizer_loss
loss = adversarial_loss + weighted_regularizer_loss

prob_to_min = model.to_min_prob
prob_to_max = model.to_max_prob
learning_rate_default = tf.constant(0.001, dtype=tf.float32)
learning_rate = tf.placeholder_with_default(learning_rate_default, name='learning_rate',
                                            shape=learning_rate_default.shape)
global_step = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradients = optimizer.compute_gradients(loss=loss, var_list=perturbation)
train_op = optimizer.apply_gradients(gradients,global_step)




feed_dict_for_train = {inputs: rgb_sample,
                       labels: [target_class_id],
                       cyclic_flag: _cyclic_flag,
                       adv_flag: _adv_flag}

res_dict['correct_cls_prob']=top_id_prob
res_dict['correct_cls'] = correct_cls
res_dict['correct_cls_id'] = correct_cls_id
res_dict['softmax_init'] =model_softmax
res_dict['rgb_sample'] = rgb_sample



total_loss_l=[]
adv_loss_l = []
reg_loss_l= []
norm_reg_loss_l=[]
diff_norm_reg_loss_l=[]
laplacian_norm_reg_l = []
roughness_l = []
thickness_l=[]

_model_softmax=[]
_perturbation=[]

correct_cls_prob_l=[]
max_non_correct_cls_prob_l=[]
max_prob_l=[]


step = 0
max_step =2500

sess.run(eps_rgb.initializer)
sess.run([global_step.initializer,tf.variables_initializer(optimizer.variables())])

plt.figure(1)
plt.figure(2)

##
#%%
while True:  # to repeat with decreased epsilons if necessary

    _, total_loss, adv_loss, reg_loss, norm_reg_loss, diff_norm_reg_loss, _laplacian_norm_reg, _thickness, _roughness,_max_non_label_prob = \
    sess.run(fetches=[train_op, loss, adversarial_loss, regularizer_loss, norm_reg,diff_norm_reg, laplacian_norm_reg, thickness, roughness,max_non_label_prob],
             feed_dict=feed_dict_for_train)

    curr_model_softmax =sess.run(softmax, feed_dict={inputs: rgb_sample})
    _model_softmax.append(curr_model_softmax)
    
    correct_cls_prob_l.append( curr_model_softmax[0, correct_cls_id])
    max_class_id=curr_model_softmax[0].argmax()
    tareget_class_prob= curr_model_softmax[0, target_class_id]
    max_prob_l.append(curr_model_softmax[0].max())
    max_non_correct_cls_prob_l.append(_max_non_label_prob)

    print(
        "Step: {:05d}, Total Loss: {:.5f}, Cls Loss: {:.5f}, Total Reg Loss: {:.5f}, Fat Loss: {:.5f}, Diff Loss: {:.5f}, prob_correct_cls: {:.5f}"
        ", top_prob: {:.5f}, target_prob:{:.5f}, thickness: {:.5f} ({:.2f} %), roughness: {:.5f} ({:.2f} %)".format(step, total_loss,
                                                                                               adv_loss,
                                                                                               reg_loss,
                                                                                               norm_reg_loss,
                                                                                               diff_norm_reg_loss,
                                                                                               curr_model_softmax[
                                                                                                   0, correct_cls_id],
                                                                                               curr_model_softmax[
                                                                                                   0].max(),
                                                                                               tareget_class_prob,
                                                                                               _thickness,
                                                                                               _thickness / 2.0 * 100,
                                                                                               _roughness,
                                                                                               _roughness / 2.0 * 100))


    total_loss_l.append(total_loss)
    adv_loss_l.append(adv_loss)
    reg_loss_l.append(reg_loss)
    norm_reg_loss_l.append(norm_reg_loss)
    diff_norm_reg_loss_l.append(diff_norm_reg_loss)
    laplacian_norm_reg_l.append(_laplacian_norm_reg)
    thickness_l.append(_thickness / 2.0 * 100)
    roughness_l.append(_roughness / 2.0 * 100)
    
    pert = sess.run(perturbation)
    _perturbation.append(pert)
    
    plt.title('beta_1: {:.2f}'.format(_beta_1))
    if np.random.rand() > 1.1:
        plt.figure(1)
        plt.clf()
        plt.subplot(4,1,2)
        plt.plot(total_loss_l,'r')
        plt.plot(adv_loss_l,'b')
        plt.plot(reg_loss_l,'g')
        plt.plot(norm_reg_loss_l,'k')
        plt.plot(diff_norm_reg_loss_l,'m')
        plt.grid(True)


        plt.subplot(4,1,1)
        plt.plot(reg_loss_l,'g')
        plt.plot(norm_reg_loss_l,'k')
        plt.plot(diff_norm_reg_loss_l,'m')
        plt.plot(laplacian_norm_reg_l, 'b')

        plt.subplot(4,1,3)
        plt.plot(thickness_l,'k')
        plt.plot(roughness_l,'m')
        
        
        plt.subplot(4,1,4)
        plt.plot(correct_cls_prob_l,'r')
        plt.plot(max_prob_l,'-g')
        plt.plot(max_non_correct_cls_prob_l,'-b')
        
        plt.grid(True)

        # plt.draw()
        
        plt.show(block=False)
        plt.pause(0.1)
        
        plt.figure(2)
        plt.clf()
        plt.plot(pert.squeeze()[...,0]/2.*100.,'r')
        plt.plot(pert.squeeze()[...,1]/2.*100.,'g')
        plt.plot(pert.squeeze()[...,2]/2.*100.,'b')
        
        plt.grid(True)
        plt.show(block=False)
        plt.pause(0.1)


        # _, total_loss, adv_loss = sess.run(fetches=[train_op, loss, ce_loss_mean],
    #                                              feed_dict=feed_dict)
    # print("Total Loss: {} , Cls Loss: {}  ".format(total_loss, adv_loss))


    _model_logits =  sess.run(model_logits, feed_dict={inputs: rgb_sample})
    # is_adversarial = criteria.is_adversarial(model_softmax, target_class)

    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    # summary = sess.run(merged, feed_dict={inputs: rgb_sample},
    #                    options=run_options,
    #                    run_metadata=run_metadata)
    # writer.add_summary(summary, step)

    is_adversarial = correct_cls_id!=max_class_id if not TARGETED_ATTACK else max_class_id==target_class_id

    if  step > max_step and is_adversarial :
        res_dict['total_loss_l'] = total_loss_l
        res_dict['adv_loss_l'] = adv_loss_l
        res_dict['reg_loss_l'] = reg_loss_l
        res_dict['norm_reg_loss_l'] = norm_reg_loss_l
        res_dict['diff_norm_reg_loss_l'] = diff_norm_reg_loss_l
        res_dict['perturbation'] = _perturbation
        res_dict['adv_video'] = sess.run(adversarial_inputs_rgb, feed_dict={inputs: rgb_sample})
        res_dict['softmax'] = _model_softmax
        res_dict['correct_cls_prob'] = correct_cls_prob_l
        res_dict['total_steps'] = step
        res_dict['beta_0'] = _beta_0
        res_dict['beta_1'] = _beta_1
        res_dict['beta_2'] = _beta_2
        res_dict['beta_3'] = _beta_3
        res_dict['thickness'] = thickness_l
        res_dict['roughness'] = roughness_l

        with open(dict_result_path, 'wb') as file:
                pickle.dump(res_dict, file)
        break

    step+=1
    
    
    if  step > 3500:
        break
    # _, total_loss, adv_loss, reg_loss = sess.run(fetches=[train_op, loss, ce_loss_mean,regularizer_loss], feed_dict=feed_dict)
    # print("Total Loss: {} , Cls Loss: {} , Reg Loss: {} ".format(total_loss, adv_loss, reg_loss))
    # pert = sess.run(perturbation)



