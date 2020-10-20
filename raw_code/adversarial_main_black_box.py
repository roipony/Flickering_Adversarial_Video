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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
# tf.enable_eager_execution()
# sys.path.insert(1, '../Adversarial/kinetics-i3d/')

sys.path.insert(1, os.path.realpath(os.path.pardir))
sys.path.insert(1, 'contextualLoss')


import foolbox_2.foolbox as fb_2
import i3d
# import skvideo
import pre_process_rgb_flow as img_tool

from CX.CX_helper import *
from model import *
from utils.FetchManager import *

_IMAGE_SIZE = 224
_BATCH_SIZE = 5

_SAMPLE_VIDEO_FRAMES = 90 #90 #250 #90 #79
_BASE_PATCH_FRAMES = 1 #_SAMPLE_VIDEO_FRAMES # 1:for sticker _SAMPLE_VIDEO_FRAMES # 1
_IND_START = 50  # 0 #50
_IND_END = 50 #_SAMPLE_VIDEO_FRAMES

_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
    'rgb_ucf_101': 'data/checkpoints/rgb_ucf_101/ucf101_rgb_0.946_model-44520.ckpt'
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'
_LABEL_MAP_PATH_UCF_101 = 'data/label_map_ucf_101.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'rgb', 'rgb, rgb600, ,  flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

# test_list_path  = '/media/ROIPO/Data/projects/Adversarial/database/UCF-101/ucfTrainTestlist/testlist01.txt'

def get_video_sample(vid_list, random=True, id=0 ):
      base_path = '/media/ROIPO/Data/projects/Adversarial/database/UCF-101/video/'

      if random:
        id = np.random.choice(a=vid_list.__len__(), size=1)[0]
      cls, vid_name = vid_list[id].split('/')
      frames, flow_frames = img_tool.video_to_image_and_of(video_path=base_path+vid_name,n_steps=80)
      return frames, flow_frames, cls


def load_i3d_model(num_classes,eval_type='rgb', scope='RGB',spatial_squeeze=True, final_endpoint='Logits'):
    with tf.variable_scope(scope):
        i3d_model = i3d.InceptionI3d(
          num_classes, spatial_squeeze=spatial_squeeze, final_endpoint=final_endpoint)

    dummy_input = tf.placeholder(
        tf.float32,
        shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    i3d_model(dummy_input, is_training=False, dropout_keep_prob=1.0)


    return i3d_model

def init_model(model,sess, ckpt_path, eval_type='rgb'):
    rgb_variable_map = {}

    for variable in model.get_all_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    for variable in model.graph.get_collection_ref('moving_average_variables'):

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    saver.restore(sess,ckpt_path)

def load_kinetics_classes(eval_type):
    if eval_type == 'rgb600':
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    else:
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    return kinetics_classes

def cx_loss(inputs,adversarial_inputs):
    vgg_real = build_vgg19(tf.squeeze(inputs))
    vgg_input = build_vgg19(tf.squeeze(adversarial_inputs))

    if config.W.CX > 0:
        CX_loss_list = [w * CX_loss_helper(vgg_real[layer], vgg_input[layer], config.CX)
                        for layer, w in config.CX.feat_layers.items()]
        CX_style_loss = tf.reduce_sum(CX_loss_list)
        CX_style_loss *= config.W.CX
    else:
        CX_style_loss = zero_tensor

    return CX_style_loss

def main(unused_argv):
    criteria = fb_2.criteria.Misclassification()
    logdir ='logs/'
    # from tensorflow.contrib.slim.nets import vgg
    #
    # images = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    # preprocessed = images - [123.68, 116.78, 103.94]
    # logits, _ = vgg.vgg_19(preprocessed, is_training=False)
    # restorer = tf.train.Saver(tf.trainable_variables())
    #
    # image, label = fb_2.utils.imagenet_example()
    # eps_rgb = tf.Variable(tf.zeros(shape=[1, 224, 224, 3], dtype=tf.float32),name='eps')
    # adversarial_inputs_rgb = preprocessed +  eps_rgb
    #
    # with tf.Session() as session:
    #   restorer.restore(session, 'vgg_19.ckpt')
    #   model = fb_2.models.TensorFlowModelCX(inputs=images,adversarial_inputs=adversarial_inputs_rgb,
    #   perturbation=eps_rgb,logits=logits,mask=None,bounds=(0, 255), session=session)
    #   print(np.argmax(model.forward_one(image)))
    #   criteria = fb_2.criteria.Misclassification()

      # target_class = kinetics_classes.index('javelin throw')
      # criteria = fb_2.criteria.TargetClass(target_class=target_class)
      # criteria = fb_2.criteria.TargetClassProbability(target_class=target_class, p=0.85)
      #
      # target_class = label
      #
      # # target_class =30
      # # criteria = fb_2.criteria.TargetClass(target_class=target_class)
      #
      # # attack = fb_2.attacks.FGSM(model=rgb_adv_model, criterion=criteria)
      #
      # attack = fb_2.attacks.MultiStepGradientBaseAttack(model=model, criterion=criteria)
      # rgb_adversarial = attack(image.squeeze(), label=target_class, unpack=False)

    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained

    NUM_CLASSES = 400
    if eval_type == 'rgb600':
            NUM_CLASSES = 600

    scope ='RGB'
    kinetics_classes = load_kinetics_classes(eval_type)

    if eval_type in ['rgb', 'rgb600', 'joint']:
      if imagenet_pretrained:
        ckpt_path = _CHECKPOINT_PATHS['rgb_imagenet']
      else:
        ckpt_path = _CHECKPOINT_PATHS[eval_type]


    # ucf_video_list = [x.strip() for x in open(test_list_path)]

    sess = tf.Session()
    with tf.variable_scope(scope):



        rgb_sample = np.load('data/triple_jump_1_kinetics.npy')[0,-_SAMPLE_VIDEO_FRAMES:]
        rgb_sample = rgb_sample[np.newaxis]
        correct_cls = 'triple jump'
        target_class = kinetics_classes.index(correct_cls)
        # rgb_sample, flow_sample, correct_cls = get_video_sample(ucf_video_list)

        default_adv_flag = tf.constant(1.0,dtype=tf.float32)
        adv_flag = tf.placeholder_with_default(default_adv_flag,shape=default_adv_flag.shape)

        # RGB input has 3 channels.
        rgb_input = tf.placeholder(tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

        eps_rgb = tf.Variable(tf.zeros(shape=[_BASE_PATCH_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3], dtype=tf.float32),name='eps')

        mask = tf.ones(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])
        # mask = np.zeros(rgb_input.shape, dtype=np.float32)
        # mask[0,:,112,112,:]=1
        # mask = tf.constant(mask)

        #_SAMPLE_VIDEO_FRAMES# 50 #_SAMPLE_VIDEO_FRAMES # _SAMPLE_VIDEO_FRAMES
        # default_T = tf.constant(int(_IND_END - _IND_START + 1), dtype=tf.int32)
        # T = tf.placeholder_with_default(default_T,
        #                                 shape=default_T.shape)

        # indices = tf.cast(tf.linspace(float(_IND_START),float(_IND_END),T), tf.int32)

        indices = np.linspace(_IND_START,_IND_END,_IND_END-_IND_START+1)
        mask_indecator = tf.one_hot(indices =indices, depth=_SAMPLE_VIDEO_FRAMES)
        mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
        mask_indecator = tf.reshape(mask_indecator, [_SAMPLE_VIDEO_FRAMES,1,1,1])
        mask_rgb = tf.convert_to_tensor(mask*mask_indecator,name='eps_mask') # same shape as input
        # adversarial_inputs_rgb = tf.nn.tanh(rgb_input + adv_flag * (mask_rgb * eps_rgb),name='adversarial_input')
        adversarial_inputs_rgb = tf.clip_by_value(rgb_input + adv_flag * (mask_rgb * eps_rgb),
                                                  clip_value_min = -1.0,
                                                  clip_value_max = 1.0,
                                                  name='adversarial_input')


    rgb_model = load_i3d_model(num_classes=NUM_CLASSES)
    init_model(model=rgb_model,sess=sess, ckpt_path=ckpt_path,eval_type=eval_type)
    sess.run(eps_rgb.initializer)
    model_logits, _ = rgb_model(adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)
    softmax = tf.nn.softmax(logits = model_logits)

    inputs = rgb_input
    perturbation = eps_rgb

    # target label: for untargeted attack set original label
    labels = tf.placeholder(tf.int64, (None,), name='labels')
    # for untargeted attack set to -1.0
    label_coeff_default = tf.constant(-1.0, dtype=tf.float32)
    labels_coeff = tf.placeholder_with_default(label_coeff_default, name='label_coeff', shape=label_coeff_default.shape)


    # adversarial classification loss
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model_logits)
    ce_loss_mean = tf.reduce_mean(labels_coeff *ce_loss)


    # regularization loss:
    beta = 10000 #100000 #0.01  # 100000 #0.0001 # 0.001 #0.01 #0.1 #0.0001 #1
    # regularizer = tf.reduce_max(perturbation) - tf.reduce_min(perturbation)
    with tf.device('/gpu:1'):
        regularizer_loss = cx_loss(tf.expand_dims(inputs[0,_IND_START:_IND_END+1,...],axis=0),tf.expand_dims(adversarial_inputs_rgb[0,_IND_START:_IND_END+1,...],axis=0))
        weighted_regularizer_loss = beta * regularizer_loss

    # total loss:
    loss = ce_loss_mean + weighted_regularizer_loss

    optimizer = tf.train.AdamOptimizer()
    gradients = optimizer.compute_gradients(loss=loss, var_list=perturbation)
    train_op = optimizer.apply_gradients(gradients)
    sess.run(tf.variables_initializer(optimizer.variables()))

    writer = tf.summary.FileWriter(logdir, sess.graph)
    with tf.name_scope('input'):
        clean_summary = tf.summary.image('clean_image', tf.expand_dims(inputs[0, _IND_START], axis=0), max_outputs=inputs.shape[0].value)
        pert_summary = tf.summary.image('perturbation', perturbation, max_outputs=perturbation.shape[0].value)
        adv_summary = tf.summary.image('adversarial_image', tf.expand_dims(adversarial_inputs_rgb[0,_IND_START],axis=0), max_outputs=adversarial_inputs_rgb.shape[0].value)

    adv_vec = np.zeros(shape=[1,_SAMPLE_VIDEO_FRAMES])
    merged = tf.summary.merge_all()

    feed_dict = {inputs: rgb_sample, labels: [target_class], labels_coeff: -1.0}
    model_softmax = sess.run(softmax, feed_dict={inputs: rgb_sample})

    base_prob =model_softmax[0,target_class]
    for step in range(10000000000):  # to repeat with decreased epsilons if necessary

        tmp_adv_vec = adv_vec.copy()
        epsilon = np.random.choice([-0.01, 0.01], 1)
        rand_index = np.random.choice(_SAMPLE_VIDEO_FRAMES, 1)
        tmp_adv_vec[0,rand_index]+=epsilon
        tmp_adv_vec = np.clip(tmp_adv_vec,a_min=0,a_max=0.1)

        rgb_sample_tmp = rgb_sample.copy()
        for i in range(rgb_sample_tmp.shape[1]):
            rgb_sample_tmp[0, i, ...] += tmp_adv_vec[0, i]

        feed_dict = {inputs: np.clip(rgb_sample_tmp,a_min=-1.,a_max=1.), labels: [target_class],labels_coeff: -1.0}
        model_softmax = sess.run(softmax, feed_dict={inputs: rgb_sample_tmp})


        if model_softmax[0,target_class] < base_prob:
            base_prob = model_softmax[0,target_class]
            adv_vec = tmp_adv_vec.copy()
            print("step: {} : {}".format(step,model_softmax[0, target_class]))

            # print(':-)')

        is_adversarial = criteria.is_adversarial(model_softmax, target_class)

        if  is_adversarial:
            a=1

        # _, total_loss, adv_loss, reg_loss = sess.run(fetches=[train_op, loss, ce_loss_mean,regularizer_loss], feed_dict=feed_dict)
        # print("Total Loss: {} , Cls Loss: {} , Reg Loss: {} ".format(total_loss, adv_loss, reg_loss))
        # pert = sess.run(perturbation)



    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary = sess.run(merged, feed_dict={inputs: rgb_sample},
                       options=run_options,
                       run_metadata=run_metadata)
    writer.add_summary(summary,step)
    # if is_adversarial:
    #     return

    out_logits, out_predictions = sess.run(
        [model_logits, model_predictions],
        feed_dict=feed_dict)

    out_logits = out_logits[0]
    out_predictions = out_predictions[0]
    sorted_indices = np.argsort(out_predictions)[::-1]

    print('Norm of logits: %f' % np.linalg.norm(out_logits))
    print('\nTop classes and probabilities')
    for index in sorted_indices[:20]:
      print(out_predictions[index], out_logits[index], kinetics_classes[index])


    feed_dict[adv_flag]=0.0
    out_logits, out_predictions = sess.run(
        [model_logits, model_predictions],
        feed_dict=feed_dict)

    out_logits = out_logits[0]
    out_predictions = out_predictions[0]
    sorted_indices = np.argsort(out_predictions)[::-1]

    print('Norm of logits: %f' % np.linalg.norm(out_logits))
    print('\nTop classes and probabilities')
    for index in sorted_indices[:20]:
      print(out_predictions[index], out_logits[index], kinetics_classes[index])

    a=1




if __name__ == '__main__':
  tf.app.run(main)


#%%
