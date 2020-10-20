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
sys.path.insert(1, '/media/ROIPO/Data/projects/Adversarial/kinetics-i3d/')

sys.path.insert(1, os.path.realpath(os.path.pardir))

import foolbox_base.foolbox as fb_0
import foolbox_2.foolbox as fb_2
import i3d
# import skvideo
import pre_process_rgb_flow as img_tool



_IMAGE_SIZE = 224
_BATCH_SIZE = 5

_SAMPLE_VIDEO_FRAMES = 90 #250 #90 #79
_BASE_PATCH_FRAMES = 1 #_SAMPLE_VIDEO_FRAMES # 1:for sticker _SAMPLE_VIDEO_FRAMES # 1
_IND_START = 0  # 0 #50
_IND_END = _SAMPLE_VIDEO_FRAMES

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

test_list_path  = '/media/ROIPO/Data/projects/Adversarial/database/UCF-101/ucfTrainTestlist/testlist01.txt'

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



def main(unused_argv):

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


    ucf_video_list = [x.strip() for x in open(test_list_path)]

    with tf.variable_scope(scope):

        sess = tf.Session()

        rgb_sample = np.load('/media/ROIPO/Data/projects/Adversarial/kinetics-i3d/data/triple_jump_1_kinetics.npy')
        correct_cls = 'triple jump'
        # rgb_sample, flow_sample, correct_cls = get_video_sample(ucf_video_list)

        default_adv_flag = tf.constant(1.0,dtype=tf.float32)
        adv_flag = tf.placeholder_with_default(default_adv_flag,shape=default_adv_flag.shape)



        # RGB input has 3 channels.
        rgb_input = tf.placeholder(tf.float32,
            shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

        eps_rgb = tf.Variable(tf.zeros(shape=[_BASE_PATCH_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3], dtype=tf.float32),name='eps')

        mask = tf.ones(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])
        # mask = np.zeros(rgb_input.shape, dtype=np.float32)
        # mask[0,:,112,112,:]=1
        # mask = tf.constant(mask)

        #_SAMPLE_VIDEO_FRAMES# 50 #_SAMPLE_VIDEO_FRAMES # _SAMPLE_VIDEO_FRAMES
        default_T = tf.constant(int(_IND_END - _IND_START + 1), dtype=tf.int32)
        T = tf.placeholder_with_default(default_T,
                                        shape=default_T.shape)

        indices = tf.cast(tf.linspace(float(_IND_START),float(_IND_END),T), tf.int32)

        # indices = np.linspace(_IND_START,_IND_END,_IND_END-_IND_START+1)
        mask_indecator = tf.one_hot(indices =indices, depth=_SAMPLE_VIDEO_FRAMES)
        mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
        mask_indecator = tf.reshape(mask_indecator, [_SAMPLE_VIDEO_FRAMES,1,1,1])
        mask_rgb = tf.convert_to_tensor(mask*mask_indecator,name='eps_mask') # same shape as input
        adversarial_inputs_rgb = tf.nn.tanh(rgb_input + adv_flag * (mask_rgb * eps_rgb),name='adversarial_input')

    rgb_model = load_i3d_model(num_classes=NUM_CLASSES)
    init_model(model=rgb_model,sess=sess, ckpt_path=ckpt_path,eval_type=eval_type)

    model_logits, _ = rgb_model(adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)
    model_predictions = tf.nn.softmax(model_logits)


    feed_dict={}
    feed_dict[rgb_input] = rgb_sample
    sess.run(eps_rgb.initializer)

    rgb_adv_model = fb_2.models.TensorFlowModel(inputs=rgb_input,
                                                adversarial_inputs=adversarial_inputs_rgb,
                                                perturbation=eps_rgb,
                                                mask=mask_rgb,
                                                logits=model_logits,
                                                T = T,
                                                bounds=(-1, 1), session=sess)

    # target_class = kinetics_classes.index(correct_cls)
    criteria = fb_2.criteria.Misclassification()

    # target_class = kinetics_classes.index('javelin throw')
    # criteria = fb_2.criteria.TargetClass(target_class=target_class)
    # criteria = fb_2.criteria.TargetClassProbability(target_class=target_class, p=0.85)
    #
    target_class = kinetics_classes.index(correct_cls)
    #
    # # target_class =30
    # # criteria = fb_2.criteria.TargetClass(target_class=target_class)
    #
    # # attack = fb_2.attacks.FGSM(model=rgb_adv_model, criterion=criteria)
    #
    attack = fb_2.attacks.MultiStepGradientBaseAttack(model=rgb_adv_model, criterion=criteria)
    #
    #
    perturbed=[]
    for t in range(1,90):
        rgb_adversarial = attack(rgb_sample.squeeze(), label=target_class, unpack=False,max_epsilon=t)
        perturbed.append(rgb_adversarial.perturbed)
        sess.run(eps_rgb.initializer)

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
