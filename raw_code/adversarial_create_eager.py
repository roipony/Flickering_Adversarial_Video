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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
import foolbox_base.foolbox as fb_0
import foolbox_2.foolbox as fb_2
import i3d_tf2 as i3d
# import skvideo
import pre_process_rgb_flow as img_tool

_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 79
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
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('eval_type', 'rgb', 'rgb, rgb600, flow, or joint')
tf.compat.v1.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def main(unused_argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  eval_type = FLAGS.eval_type

  imagenet_pretrained = FLAGS.imagenet_pretrained

  NUM_CLASSES = 400
  if eval_type == 'rgb600':
    NUM_CLASSES = 600

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

  if eval_type == 'rgb600':
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
  else:
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    # rgb_input = tf.placeholder(
    #     tf.float32,
    #     shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

    rgb_input = tf.keras.Input(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3],batch_size=1, dtype=tf.float32)
    # eps_rgb = tf.Variable(tf.zeros(shape=[1, 1, _IMAGE_SIZE, _IMAGE_SIZE, 3], dtype=tf.float32))
    # mask = tf.ones_like(rgb_input)
    #
    # ind_start = 0
    # ind_end = _SAMPLE_VIDEO_FRAMES # _SAMPLE_VIDEO_FRAMES
    # indices = np.linspace(ind_start,ind_end,ind_end+1)
    # mask_indecator = tf.one_hot(indices =indices, depth=_SAMPLE_VIDEO_FRAMES)
    # mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
    # mask_indecator = tf.reshape(mask_indecator, [1,_SAMPLE_VIDEO_FRAMES,1,1,1])
    # mask_rgb = mask*mask_indecator
    # mask = tf.ones_like(eps_rgb)
    # paddings = tf.constant([[0, 0, ], [39, 39], [0, 0], [0, 0], [0, 0]])
    # mask = tf.pad(mask, paddings)

    # rgb_input_ =tf.Variable(rgb_input)
    # rgb_input_ = tf.assign(rgb_input_[0, 40, ...], rgb_input_[0, 40, ...] + eps_rgb)
    # tf.assign(rgb_input[0, 40, ...], rgb_input[0, 40, ...]+eps_rgb)
    # rgb_input[1,40,...] += eps_rgb
    with tf.compat.v1.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits',dropout_keep_prob=1.0)
      adversarial_inputs_rgb = rgb_input #+ mask_rgb*eps_rgb
      rgb_logits, _ = rgb_model(
        adversarial_inputs_rgb, is_training=False)


    rgb_variable_map = {}
    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  # eps = tf.placeholder(
  #       tf.float32,
  #       shape=(1, 1, _IMAGE_SIZE, _IMAGE_SIZE, 2))

  # eps = tf.constant(np.zeros(shape=[1, 1, _IMAGE_SIZE, _IMAGE_SIZE, 2]),dtype=tf.float32)
  # eps = tf.Variable(tf.zeros(shape=[1, 1, _IMAGE_SIZE, _IMAGE_SIZE, 2],dtype=tf.float32))

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))

    eps_flow= tf.Variable(tf.zeros(shape=[1, 1, _IMAGE_SIZE, _IMAGE_SIZE, 2], dtype=tf.float32))
    mask = tf.ones_like(flow_input)
    # paddings = tf.constant([[0, 0, ], [39, 39], [0, 0], [0, 0], [0, 0]])
    # mask = tf.pad(mask, paddings)
    indices = np.linspace(0,_SAMPLE_VIDEO_FRAMES,_SAMPLE_VIDEO_FRAMES+1)
    mask_indecator = tf.one_hot(indices =indices, depth=_SAMPLE_VIDEO_FRAMES)
    mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
    mask_indecator = tf.reshape(mask_indecator, [1,_SAMPLE_VIDEO_FRAMES,1,1,1])
    mask_flow = mask*mask_indecator


    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')

      adversarial_inputs_flow = flow_input + mask_flow* eps_flow

      flow_logits, _ = flow_model(
        adversarial_inputs_flow, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb' or eval_type == 'rgb600':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)

  with tf.Session() as sess:
    feed_dict = {}
    if eval_type in ['rgb', 'rgb600', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
      tf.logging.info('RGB checkpoint restored')
      rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = rgb_sample

      sess.run(eps_rgb.initializer)
      rgb_adv_model = fb_2.models.TensorFlowModel(inputs=rgb_input,
                                                  adversarial_inputs=adversarial_inputs_rgb,
                                                  perturbation= eps_rgb,
                                                  mask = mask_rgb,
                                                  logits=rgb_logits,
                                                  bounds=(-1, 1))
      # criteria = fb_2.criteria.ConfidentMisclassification(p=0.9)
      criteria = fb_2.criteria.Misclassification()
      # attack = fb_2.attacks.FGSM(model=rgb_adv_model, criterion=criteria)
      attack = fb_2.attacks.MultiStepGradientBaseAttack(model=rgb_adv_model, criterion=criteria)


      rgb_adversarial = attack(rgb_sample.squeeze(), 227,unpack=False)
      # adv_image = rgb_adv_model.session.run(rgb_adv_model._pert)

    if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.logging.info('Flow checkpoint restored')
      flow_sample = np.load(_SAMPLE_PATHS['flow'])
      tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))

      # eps = tf.Variable(tf.zeros(shape = [1,224,224,2]))
      # flow_input = flow_input + eps

      feed_dict[flow_input] = flow_sample
      # feed_dict[eps] = np.zeros(shape = [1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2])

      sess.run(eps_flow.initializer)
      flow_adv_model = fb_2.models.TensorFlowModel(inputs=flow_input,
                                                   adversarial_inputs=adversarial_inputs_flow,
                                                   perturbation=eps_flow,
                                                   mask = mask_flow,
                                                   logits=flow_logits,
                                                   bounds=(-1, 1))
      criteria = fb_2.criteria.ConfidentMisclassification(p=0.9)
      attack = fb_2.attacks.FGSM(model=flow_adv_model,criterion=criteria)
      flow_adversarial =  attack(flow_sample.squeeze(), 227, unpack=False)
      # adv_image = flow_adv_model.session.run(flow_adv_model._dx)

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


if __name__ == '__main__':
  tf.compat.v1.app.run(main)
