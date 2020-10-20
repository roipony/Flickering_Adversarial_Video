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

import sys
import os
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
sys.path.insert(1, '/media/ROIPO/Data/projects/Adversarial/kinetics-i3d/')

sys.path.insert(1, os.path.realpath(os.path.pardir))

# import foolbox_base.foolbox as fb_0
import foolbox_2.foolbox as fb_2
import i3d
# import skvideo
import pre_process_rgb_flow as img_tool

_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 250 #90 #79
_BATCH_SIZE = 2
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

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
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

    # ucf_video_list = [x.strip() for x in open(test_list_path)]

    rgb_sample = np.load('/data/DL/Adversarial/kinetics-i3d/data/npy_videos/7pxmupXYnuo.npy')[0, -_SAMPLE_VIDEO_FRAMES:]
    correct_cls = 'triple jump'
    # rgb_sample, flow_sample, correct_cls = get_video_sample(ucf_video_list)

    default_adv_flag = tf.constant(1.0,dtype=tf.float32)
    adv_flag = tf.placeholder_with_default(default_adv_flag,
      shape=default_adv_flag.shape)


  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

    eps_rgb = tf.Variable(tf.zeros(shape=[1, _IMAGE_SIZE, _IMAGE_SIZE, 3], dtype=tf.float32))
    # eps_rgb = tf.Variable(tf.zeros(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3], dtype=tf.float32))

    mask = tf.ones(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])
    # mask = np.zeros(rgb_input.shape, dtype=np.float32)
    # mask[0,:,112,112,:]=1
    # mask = tf.constant(mask)
    ind =tf.random.uniform(shape=(),minval=0, maxval=_SAMPLE_VIDEO_FRAMES)
    # ind =tf.placeholder(tf.float32,None)
    ind_start =125.0 # 0.0 # 125.0 #ind#0 #50
    ind_end =125.0 #1.0*_SAMPLE_VIDEO_FRAMES # 125.0 # ind #_SAMPLE_VIDEO_FRAMES #_SAMPLE_VIDEO_FRAMES# 50 #_SAMPLE_VIDEO_FRAMES # _SAMPLE_VIDEO_FRAMES
    indices = tf.cast(tf.linspace(ind_start,ind_end,tf.cast(ind_end-ind_start+1,tf.int32)), tf.int32)
    mask_indecator = tf.one_hot(indices =indices, depth=_SAMPLE_VIDEO_FRAMES)
    mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
    mask_indecator = tf.reshape(mask_indecator, [_SAMPLE_VIDEO_FRAMES,1,1,1])
    mask_rgb = mask*mask_indecator
    # mask = tf.ones_like(eps_rgb)
    # paddings = tf.constant([[0, 0, ], [39, 39], [0, 0], [0, 0], [0, 0]])
    # mask = tf.pad(mask, paddings)

    # rgb_input_ =tf.Variable(rgb_input)
    # rgb_input_ = tf.assign(rgb_input_[0, 40, ...], rgb_input_[0, 40, ...] + eps_rgb)
    # tf.assign(rgb_input[0, 40, ...], rgb_input[0, 40, ...]+eps_rgb)
    # rgb_input[1,40,...] += eps_rgb
    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      adversarial_inputs_rgb = tf.nn.tanh(rgb_input + adv_flag*(mask_rgb*eps_rgb))
      # adversarial_inputs_rgb = tf.nn.tanh(rgb_input + adv_flag*eps_rgb)

      rgb_logits, _ = rgb_model(
        adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)


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

      adversarial_inputs_flow = flow_input + adv_flag*(mask_flow* eps_flow)

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
      #rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = rgb_sample

      sess.run(eps_rgb.initializer)
      rgb_adv_model = fb_2.models.TensorFlowModel(inputs=rgb_input,
                                                  adversarial_inputs=adversarial_inputs_rgb,
                                                  perturbation= eps_rgb,
                                                  mask = mask_rgb,
                                                  logits=rgb_logits,
                                                  bounds=(-1, 1))
      if 0:
          
          # target_class = kinetics_classes.index(correct_cls)
          # criteria = fb_2.criteria.Misclassification()
          # criteria = fb_2.criteria.ConfidentMisclassification(p=0.85)

          target_class = kinetics_classes.index('javelin throw')
          criteria = fb_2.criteria.TargetClass(target_class=target_class)

          # target_class = kinetics_classes.index(correct_cls)
          #
          # # target_class =30
          # # criteria = fb_2.criteria.TargetClass(target_class=target_class)
          #
          # # attack = fb_2.attacks.FGSM(model=rgb_adv_model, criterion=criteria)
          #
          attack = fb_2.attacks.MultiStepGradientBaseAttack(model=rgb_adv_model, criterion=criteria)
          #
          #
          rgb_adversarial = attack(rgb_sample.squeeze(), label=target_class,unpack=False)
          # adv_image = rgb_adv_model.session.run(rgb_adv_model._pert)
#%%
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
    #
    # label_coeff = -1.0
    #     else:
    #     label_coeff =  1.0
    #
    #     x = a.unperturbed
    #     min_, max_ = a.bounds()
    if 1:

      train_tf_record_path ='data/tfrecords/dbg_train_tfrecords.txt'
      test_tf_record_path ='data/tfrecords/dbg_train_tfrecords.txt'

      # valid_tf_record_path = '/media/ROIPO/Data/projects/Adversarial/kinetics-i3d/data/tfrecords/valid_tfrecords.txt'

      train_tf_record_list=[x.strip() for x in open(train_tf_record_path)]
      test_tf_record_list=[x.strip() for x in open(test_tf_record_path)]


      trainset = tf.data.TFRecordDataset(filenames=train_tf_record_list, num_parallel_reads=train_tf_record_list.__len__())
      trainset = trainset.map(map_func=img_tool.parse_example, num_parallel_calls=_BATCH_SIZE)
      trainset = trainset.batch(batch_size=_BATCH_SIZE,drop_remainder=False).prefetch(_BATCH_SIZE)

      testset = tf.data.TFRecordDataset(filenames=test_tf_record_list, num_parallel_reads=test_tf_record_list.__len__())
      testset = testset.map(map_func=img_tool.parse_example, num_parallel_calls=_BATCH_SIZE)
      testset = testset.batch(batch_size=_BATCH_SIZE,drop_remainder=False).prefetch(_BATCH_SIZE)

      global_step = tf.train.get_or_create_global_step()


      trainset_itr_init = trainset.make_initializable_iterator()
      trainset_itr = trainset_itr_init.get_next()

      testset_itr_init = testset.make_initializable_iterator()
      testset_itr = testset_itr_init.get_next()

      optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

      grad = optimizer.compute_gradients(loss= rgb_adv_model._loss, var_list=rgb_adv_model._pert)
      #     # self._optimizer = tf.train.AdamOptimizer()
      #     # train_op = self._optimizer.apply_gradients(zip([grad[0][0]], [a._model._pert]))
      train_op = optimizer.apply_gradients(grads_and_vars=grad,global_step=global_step)


      rgb_adv_model.session.run(tf.variables_initializer(var_list=[global_step]))
      rgb_adv_model.session.run(tf.variables_initializer(var_list=optimizer.variables()))


      tf.summary.scalar('train/total_loss', rgb_adv_model._loss)

      pertubation = rgb_adv_model._pert - tf.reduce_min(rgb_adv_model._pert)
      pertubation = pertubation / tf.reduce_max(pertubation)
      pertubation = pertubation*255.


      tf.summary.image('train/pertubation', pertubation)
      logdir = '/media/ROIPO/Data/projects/Adversarial/kinetics-i3d/log/train/'

      summary_writer_train = tf.summary.FileWriter(logdir + "/0005", graph=sess.graph)
      summary_merge = tf.summary.merge_all()

      mini_batch = 10
      # target_class = kinetics_classes.index('javelin throw')
      target_class = kinetics_classes.index(correct_cls)

      acc=0
      N=0
      try:
          sess.run(testset_itr_init.initializer)
          while True:
              sample = sess.run(testset_itr)
              softmax = rgb_adv_model.session.run(fetches=rgb_adv_model._softmax,
                                                  feed_dict={rgb_adv_model._inputs: sample[0]})
              acc += (softmax.argmax(axis=1) == sample[1]).sum()
              N += sample[1].shape[0]
      except tf.errors.OutOfRangeError:
          print(acc / N)
          pass




      while True:

          sess.run(trainset_itr_init.initializer)

          try:
              while True:
                  sample = sess.run(trainset_itr)

                  x = sample[0]
                  label = sample[1]

                  label_coeff = -1.0

                  _, gradient, _loss, _summary_merge, _global_step = rgb_adv_model.session.run(
                      fetches=[train_op, grad, rgb_adv_model._loss, summary_merge, global_step],
                      feed_dict={rgb_adv_model._inputs: x,
                                 rgb_adv_model._labels: label,
                                 rgb_adv_model._labels_coeff: label_coeff})

                  pert = rgb_adv_model.session.run(rgb_adv_model._pert)

                  _w_reg_loss, _ce_loss = rgb_adv_model.session.run(
                      fetches=[rgb_adv_model._w_regularizer, rgb_adv_model._ce_loss_mean],
                      feed_dict={rgb_adv_model._inputs: x,
                                 rgb_adv_model._labels: label,
                                 rgb_adv_model._labels_coeff: label_coeff})
                  # print((pert.max()-pert.min())/2.)

                  print("w_reg_loss:{}  ce_loss:{}  minimax:{}".format(_w_reg_loss,_ce_loss,(pert.max()-pert.min())/2. ))


          except tf.errors.OutOfRangeError:
              pass


          sess.run(testset_itr_init.initializer)

          acc = 0
          N = 0
          try:
              while True:
                  sample = sess.run(testset_itr)
                  softmax = rgb_adv_model.session.run(fetches=rgb_adv_model._softmax,
                                                      feed_dict={rgb_adv_model._inputs: sample[0]})
                  acc += (softmax.argmax(axis=1) == sample[1]).sum()
                  N += sample[1].shape[0]
          except tf.errors.OutOfRangeError:
              pass

          print(acc/N)
          # adv = a._model.session.run(a._model._adversarial, feed_dict={a._model._inputs: x[np.newaxis]})
          # adv = x
         # mask  = rgb_adv_model.session.run(rgb_adv_model._mask)


      pert = rgb_adv_model.session.run(rgb_adv_model._pert)
      num_of_miss = 0
      for j in range(_SAMPLE_VIDEO_FRAMES):
          _x=rgb_sample.copy()
          _x[0,j]+=pert.squeeze()
          softmax = rgb_adv_model.session.run(rgb_adv_model._softmax, feed_dict={rgb_adv_model._inputs: _x, adv_flag:0.0})
          if softmax.argmax()!=kinetics_classes.index(correct_cls):
            num_of_miss+=1

      if num_of_miss > 0.6*_SAMPLE_VIDEO_FRAMES:
          a=1
      summary_writer_train.add_summary(summary=_summary_merge,global_step=_global_step)
          # stop_flag = (rgb_adv_model.session.run(rgb_adv_model._softmax, feed_dict={rgb_adv_model._inputs: x}).argmax(
          #     axis=1) != label).all()
          # if stop_flag:
          #     a=1
      _reg_loss = rgb_adv_model.session.run(
          fetches=[rgb_adv_model._regularizer],
          feed_dict={rgb_adv_model._inputs: x,
                    rgb_adv_model._labels: label,
                    rgb_adv_model._labels_coeff: label_coeff})
      print(_reg_loss[0])
      # _, loss = rgb_adv_model.session.run(fetches=[train_op, rgb_adv_model._loss], feed_dict={rgb_adv_model._inputs: x,
      #                                                                                                  rgb_adv_model._labels: [label],
      #                                                                                                 rgb_adv_model._labels_coeff: label_coeff})
      # # pert = a._model.session.run(a._model._pert)


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
