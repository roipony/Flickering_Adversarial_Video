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
import glob

# matplotlib.use('TkAgg')
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



_NUM_OF_CORES = 32
_NUM_EPOCHS = 200
_NUM_CLASSES = 400

_IMAGE_SIZE = 224
_BATCH_SIZE = 8

_SAMPLE_VIDEO_FRAMES =90 #250# 90 #90 #250 #90 #79
_BASE_PATCH_FRAMES = _SAMPLE_VIDEO_FRAMES # 1:for sticker _SAMPLE_VIDEO_FRAMES # 1
_IND_START = 0  # 0 #50
_IND_END =_SAMPLE_VIDEO_FRAMES

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

def get_i3d_model_variable(model, eval_type):
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

    return rgb_variable_map


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


def create_adversarial_model(rgb_input):

    default_adv_flag = tf.constant(1.0, dtype=tf.float32)
    adv_flag = tf.placeholder_with_default(default_adv_flag, shape=default_adv_flag.shape)

    eps_rgb = tf.Variable(tf.zeros(shape=[_BASE_PATCH_FRAMES, 1, 1, 3], dtype=tf.float32), name='eps')

    mask = tf.ones(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])

    indices = np.linspace(_IND_START, _IND_END, _IND_END - _IND_START + 1)
    mask_indecator = tf.one_hot(indices=indices, depth=_SAMPLE_VIDEO_FRAMES)
    mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
    mask_indecator = tf.reshape(mask_indecator, [_SAMPLE_VIDEO_FRAMES, 1, 1, 1])
    mask_rgb = tf.convert_to_tensor(mask * mask_indecator, name='eps_mask')  # same shape as input
    # adversarial_inputs_rgb = tf.nn.tanh(rgb_input + adv_flag * (mask_rgb * eps_rgb),name='adversarial_input')
    random_shift = tf.random_uniform(dtype=tf.int32, minval=0, maxval=_SAMPLE_VIDEO_FRAMES, shape=[])
    cyclic_rgb_input = tf.roll(rgb_input, shift=random_shift, axis=1)

    cyclic_flag_default = tf.constant(0.0, dtype=tf.float32)
    cyclic_flag = tf.placeholder_with_default(cyclic_flag_default, name='cyclic_flag',
                                              shape=cyclic_flag_default.shape)

    model_input = cyclic_flag * cyclic_rgb_input + (1 - cyclic_flag) * rgb_input

    adversarial_inputs_rgb = tf.clip_by_value(model_input + adv_flag * (mask_rgb * eps_rgb),
                                              clip_value_min=-1.0,
                                              clip_value_max=1.0,
                                              name='adversarial_input')

    # return pertubation, adversarial, adversarial ph, cyclic ph
    return eps_rgb, adversarial_inputs_rgb, adv_flag, cyclic_flag




def model_fn(features, labels, mode, params):



    # _labels_coeff = 1.0
    # _cyclic_flag = 0.0
    # _adv_flag = 1.0
    #
    # feed_dict = {inputs: rgb_sample, labels: [target_class], labels_coeff: _labels_coeff, cyclic_flag: _cyclic_flag}


    rgb_sample =features

    eval_type = params['eval_type']
    ckpt_path = params['ckpt_path']
    labels_coeff = params['labels_coeff']

    rgb_model = load_i3d_model(num_classes=_NUM_CLASSES)
    
    rgb_variable_map = get_i3d_model_variable(rgb_model,eval_type)
    
    # tf.train.init_from_checkpoint('/data/DL/Adversarial/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    #                                  rgb_variable_map)
    
    
    
    # checkpoint_state = tf.train.get_checkpoint_state('pretrained')
    # input_checkpoint = '/data/DL/Adversarial/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
    # pretrain_saver = tf.train.Saver(rgb_variable_map,reshape=True)
    
    # def init_fn(scaffold, session):
    #     pretrain_saver.restore(session, input_checkpoint)
    
    # scaffold = tf.train.Scaffold(init_fn=init_fn)
    
    eps_rgb, adversarial_inputs_rgb, adv_flag, cyclic_flag = create_adversarial_model(rgb_sample)

    
    model_logits, _ = rgb_model(adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)
    softmax = tf.nn.softmax(logits=model_logits)
    
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model_logits)
    ce_loss_mean = tf.reduce_mean(labels_coeff * ce_loss)
    # total loss:
    loss = ce_loss_mean

    # loss = tf.Print(loss, [loss], message='Loss Value: ', name='P')
    perturbation = eps_rgb


    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer()
    gradients = optimizer.compute_gradients(loss=loss, var_list=perturbation)
    train_op = optimizer.apply_gradients(gradients,global_step=global_step)
    
    logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode, loss=loss,train_op=train_op,training_hooks =[logging_hook])

    
    # tf.train.init_from_checkpoint('/data/DL/Adversarial/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    #                                 rgb_variable_map)
    
    sess= tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    # init_model(rgb_model,sess,'/data/DL/Adversarial/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt')

    eps_rgb, adversarial_inputs_rgb, adv_flag, cyclic_flag = create_adversarial_model(rgb_sample)


    #var_to_restore = get_i3d_model_variable(rgb_model,eval_type=eval_type)
    # tf.train.init_from_checkpoint('/data/DL/Adversarial/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    #                               var_to_restore)
    # init_model(model=rgb_model, sess=sess, ckpt_path=ckpt_path, eval_type=eval_type)
    # sess.run(eps_rgb.initializer)
    model_logits, _ = rgb_model(adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)
    softmax = tf.nn.softmax(logits=model_logits)






    # inputs = rgb_input
    perturbation = eps_rgb

    # target label: for untargeted attack set original label
    # labels = tf.placeholder(tf.int64, (None,), name='labels')

    # for untargeted attack set to -1.0
    # label_coeff_default = tf.constant(-1.0, dtype=tf.float32)
    # labels_coeff = tf.placeholder_with_default(label_coeff_default, name='label_coeff', shape=label_coeff_default.shape)

    # adversarial classification loss
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model_logits)
    ce_loss_mean = tf.reduce_mean(labels_coeff * ce_loss)

    # regularization loss:
    beta_0 = 1.0  # 1.0
    beta_1 = 0.1  # 0.1 #0.1 #0.1 #100000 #0.01  # 100000 #0.0001 # 0.001 #0.01 #0.1 #0.0001 #1
    beta_2 = 1.0
    norm_reg = tf.sqrt(tf.reduce_sum(perturbation ** 2) + 1e-12)
    diff_norm_reg = tf.sqrt(tf.reduce_sum((perturbation - tf.roll(perturbation, 1, axis=0)) ** 2) + 1e-12)
    regularizer_loss = beta_1 * norm_reg + beta_2 * diff_norm_reg
    # regularizer = tf.reduce_max(perturbation) - tf.reduce_min(perturbation)
    # with tf.device('/gpu:1'):
    # regularizer_loss = cx_loss(tf.expand_dims(inputs[0,_IND_START:_IND_END+1,...],axis=0),tf.expand_dims(adversarial_inputs_rgb[0,_IND_START:_IND_END+1,...],axis=0))
    weighted_regularizer_loss = beta_0 * regularizer_loss

    # total loss:
    loss = ce_loss_mean + weighted_regularizer_loss

    # loss = tf.Print(loss, [loss], message='Loss Value: ', name='P')


    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer()
    gradients = optimizer.compute_gradients(loss=loss, var_list=perturbation)
    train_op = optimizer.apply_gradients(gradients,global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss,train_op=train_op,scaffold=scaffold)

    # sess.run(tf.variables_initializer(optimizer.variables()))

    # writer = tf.summary.FileWriter(logdir, sess.graph)
    # with tf.name_scope('input'):
    #     clean_summary = tf.summary.image('clean_image', tf.expand_dims(inputs[0, _IND_START], axis=0), max_outputs=inputs.shape[0].value)
    #     pert_summary = tf.summary.image('perturbation', perturbation, max_outputs=perturbation.shape[0].value)
    #     adv_summary = tf.summary.image('adversarial_image', tf.expand_dims(adversarial_inputs_rgb[0,_IND_START],axis=0), max_outputs=adversarial_inputs_rgb.shape[0].value)

    # adv_vec = np.zeros(shape=[1,_SAMPLE_VIDEO_FRAMES])
    # merged = tf.summary.merge_all()
    #
    # feed_dict = {inputs: rgb_sample, labels: [target_class], labels_coeff: -1.0}
    # model_softmax = sess.run(softmax, feed_dict={inputs: rgb_sample})
    #
    # base_prob =model_softmax[0,target_class]
    # criteria = fb_2.criteria.Misclassification()
    # criteria = fb_2.criteria.ConfidentMisclassification(p=0.95)
    # criteria = fb_2.criteria.TargetClass(target_class=kinetics_classes.index('javelin throw'))



def generate_input_fn(file_names, mode=tf.estimator.ModeKeys.EVAL, batch_size=1):
    def _input_fn():
        
        
        
        
        tf_records_list_path_train = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/val/'
        tf_record_list_train = sorted(glob.glob(tf_records_list_path_train+ '*.tfrecords'))[0:10]
        
        # tf_record_list_train = [x.strip() for x in open(tf_records_list_path_train)]
        dataset_train = tf.data.TFRecordDataset(filenames=tf_record_list_train, num_parallel_reads=os.cpu_count())
        dataset_train = dataset_train.prefetch(buffer_size=_BATCH_SIZE*5)
        dataset_train = dataset_train.batch(_BATCH_SIZE,drop_remainder=True)
        dataset_train = dataset_train.map(img_tool.parse_example_uint8, num_parallel_calls=os.cpu_count())
        # dataset_train = dataset_train.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
            # dataset = dataset.shuffle(buffer_size=buffer_size)
        
        # Transformation
        # dataset_train = dataset_train.apply(tf.contrib.data.map_and_batch(img_tool.parse_example, batch_size=_BATCH_SIZE,drop_remainder=True,num_parallel_calls=20))
        dataset_train =dataset_train.prefetch(1)
        
        return dataset_train
        
        tf_record_list = [x.strip() for x in open(file_names)]
        dataset = tf.data.TFRecordDataset(filenames=tf_record_list, num_parallel_reads=10)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            buffer_size = batch_size * 2 + 1
            # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
            # dataset = dataset.shuffle(buffer_size=buffer_size)

        # Transformation
        dataset = dataset.apply(tf.contrib.data.map_and_batch(img_tool.parse_example, batch_size=batch_size,num_parallel_calls=1,drop_remainder=True))

        # dataset = dataset.map(parse_record)
        # dataset = dataset.map(
        #     lambda image, label: (preprocess_image(image, is_training), label))
        #
        # dataset = dataset.repeat()
        # dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=1)

        # images, labels = dataset.make_one_shot_iterator().get_next()
        #
        # features = {'images': images}
        return dataset

    return _input_fn



#%%

logdir ='logs/'

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

destribution = tf.distribute.MirroredStrategy() #devices=["/gpu:0", "/gpu:1"]
run_config = tf.estimator.RunConfig(train_distribute=destribution,eval_distribute=destribution)
#%%
params = {
    'eval_type': 'rgb',
    'labels_coeff': 1,
    'ckpt_path': '/tmp.ckpt'
}
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir='model',  # Will save the weights here automatically
    params=params ,config=run_config)
train_spec = tf.estimator.TrainSpec(
    input_fn =generate_input_fn('/data/DL/Adversarial/kinetics-i3d/data/tfrecords/triple_jump/val_tfrecords.txt',mode=tf.estimator.ModeKeys.TRAIN),
    max_steps=6000,
    hooks=None
)
eval_spec = tf.estimator.EvalSpec(
    input_fn =generate_input_fn('/data/DL/Adversarial/kinetics-i3d/data/tfrecords/triple_jump/val_tfrecords.txt',mode=tf.estimator.ModeKeys.EVAL),
    hooks=None
)

tf.estimator.train_and_evaluate(estimator=  estimator,train_spec=train_spec,eval_spec=eval_spec)
# estimator.train(generate_input_fn('/data/DL/Adversarial/kinetics-i3d/data/tfrecords/triple_jump/val_tfrecords.txt',mode=tf.estimator.ModeKeys.TRAIN))
sess = tf.Session()
with tf.variable_scope(scope):

    # rgb_sample = np.load('/data/DL/Adversarial/kinetics-i3d/data/san_diego_ca_jadel_gregorio_triple_jump_1532m_short_approach.npy')[0, -_SAMPLE_VIDEO_FRAMES:]
    # rgb_sample = np.load('/data/DL/Adversarial/kinetics-i3d/data/triple_jump_kinetics.npy')[ -_SAMPLE_VIDEO_FRAMES:]
    rgb_sample = np.load('data/triple_jump_1_kinetics.npy')[0,-_SAMPLE_VIDEO_FRAMES:]
    rgb_sample = rgb_sample[np.newaxis]
    correct_cls = 'triple jump'
    target_class = kinetics_classes.index(correct_cls)
    # rgb_sample, flow_sample, correct_cls = get_video_sample(ucf_video_list)

    # RGB input has 3 channels.
    rgb_input = tf.placeholder(tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

eps_rgb, adversarial_inputs_rgb, adv_flag, cyclic_flag = create_adversarial_model(rgb_sample)

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
beta_0 = 1.0 #1.0
beta_1 = 0.1 #0.1 #0.1 #0.1 #100000 #0.01  # 100000 #0.0001 # 0.001 #0.01 #0.1 #0.0001 #1
beta_2 =1.0
norm_reg = tf.sqrt(tf.reduce_sum(perturbation ** 2) + 1e-12)
diff_norm_reg =  tf.sqrt(tf.reduce_sum( (perturbation - tf.roll(perturbation,1,axis=0))**2) + 1e-12)
regularizer_loss = beta_1*norm_reg + beta_2*diff_norm_reg
# regularizer = tf.reduce_max(perturbation) - tf.reduce_min(perturbation)
# with tf.device('/gpu:1'):
# regularizer_loss = cx_loss(tf.expand_dims(inputs[0,_IND_START:_IND_END+1,...],axis=0),tf.expand_dims(adversarial_inputs_rgb[0,_IND_START:_IND_END+1,...],axis=0))
weighted_regularizer_loss = beta_0 * regularizer_loss

# total loss:
loss = ce_loss_mean + weighted_regularizer_loss

optimizer = tf.train.AdamOptimizer()
gradients = optimizer.compute_gradients(loss=loss, var_list=perturbation)
train_op = optimizer.apply_gradients(gradients)
sess.run(tf.variables_initializer(optimizer.variables()))

writer = tf.summary.FileWriter(logdir, sess.graph)
# with tf.name_scope('input'):
#     clean_summary = tf.summary.image('clean_image', tf.expand_dims(inputs[0, _IND_START], axis=0), max_outputs=inputs.shape[0].value)
#     pert_summary = tf.summary.image('perturbation', perturbation, max_outputs=perturbation.shape[0].value)
#     adv_summary = tf.summary.image('adversarial_image', tf.expand_dims(adversarial_inputs_rgb[0,_IND_START],axis=0), max_outputs=adversarial_inputs_rgb.shape[0].value)

# adv_vec = np.zeros(shape=[1,_SAMPLE_VIDEO_FRAMES])
# merged = tf.summary.merge_all()
#
# feed_dict = {inputs: rgb_sample, labels: [target_class], labels_coeff: -1.0}
# model_softmax = sess.run(softmax, feed_dict={inputs: rgb_sample})
#
# base_prob =model_softmax[0,target_class]
# criteria = fb_2.criteria.Misclassification()
# criteria = fb_2.criteria.ConfidentMisclassification(p=0.95)
criteria = fb_2.criteria.TargetClass(target_class=kinetics_classes.index('javelin throw'))

_labels_coeff = 1.0
_cyclic_flag = 0.0
_adv_flag =1.0

feed_dict = {inputs: rgb_sample, labels: [target_class], labels_coeff: _labels_coeff, cyclic_flag: _cyclic_flag}

train_tf_record_path = 'data/tfrecords/dbg_train_tfrecords.txt'
test_tf_record_path = 'data/tfrecords/dbg_train_tfrecords.txt'

# valid_tf_record_path = '/media/ROIPO/Data/projects/Adversarial/kinetics-i3d/data/tfrecords/valid_tfrecords.txt'

train_tf_record_list = [x.strip() for x in open(train_tf_record_path)]
test_tf_record_list = [x.strip() for x in open(test_tf_record_path)]

trainset = tf.data.TFRecordDataset(filenames=train_tf_record_list,
                                   num_parallel_reads=train_tf_record_list.__len__())
trainset = trainset.map(map_func=img_tool.parse_example, num_parallel_calls=_BATCH_SIZE)
trainset = trainset.batch(batch_size=_BATCH_SIZE, drop_remainder=False).prefetch(_BATCH_SIZE)

testset = tf.data.TFRecordDataset(filenames=test_tf_record_list, num_parallel_reads=test_tf_record_list.__len__())
testset = testset.map(map_func=img_tool.parse_example, num_parallel_calls=_BATCH_SIZE)
testset = testset.batch(batch_size=_BATCH_SIZE, drop_remainder=False).prefetch(_BATCH_SIZE)

global_step = tf.train.get_or_create_global_step()

trainset_itr_init = trainset.make_initializable_iterator()
trainset_itr = trainset_itr_init.get_next()

testset_itr_init = testset.make_initializable_iterator()
testset_itr = testset_itr_init.get_next()

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

grad = optimizer.compute_gradients(loss=rgb_adv_model._loss, var_list=rgb_adv_model._pert)
#     # self._optimizer = tf.train.AdamOptimizer()
#     # train_op = self._optimizer.apply_gradients(zip([grad[0][0]], [a._model._pert]))
train_op = optimizer.apply_gradients(grads_and_vars=grad, global_step=global_step)



tf.summary.scalar('train/total_loss', rgb_adv_model._loss)

pertubation = rgb_adv_model._pert - tf.reduce_min(rgb_adv_model._pert)
pertubation = pertubation / tf.reduce_max(pertubation)
pertubation = pertubation * 255.

tf.summary.image('train/pertubation', pertubation)
logdir = '/media/ROIPO/Data/projects/Adversarial/kinetics-i3d/log/train/'

summary_writer_train = tf.summary.FileWriter(logdir + "/0005", graph=sess.graph)
summary_merge = tf.summary.merge_all()

mini_batch = 10
# target_class = kinetics_classes.index('javelin throw')
target_class = kinetics_classes.index(correct_cls)

acc = 0
N = 0
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

            print("w_reg_loss:{}  ce_loss:{}  minimax:{}".format(_w_reg_loss, _ce_loss,
                                                                 (pert.max() - pert.min()) / 2.))


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

    print(acc / N)
    # adv = a._model.session.run(a._model._adversarial, feed_dict={a._model._inputs: x[np.newaxis]})
    # adv = x
# mask  = rgb_adv_model.session.run(rgb_adv_model._mask)

pert = rgb_adv_model.session.run(rgb_adv_model._pert)
num_of_miss = 0
for j in range(_SAMPLE_VIDEO_FRAMES):
    _x = rgb_sample.copy()
    _x[0, j] += pert.squeeze()
    softmax = rgb_adv_model.session.run(rgb_adv_model._softmax,
                                        feed_dict={rgb_adv_model._inputs: _x, adv_flag: 0.0})
    if softmax.argmax() != kinetics_classes.index(correct_cls):
        num_of_miss += 1

if num_of_miss > 0.6 * _SAMPLE_VIDEO_FRAMES:
    a = 1
summary_writer_train.add_summary(summary=_summary_merge, global_step=_global_step)
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



for step in range(10000000000):  # to repeat with decreased epsilons if necessary

    _, total_loss, adv_loss, reg_loss = sess.run(fetches=[train_op, loss, ce_loss_mean, regularizer_loss],
                                                 feed_dict=feed_dict)
    print("Total Loss: {} , Cls Loss: {} , Reg Loss: {} ".format(total_loss, adv_loss, reg_loss))

    # _, total_loss, adv_loss = sess.run(fetches=[train_op, loss, ce_loss_mean],
    #                                              feed_dict=feed_dict)
    # print("Total Loss: {} , Cls Loss: {}  ".format(total_loss, adv_loss))
    pert = sess.run(perturbation)
    model_softmax = sess.run(softmax, feed_dict={inputs: rgb_sample})
    _model_logits =  sess.run(model_logits, feed_dict={inputs: rgb_sample})
    # is_adversarial = criteria.is_adversarial(model_softmax, target_class)

    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    # summary = sess.run(merged, feed_dict={inputs: rgb_sample},
    #                    options=run_options,
    #                    run_metadata=run_metadata)
    # writer.add_summary(summary, step)

    is_adversarial = criteria.is_adversarial(_model_logits.squeeze(), target_class)

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




#%%
