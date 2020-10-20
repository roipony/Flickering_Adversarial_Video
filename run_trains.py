# -*- coding: utf-8 -*-

import subprocess
import utils.pre_process_rgb_flow as img_tool

import utils.kinetics_i3d_utils as ki3du
import yaml
import os
import shutil

all_cls_shuffle_t300_v1000database_path='/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/train/'
train_tfrecord_list_path='/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/train/'
val_tfrecord_path='/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/val/'
test_tfrecord_path='/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_uint8/test/'

train_tfrecord_cls_list = [ff for ff in os.listdir(train_tfrecord_list_path) if os.path.isdir(os.path.join(train_tfrecord_list_path,ff))]


train_data_path = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/train/'
allcls = [ff for ff in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path,ff))]


results_path = '/data/DL/Adversarial/kinetics-i3d/result/generalization/single_class/'

vnev_path= '/data/DL/Adversarial/kinetics-i3d/venv3/bin/python'
yml_path ='many_runs_config.yml'

#%%
for cls in allcls:
    
    print(cls)
    allclsfinish = [ff for ff in os.listdir(results_path) if os.path.isdir(os.path.join(results_path,ff))]

    if cls in allclsfinish:
        print('class {} allready finish, skip {}'.format(cls,cls))
        continue
    #create tfrecord
    train_tfrecord_path = os.path.join(train_tfrecord_list_path,'{}/'.format(cls))

    if not os.path.exists(train_tfrecord_path) :
        print('Creating tfrecord for {} ...'.format(cls))
        cmd = [vnev_path, 'kinetics_to_tf_record_uint8.py', database_path,cls, train_tfrecord_list_path]
        subprocess.Popen(cmd).wait()
        print('Finish create tfrecord for {}'.format(cls))
    
    kinetics_classes = ki3du.load_kinetics_classes()
    
    with open(yml_path,'r') as f:
        cfg = yaml.load(f)
        
    cfg["CLASS_GEN_ATTACK"]["TF_RECORDS_TRAIN_PATH"] = [train_tfrecord_path]
    cfg["CLASS_GEN_ATTACK"]["TF_RECORDS_VAL_PATH"] = [os.path.join(val_tfrecord_path,'{}/'.format(cls)),os.path.join(val_tfrecord_path,'{}/'.format(cls))]
    

    with open(yml_path, "w") as f:
        yaml.dump(cfg, f)
    
    
    cmd = [vnev_path, 'adversarial_main_gain_batch.py', yml_path]
    print('Training...'.format(cls))
    subprocess.Popen(cmd).wait()
    print('Finish training!!'.format(cls))
    
    if os.path.exists(train_tfrecord_path):
        print('Deleting {} tfrecord'.format(cls))
        shutil.rmtree(train_tfrecord_path)
