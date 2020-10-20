import cv2
import numpy as np
import imageio
import tensorflow as tf
import i3d

import random
import skvideo.io
import sys
import os

import setGPU
_IMAGE_SIZE = 224
_BATCH_SIZE = 1

_SAMPLE_VIDEO_FRAMES =250 #79 # 79 #90 #90 #250 #90 #79
_BASE_PATCH_FRAMES = _SAMPLE_VIDEO_FRAMES #_SAMPLE_VIDEO_FRAMES #_SAMPLE_VIDEO_FRAMES # 1# _SAMPLE_VIDEO_FRAMES # 1:for sticker _SAMPLE_VIDEO_FRAMES # 1
_IND_START = 0  # 0 #50
_IND_END =_SAMPLE_VIDEO_FRAMES

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



def image_resize(image, width = None, height = None, inter = cv2.INTER_LINEAR):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image

    r = max(float(width) / w, float(height) / h)
    dim = (int(w * r), int(h * r))

    # if width is None and height is None:
    #     return image
    #
    # # check to see if the width is None
    # if width is None:
    #     # calculate the ratio of the height and construct the
    #     # dimensions
    #     r = height / float(h)
    #     dim = (int(w * r), height)
    #
    # # otherwise, the height is None
    # else:
    #     # calculate the ratio of the width and construct the
    #     # dimensions
    #     r = width / float(w)
    #     dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def crop_center_image(image, rec_len):
    # crop center
    (h, w) = image.shape[:2]
    x_1 = (w - rec_len) // 2
    x_2 = (h - rec_len) // 2
    cropped = image[x_2:x_2+rec_len, x_1:x_1+rec_len]
    return cropped


def video_to_image_and_of(video_path, target_fps=25, resize_height=256, crop_size=224, n_steps=100,plotting=False,flow=False):
# preprocessing:

    # video_path = '/media/ROIPO/Data/projects/Adversarial/database/video/v_LongJump_g03_c06.avi'
    #video_path = '/home/ADAMGE/Downloads/test_jog.mp4'
    # target_fps = 25.0
    # n_steps = 100
    # resize_height = 256
    # crop_size = 224


    clip_frames = []
    clip_frames_flow = []
    bit = 0
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)

    frame_gap = int(round(fps / target_fps))
    if frame_gap == 0:
        raise ValueError
    frame_num = 0

    ret, frame1 = capture.read()
    resized_frame1 = image_resize(frame1, height = resize_height,width=resize_height)
    prvs = cv2.cvtColor(resized_frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    # extract features
    while (capture.isOpened()) & (bit == 0):
        flag, frame = capture.read()
        if flag == 0:
            bit = 1
            print("******ERROR: Could not read frame in " + video_path + " frame_num: " + str(frame_num))
            break

        #name = params['res_vids_path'] + str(frame_num) + 'frame.jpg'
        #cv2.imwrite(name, frame)
        #cv2.imshow("Vid", frame)
        #key_pressed = cv2.waitKey(10)  # Escape to exit

        # process frame (according to the correct frame rate)
        if frame_num % frame_gap == 0:
            # RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = image_resize(image,width=resize_height, height = resize_height)
            res = resized_frame.astype('float32') / 128. - 1
            res = crop_center_image(res, crop_size)
            clip_frames.append(res)

            if plotting:
                res_to_plot = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
                res_to_plot = res_to_plot + 1.0 / 2.0
                cv2.imshow("Vid", res_to_plot)
                key_pressed = cv2.waitKey(10)  # Escape to exit

            # FLOW
            if flow:
                image_flow = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
                # flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                optical_flow = cv2.DualTVL1OpticalFlow_create()
                flow = optical_flow.calc(prvs, image_flow, None)
                flow[flow > 20] = 20
                flow[flow < -20] = -20
                flow = flow / 20.
                flow = crop_center_image(flow, crop_size)
                clip_frames_flow.append(flow)

                # potting:
                if plotting:
                    flow_temp = (flow + 1.0) / 2.0
                    last_channel = np.zeros((crop_size,crop_size), dtype=float) + 0.5
                    flow_to_plot = np.dstack((flow_temp, last_channel))
                    cv2.imshow("Vid-flow", flow_to_plot)
                    key_pressed = cv2.waitKey(10)  # Escape to exit


            #mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #hsv[..., 0] = ang * 180 / np.pi / 2
            #hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #cv2.imshow('frame2', bgr)
            #k = cv2.waitKey(30) & 0xff

                prvs = image_flow
        frame_num += 1



    capture.release()


     # (int(round(frame_num / frame_gap)) < n_steps)
    if frame_num>=n_steps:
        frames = np.array(clip_frames)[-n_steps:]
        frames_flow = np.array(clip_frames_flow)
        frames = np.expand_dims(frames, axis=0)
        frames_flow = np.expand_dims(frames_flow, axis=0)
    else:

        frames = np.array(clip_frames)
        frames = np.pad(frames, ((0, n_steps-frames.shape[0]),(0,0),(0,0),(0,0)),'wrap')
        frames_flow = np.array(clip_frames_flow)
        frames = np.expand_dims(frames, axis=0)
        frames_flow = np.expand_dims(frames_flow, axis=0)



    return frames, frames_flow  


def frames_to_gif(frames, output_path, fps=25):
    imageio.mimsave(output_path, frames, fps=fps)

# np.save('rgb.npy', frames)
# np.save('flow.npy', frames_flow)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def parse_example(serialized):
    context_features = {'train/label' : tf.FixedLenFeature((), tf.int64),
                        'train/video' : tf.VarLenFeature(dtype=tf.float32)}
    # context_features = {'train/label' : tf.FixedLenFeature((), tf.int64),
    #                     'train/video' : tf.FixedLenFeature((), tf.string)}

    context_parsed,_ = tf.parse_single_sequence_example(serialized=serialized,context_features =context_features,sequence_features={})

    # video =  tf.image.decode_jpeg(context_parsed['train/video'], channels=3)
    video =  tf.reshape(tf.sparse.to_dense(context_parsed['train/video']), shape=[-1,224,224,3])
    label = context_parsed['train/label']

    return video, label

from os import listdir
tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99

scope ='RGB'

# sess = tf.Session(config  = tf_config)
# with tf.variable_scope(scope):
#     rgb_input = tf.placeholder(tf.float32,
#     shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
# rgb_model = load_i3d_model(num_classes=400)
# init_model(model=rgb_model,sess=sess, ckpt_path='data/checkpoints/rgb_imagenet/model.ckpt',eval_type='rgb')

# model_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
# softmax = tf.nn.softmax(logits = model_logits)

# f, of, = video_to_image_and_of('/home/ROIPO/Downloads/y2mate.com - utsa_track_devon_bond_triple_jump_67R1t4-b1nw_360p.mp4',n_steps=90)

video_list_path = [x.strip() for x in open('/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/val_path_list.txt')]#'/media/ROIPO/Data/projects/Adversarial/kinetics-downloader/dataset/train/triple_jump/'
# video_list_path = listdir(video_base_path)
label_map_path='/data/DL/Adversarial/kinetics-i3d/data/label_map.txt'
random.shuffle(video_list_path)
# class_target = ['juggling balls']
# class_target = ['juggling balls']
n_frames = _SAMPLE_VIDEO_FRAMES

kinetics_classes = [x.strip() for x in open(label_map_path)]

#
# video_list = [x.strip() for x in open(vl)]
k=0

tf_dst_folder = '/data/DL/Adversarial/ActivityNet/Crawler/Kinetics/database/tfrecord_250/val/'

for i, v in enumerate(video_list_path):
    if i%10 ==0:
        if i>0:
            writer.close()
            k+=1
        train_filename = tf_dst_folder+'kinetics_{}_{:04}.tfrecords'.format( v.split('/')[-3],k)  # address to save the TFRecords file
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(train_filename)

    cls =  v.split('/')[-2]
    cls_id = kinetics_classes.index(cls)
    vid_path = v
    
    
    if os.path.exists(v)==False:
        continue
    try:
        frames = skvideo.io.vread(vid_path)
        frames = frames.astype('float32') / 128. - 1
    except:
        os.remove(vid_path)
        continue
        
    if frames.shape[0] < _SAMPLE_VIDEO_FRAMES:
        continue
        #frames = np.pad(frames, ((0, _SAMPLE_VIDEO_FRAMES-frames.shape[0]),(0,0),(0,0),(0,0)),'wrap')
    else:
        frames=frames[-_SAMPLE_VIDEO_FRAMES:]
        
    # prob = sess.run(softmax,feed_dict={rgb_input:frames})
    # top_id = prob.argmax()
    # if cls_id!=top_id:
    #     continue
    feature = {'train/label': _int64_feature(cls_id),
               'train/video': _float_list_feature(frames.reshape([-1]))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()


        # video_path= ['/media/ROIPO/Data/projects/Adversarial/database/UCF-101/video/v_BreastStroke_g01_c01.avi',
        # '/media/ROIPO/Data/projects/Adversarial/database/UCF-101/video/v_BreastStroke_g01_c02.avi']
        #



    # for i, vp in enumerate(video_path):
    #     frames, flow_frames = video_to_image_and_of(video_path=vp,n_steps=80)
    #     feature = {'train/label': _int64_feature(1),
    #                'train/video': _float_list_feature(frames.reshape([-1]))}
    #     example = tf.train.Example(features=tf.train.Features(feature=feature))
    #     writer.write(example.SerializeToString())
    

    # sys.stdout.flush()
    
    #     frames_list.append(frames)
    # np.save('data/db.npy', frames_list)
