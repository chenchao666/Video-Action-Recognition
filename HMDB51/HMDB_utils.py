import os
import tensorflow as tf
import skvideo
skvideo.setFFmpegPath("/home/ercong.cc/software/ffmpeg/bin")
import skvideo.io
import numpy as np
import cv2
import time
import random

class DataLoader():
    def __init__(self,dataset):
        self.dataset=dataset
        if self.dataset=='hmdb51':
            self.train_file = 'train_list1.txt'
            self.test_file = 'test_list1.txt'

    def get_TrainData(self):
        with open(self.train_file) as file:
            train_list = [row.strip() for row in list(file)]
            file_path_list = [row.split(' ')[0] for row in train_list]
            file_label_list = [row.split(' ')[1] for row in train_list]
        return file_path_list, file_label_list

    def get_TestData(self):
        with open(self.test_file) as file:
            train_list = [row.strip() for row in list(file)]
            file_path_list = [row.split(' ')[0] for row in train_list]
            file_label_list = [row.split(' ')[1] for row in train_list]
        return file_path_list, file_label_list



def avi_2_npy(path,sampled_frame,mode,crop_flag):
    if mode=='avi':
        videodata = skvideo.io.vread(path,as_grey=False)
        videodata=videodata/128.0-1
    elif mode=='npy':
        split=path.split('.')
        file=split[0]+'.npy'
        videodata=np.load(file)
        videodata = videodata/128.0-1

    # if videodata.shape[1]>=videodata.shape[2]:
    #     videodata_0=np.zeros((videodata.shape[0],int(videodata.shape[1]*256.0/videodata.shape[2]),256,3))
    #     for i in range(videodata.shape[0]):
    #         videodata_0[i,:,:,:]=cv2.resize(videodata[i,:,:,:],(256,int(videodata.shape[1]*256.0/videodata.shape[2])),interpolation=cv2.INTER_CUBIC)
    # if videodata.shape[1]<videodata.shape[2]:
    #     videodata_0=np.zeros((videodata.shape[0],256,int(videodata.shape[2] * 256.0/videodata.shape[1]),3))
    #     for i in range(videodata.shape[0]):
    #         videodata_0[i,:,:,:]=cv2.resize(videodata[i,:,:,:],(int(videodata.shape[2] * 256.0/videodata.shape[1]),256),interpolation=cv2.INTER_CUBIC)

    # video_frame=videodata.shape[0]
    # print videodata.shape[0]
    while videodata.shape[0]<sampled_frame+1:
        videodata=np.concatenate((videodata,videodata),axis=0)
        # print videodata.shape[0]
    ind=np.random.randint(videodata.shape[0]-sampled_frame)
    sampled_video=videodata[ind:ind+sampled_frame,:,:,:]
    croped_video=crop(sampled_video,crop_flag)
    if croped_video.shape!=(sampled_frame,224,224,3):
        print croped_video.shape
        print path
    return croped_video


def crop(video,crop_flag):
    if crop_flag=='random_crop':
        ind_x=np.random.randint(video.shape[1]-224)
        ind_y=np.random.randint(video.shape[2]-224)
    if crop_flag=='center_crop':
        ind_x=int(np.floor((video.shape[1]-224)/2))
        ind_y=int(np.floor((video.shape[2]-224)/2))
    croped_video=video[:,ind_x:ind_x+224,ind_y:ind_y+224,:]
    return croped_video



#####################################################################################################
# ----------------------------------  Add my Loss Function  -----------------------------------------
#####################################################################################################


def get_center_loss(features, labels, alpha, num_classes):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])

    ##############################################################
    # centers0=tf.unsorted_segment_mean(features,labels,num_classes)
    # EdgeWeights=tf.ones((num_classes,num_classes))-tf.eye(num_classes)
    # margin=tf.constant(1000,dtype="float32")
    # norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    # center_pairwise_dist = tf.transpose(norm(tf.expand_dims(centers0, 2) - tf.transpose(centers0)))
    # loss_0= tf.reduce_sum(tf.multiply(tf.maximum(0.0, margin-center_pairwise_dist),EdgeWeights))
   ################################################################

    centers_batch = tf.gather(centers, labels)
    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    loss= tf.nn.l2_loss(features - centers_batch)
    centers_update_op= tf.scatter_sub(centers, labels, diff)
    return loss, centers_update_op


def getClassInvariantLoss(features, labels, num_classes):
    Normalized_features=features/(tf.norm(features,axis=1))
