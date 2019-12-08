import os
import tensorflow as tf
import skvideo
skvideo.setFFmpegPath("/home/ercong.cc/software/ffmpeg/bin")
import skvideo.io
import numpy as np
import json
import io
from skimage.transform import resize
import skimage.io as skio
# import cv2

class DataLoader():
    def __init__(self,dataset):
        self.dataset=dataset
        if self.dataset=='something-something-v2':
            self.train_file = 'trainlist.txt'
            self.test_file = 'testlist.txt'


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
    if mode=='webm':
        videodata = skvideo.io.vread(path,as_grey=False)
        if videodata.shape[0] > 32:
            ind = np.linspace(0, videodata.shape[0], 32, endpoint=False)
            ind = ind.astype('int')
            videodata = videodata[ind, :, :, :]
    if mode=='avi':
        videodata = skvideo.io.vread(path,as_grey=False)
        if videodata.shape[0] > 32:
            ind = np.linspace(0, videodata.shape[0], 32, endpoint=False)
            ind = ind.astype('int')
            videodata = videodata[ind, :, :, :]
    elif mode=='npy':
        split=path.split('.')
        file=split[0]+'.npy'
        videodata=np.load(file)
    while videodata.shape[0]<sampled_frame+1:
        videodata=np.concatenate((videodata,videodata),axis=0)
    ind=np.random.randint(videodata.shape[0]-sampled_frame)
    sampled_video=videodata[ind:ind+sampled_frame,:,:,:]
    croped_video=crop(sampled_video,crop_flag)
    # if croped_video.shape!=(sampled_frame,224,224,3):
    #     print croped_video.shape
    #     print path
    return croped_video/128.0-1


def crop(video,crop_flag):
    if crop_flag=='random_crop':
        ind_x=np.random.randint(video.shape[1]-224)
        ind_y=np.random.randint(video.shape[2]-224)
    if crop_flag=='center_crop':
        ind_x=int(np.floor((video.shape[1]-224)/2))
        ind_y=int(np.floor((video.shape[2]-224)/2))
    croped_video=video[:,ind_x:ind_x+224,ind_y:ind_y+224,:]
    return croped_video



###########################################################################################
##     Load video data with multi-processing
import threading
def read_paths(paths,sampled_frame,crop_flag,max_thread_num=32):
    threads = []
    videos = []
    for p in paths:
        while len(threads) >= max_thread_num:
            idx = 0
            while idx < len(threads):
                t = threads[idx]
                if not t.isAlive():
                    videos.append(t.result)
                    del threads[idx]
                else:
                    idx = idx+1
        t = ReadPath(p,sampled_frame,crop_flag)
        t.start()
        threads.append(t)
    # print(threads)
    for t in threads:
        t.join()
        videos.append(t.result)
    return videos
class ReadPath(threading.Thread):
    def __init__(self,path,sampled_frame,crop_flag):
        super(ReadPath,self).__init__()
        self.path = path
        self.sampled_frame=sampled_frame
        self.crop_flag=crop_flag
        self.result = None
    def run(self):
        self.result = ReadVideo(self.path,self.sampled_frame,self.crop_flag)


def ReadVideo(path,sampled_frame,crop_flag):
    videodata=np.load(path)
    # videodata = skvideo.io.vread(path, as_grey=False)
    if videodata.shape[0] > 20:
        ind = np.linspace(0, videodata.shape[0], 20, endpoint=False)
        ind = ind.astype('int')
        videodata = videodata[ind, :, :, :]
    while videodata.shape[0] < sampled_frame + 1:
        videodata = np.concatenate((videodata, videodata), axis=0)
    ind = np.random.randint(videodata.shape[0] - sampled_frame)
    sampled_video = videodata[ind:ind + sampled_frame, :, :, :]
    croped_video = crop(sampled_video, crop_flag)
    croped_video=croped_video/128.0-1
    return croped_video

##########################################################################################

from multiprocessing import Pool
class Multiprocessing_LoadVideo():
    def __init__(self,batch_path,sampled_frame,crop_flag):
        self.path_list=batch_path
        self.sampled_frame=sampled_frame
        self.crop_flag=crop_flag
        self.pool = Pool(12)
        self.videodata=[]

    def LoadVideo(self):
        Video=[]
        for ind in list(range(len(self.path_list))):
            path=self.path_list[ind]
            result = self.pool.apply_async(read_video, args=(path,self.sampled_frame,self.crop_flag))
            self.videodata.append(result)
        self.pool.close()
        self.pool.join()
        for videodata in self.videodata:
            Video.append(videodata.get())
        return Video

def read_video(path,sampled_frame,crop_flag):
    videodata = skvideo.io.vread(path, as_grey=False)
    if videodata.shape[0] > 32:
        ind = np.linspace(0, videodata.shape[0], 32, endpoint=False)
        ind = ind.astype('int')
        videodata = videodata[ind, :, :, :]
    while videodata.shape[0] < sampled_frame + 1:
        videodata = np.concatenate((videodata, videodata), axis=0)
    ind = np.random.randint(videodata.shape[0] - sampled_frame)
    sampled_video = videodata[ind:ind + sampled_frame, :, :, :]
    croped_video = crop(sampled_video, crop_flag)
    croped_video=croped_video/128.0-1
    return croped_video

#####################################################################################################
# ----------------------------------  Add Loss Function  -----------------------------------------
#####################################################################################################

def get_center_loss(features, labels, alpha, num_classes):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
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

#############################################################################
##  generate the training set path list and test set path list from the given json data
def load_json(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        obj = json.load(f, encoding='utf-8')
    return obj

def GenerateTrainTestData():
    TrainData=load_json('data/something-something-v2-train.json')
    TestData=load_json('data/something-something-v2-validation.json')
    Label_map=load_json('data/something-something-v2-labels.json')
    train_file=open('trainlist.txt','w+')
    for item in TrainData:
        label=item['template']
        label=str(label)
        label = label.replace('[','')
        label = label.replace(']','')
        if label in Label_map.keys():
            id=int(item['id'])
            class_ind=int(Label_map[label])
            str_list=str(id)+' '+str(class_ind)
            print (str_list)
            train_file.write(str_list)
            train_file.write('\n')
    train_file.close()

    test_file = open('testlist.txt', 'w+')
    for item in TestData:
        label = item['template']
        label = str(label)
        label = label.replace('[', '')
        label = label.replace(']', '')
        if label in Label_map.keys():
            id = int(item['id'])
            class_ind = int(Label_map[label])
            str_list = str(id) + ' ' + str(class_ind)
            print (str_list)
            test_file.write(str_list)
            test_file.write('\n')
    test_file.close()


#######################################################################
####   generate npy data from webm, using multi-rpocessing module pool.map(func,arg=())
def webm2npy():
    files=os.listdir('data/something-something-v2/20bn-something-something-v2/')
    path=[]
    path_1=[]
    for item in files:
        path.append('data/something-something-v2/20bn-something-something-v2/'+item)
        path_0='/gruntdata/ercong/something-something-v2/'+item
        path_1.append(path_0.split('.')[0])
    pool = Pool(64)
    pool.map(save_video, path)
    pool.close()
    pool.join()


def save_video(path):
    if '.webm' in path:
        video= skvideo.io.vread(path,as_grey=False)
        path_0='/gruntdata/ercong/something-something-v2/'+path.split('/')[-1]
        path_0=path_0.split('.')[0]
        if video.shape[0] > 28:
            ind = np.linspace(0, video.shape[0], 28, endpoint=False)
            ind = ind.astype('int')
            video = video[ind, :, :, :]
        video_1 = []
        for i in list(range(video.shape[0])):
            video1 = video[i, :, :, :]
            video2 = resize(video1, (240, 260))
            video_1.append(video2)
        video_1 = np.array(video_1)
        video_1=255*video_1
        video_1=video_1.astype('uint8')
        np.save(path_0,video_1)



if __name__=='__main__':
    # GenerateTrainTestData()
    webm2npy()

