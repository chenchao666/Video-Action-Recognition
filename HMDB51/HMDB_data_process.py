import os
import skvideo
skvideo.setFFmpegPath("/home/ercong.cc/software/ffmpeg/bin")
import skvideo.io
import os.path
import random
import cv2
import numpy as np

def get_class_Ind():
    path=os.listdir('hmdb51')
    path.sort()
    return path


def get_train_list(path,split='_test_split1'):
    train_list=open('train_list1.txt','w+')
    test_list=open('test_list1.txt','w+')
    for i in range(len(path)):
        file_name=path[i]+split
        file_name='test_train_splits/'+file_name+'.txt'
        with open(file_name) as file:
            path_list=[row.strip() for row in list(file)]
            file_path_list = [row.split(' ')[0] for row in path_list]
            file_label_list = [row.split(' ')[1] for row in path_list]
            for j in range(len(file_path_list)):
                if file_label_list[j]=='1':
                    string_name=path[i]+'/'+file_path_list[j]+' '+str(i)
                    train_list.write(string_name)
                    train_list.write('\n')
                if file_label_list[j]=='2':
                    string_name=path[i]+'/'+file_path_list[j]+' '+str(i)
                    test_list.write(string_name)
                    test_list.write('\n')
    train_list.close()
    test_list.close()



def avi2npy():
    files=os.listdir('hmdb51/')
    id=0
    for file in files:
        id=id+1
        print id
        avi_name_list=os.listdir('hmdb51/'+file)
        for name in avi_name_list:
            if '.avi' in name:
                split=name.split('.')
                print name
                video=skvideo.io.vread('hmdb51/'+file+'/'+name,as_grey=False)
                if video.shape[0]>150:
                    ind=int(np.floor((video.shape[0]-150)/2))
                    video=video[ind:ind+150,:,:,:]
                frame=video.shape[0]
                if frame>48:
                    ind = np.linspace(0,frame,48,endpoint=False)
                    ind=ind.astype('int')
                    video = video[ind, :, :, :]
                elif frame>32:
                    ind = np.linspace(0,frame,32,endpoint=False)
                    ind=ind.astype('int')
                    video = video[ind, :, :, :]

                videodata=video
                videodata = videodata / 128.0 - 1
                if videodata.shape[1] >= videodata.shape[2]:
                    videodata_0 = np.zeros((videodata.shape[0], int(videodata.shape[1] * 256.0 / videodata.shape[2]), 256, 3))
                    for i in range(videodata.shape[0]):
                        videodata_0[i, :, :, :] = cv2.resize(videodata[i, :, :, :], (256, int(videodata.shape[1] * 256.0 / videodata.shape[2])), interpolation=cv2.INTER_CUBIC)
                if videodata.shape[1] < videodata.shape[2]:
                    videodata_0 = np.zeros((videodata.shape[0], 256, int(videodata.shape[2] * 256.0 / videodata.shape[1]), 3))
                    for i in range(videodata.shape[0]):
                        videodata_0[i, :, :, :] = cv2.resize(videodata[i, :, :, :], (int(videodata.shape[2] * 256.0 / videodata.shape[1]), 256), interpolation=cv2.INTER_CUBIC)

                videodata=128*(videodata_0+1)
                videodata=videodata.astype('uint8')
                file_path = 'hmdb51/' + file + '/' + split[0] + '.npy'
                np.save(file_path,videodata)




def main():
    # path=get_class_Ind()
    # get_train_list(path,split='_test_split1')
    avi2npy()

if __name__ == '__main__':
    main()
