import os
import skvideo
skvideo.setFFmpegPath("/home/ercong.cc/software/ffmpeg/bin")
import skvideo.io
import os.path
import random
import numpy as np

def get_train_test_lists(version='01'):
    test_file = os.path.join('ucfTrainTestlist', 'testlist' + version + '.txt')
    train_file = os.path.join('ucfTrainTestlist', 'trainlist' + version + '.txt')
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]

    # Build the train list. Extra step to remove the class index.
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]

    # Set the groups in a dictionary.
    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups

def process_test_data(group_lists):
    Dict={}
    with open('ucfTrainTestlist/classInd.txt') as file:
        classInd = [row.strip() for row in list(file)]
        for row in classInd:
            split = row.split(' ')
            Dict[split[1]]=split[0]
        file.close()

    test_list=group_lists['test']
    file=open('ucfTrainTestlist/testlist_3.txt','w+')
    for test_file in test_list:
        split = test_file.split('/')
        class_name = split[0]
        class_label=Dict[class_name]
        str=test_file+' '+class_label
        print str
        file.write(str)
        file.write('\n')
    file.close()
    print ('done')


def avi2npy():
    files=os.listdir('UCF-101/')
    id=0
    for file in files:
        id=id+1
        print id
        avi_name_list=os.listdir('UCF-101/'+file)
        for name in avi_name_list:
            if '.avi' in name:
                split=name.split('.')
                print name
                video=skvideo.io.vread('UCF-101/'+file+'/'+name,as_grey=False)
                if video.shape[0]>150:
                    ind=int(np.floor((video.shape[0]-150)/2))
                    video=video[ind:ind+150,:,:,:]
                frame=video.shape[0]
                if frame>32:
                    ind = np.linspace(0,frame,32,endpoint=False)
                    ind=ind.astype('int')
                    video = video[ind, :, :, :]
                file_path='UCF-101/'+file+'/'+split[0]+'.npy'
                np.save(file_path,video)




def main():
    # group_lists = get_train_test_lists()
    # process_test_data(group_lists)
    avi2npy()

if __name__ == '__main__':
    main()
