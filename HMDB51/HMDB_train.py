import os
import numpy as np
import tensorflow as tf
import HMDB_net
from HMDB_utils import *
import time
os.environ['CUDA_VISIBLE_DEVICES']='0'

NUM_CLASSES = 400
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('num_video_frames',28,'define the num of frames sampled in a video')
tf.flags.DEFINE_integer('image_size',224,'define the croped image size')
tf.flags.DEFINE_string('input_mode', 'rgb', 'rgb or flow')
tf.flags.DEFINE_float('LearningRate',0.001,'define the learn rate')
tf.flags.DEFINE_float('DropoutRate',0.3,'define the dropout rate')
tf.flags.DEFINE_integer('logging_step',20,'define the croped image size')
tf.flags.DEFINE_string('test_mode', 'partial', 'provide the test mode, full or partial')

Channel={'rgb':3,'flow':2}
Scope = {
    'rgb': 'RGB',
    'flow': 'Flow',
}
_CHECKPOINT_PATHS = {
    'rgb_only': '/home/ercong.cc/code/I3D/UCF101/data/checkpoints/rgb_scratch/model.ckpt',
    'flow_only': '/home/ercong.cc/code/I3D/UCF101/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': '/home/ercong.cc/code/I3D/UCF101/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': '/home/ercong.cc/code/I3D/UCF101/data/checkpoints/flow_imagenet/model.ckpt',
}

class I3D():
    def __init__(self,input_mode,class_num,batch_size,iters):
        self.input_mode=input_mode
        self.class_num=class_num
        self.batch_size=batch_size
        self.iters=iters
         #################  set the placehoder   ######################################
        self.input=tf.placeholder(tf.float32,shape=(self.batch_size, FLAGS.num_video_frames, FLAGS.image_size, FLAGS.image_size, Channel[self.input_mode]))
        self.label=tf.placeholder(tf.int32,shape=(self.batch_size))
        self.training_flag=tf.placeholder(tf.bool,shape=None)
        self.dropout_rate = tf.placeholder(tf.float32, shape=None)

        with tf.variable_scope(Scope[self.input_mode]):
            I3D = HMDB_net.InceptionI3d(NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            self.logits, _ = I3D(self.input, is_training=self.training_flag, dropout_keep_prob=self.dropout_rate)
            ###########  add a extra dropout layer and FC layer #########
            self.dropout=tf.nn.dropout(self.logits,keep_prob=0.1*self.dropout_rate)
            self.output = tf.layers.dense(self.dropout, self.class_num, use_bias=True)
            self.softmax = tf.nn.softmax(self.output)
            self.count = tf.nn.in_top_k(self.softmax, self.label, 1)



    def load_pretained_I3D(self):
        self.variable_map = {}
        self.var_list=tf.trainable_variables()
        self.train_var=[var for var in self.var_list if 'Mixed_5' in var.name or 'Mixed_4' in var.name or 'Logits' in var.name or 'dense' in var.name]
        self.train_var_1 = [var for var in self.var_list if 'Mixed_4' in var.name]
        self.train_var_2 =[var for var in self.var_list if 'Mixed_5' in var.name]
        self.train_var_3=[var for var in self.var_list if 'Logits' in var.name or 'dense' in var.name]
        print (self.train_var)
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == Scope[self.input_mode] and tmp[1] != 'dense' and tmp[2]!='Logits':
                self.variable_map[variable.name.replace(':0', '')] = variable
        self.saver = tf.train.Saver(var_list=self.variable_map, reshape=True)


    def Calculate_loss(self):
        self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.output))
        self.Cal_discriminative_loss()
        self.Cal_norm_loss()
        self.loss=self.class_loss


    def Cal_discriminative_loss(self):
        dis_loss,self.centers_update_op=get_center_loss(self.logits,self.label,0.2,101)
        self.discriminative_loss=1.0*dis_loss/(self.batch_size)

    def Cal_norm_loss(self):
        NORM=10.0
        self.norm_loss=tf.reduce_mean(tf.square(NORM-tf.norm(self.logits,axis=1)))


    def Train(self):
        self.load_pretained_I3D()
        self.Calculate_loss()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        boundaries = [300,10000, 15000, 20000]
        values = [0.1*FLAGS.LearningRate, FLAGS.LearningRate, 0.2*FLAGS.LearningRate, 0.05*FLAGS.LearningRate,0.01*FLAGS.LearningRate]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.control_dependencies([self.centers_update_op]):
                self.train_OP1 = tf.train.AdamOptimizer(learning_rate=0.05 * learning_rate).minimize(self.loss,var_list=self.train_var_1,global_step=global_step)
                self.train_OP2 = tf.train.AdamOptimizer(learning_rate=0.3 * learning_rate).minimize(self.loss,var_list=self.train_var_2,global_step=global_step)
                self.train_OP3 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,var_list=self.train_var_3,global_step=global_step)
                self.optimizer = tf.group(self.train_OP1,self.train_OP2,self.train_OP3)
                # self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,var_list=self.train_var,global_step=global_step)

        Data=DataLoader('hmdb51')
        path_list,label_list=Data.get_TrainData()
        path_list=np.array(path_list)
        label_list=np.array(label_list)
        label_list=label_list.astype('int32')
        num_samples=len(path_list)
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            self.sess=sess
            self.sess.run(tf.global_variables_initializer())
            sess.graph.finalize()
            self.saver.restore(self.sess, _CHECKPOINT_PATHS[self.input_mode + '_imagenet'])
            print ('Load pretained model successfully')
            step=0
            total_count=0
            while step < self.iters:
                start_time=time.time()
                step+=1
                sampled_idx=np.random.randint(0,num_samples,self.batch_size)
                batch_files=path_list[sampled_idx]
                batch_x=[avi_2_npy(os.path.join('hmdb51'+'/'+file),sampled_frame=FLAGS.num_video_frames,mode='npy',crop_flag='random_crop') for file in batch_files]
                batch_x=np.array(batch_x)
                batch_y=label_list[sampled_idx]
                _,loss,norm_loss,count=self.sess.run([self.optimizer, self.loss, self.norm_loss, self.count],
                                                    feed_dict={self.input: batch_x,
                                                               self.label: batch_y,
                                                               self.training_flag:True,
                                                               self.dropout_rate:FLAGS.DropoutRate})
                batch_count=np.sum(count)
                total_count+=batch_count
                if step%FLAGS.logging_step==0:
                    duration = time.time() - start_time
                    Accuracy=1.0*total_count/(FLAGS.logging_step*self.batch_size)
                    total_count = 0
                    print('#### Step = {} #### Total-Loss = {} ### Disc-Loss = {} #### Train-Accuracy ={} #### Time = {}####'.format(step, loss, norm_loss, Accuracy,duration))

                if step%500==0:
                    self.Test()
                if step%3000==0:
                    FLAGS.test_mode ='full'
                    print ('----------------------- Full Test Accuracy --------------------')
                    self.Test()
                    FLAGS.test_mode='partial'

    def Test(self):
        Data = DataLoader('hmdb51')
        path_list, label_list = Data.get_TestData()
        path_list = np.array(path_list)
        label_list = np.array(label_list)
        label_list = label_list.astype('int32')
        num_samples = len(path_list)
        total_count=0.0
        num_batch=0
        if FLAGS.test_mode=='partial':
            num_batch=50
            for i in range(num_batch):
                sampled_idx = np.random.randint(0, num_samples, self.batch_size)
                batch_file=path_list[sampled_idx]
                batch_x = [avi_2_npy(os.path.join('hmdb51'+'/'+file), sampled_frame=FLAGS.num_video_frames,mode='npy',crop_flag='center_crop') for file in batch_file]
                batch_x=np.array(batch_x)
                batch_y=label_list[sampled_idx]
                count = self.sess.run(self.count,feed_dict={self.input: batch_x, self.label: batch_y,self.training_flag: False, self.dropout_rate: 1.0})
                total_count+=np.sum(count)
        elif FLAGS.test_mode=='full':
            average_score=np.zeros((self.batch_size,self.class_num))
            num_batch=int(np.floor(num_samples/self.batch_size))
            for i in range (num_batch):
                for j in range(10):
                    batch_file=path_list[i*self.batch_size:(i+1)*self.batch_size]
                    batch_x = [avi_2_npy(os.path.join('hmdb51'+'/'+file), sampled_frame=FLAGS.num_video_frames,mode='npy',crop_flag='random_crop') for file in batch_file]
                    batch_x = np.array(batch_x)
                    batch_y = label_list[i*self.batch_size:(i+1)*self.batch_size]
                    score=self.sess.run(self.softmax,feed_dict={self.input: batch_x, self.label: batch_y, self.training_flag: False,self.dropout_rate: 1.0})
                    score = np.transpose(score) >= np.max(score, axis=1)
                    average_score = average_score + np.transpose(score.astype('float'))
                count=np.equal(np.argmax(average_score,1),batch_y)
                total_count += np.sum(count)
                average_score=np.zeros(self.class_num)
        Accuracy = 1.0 * total_count / (num_batch * self.batch_size)
        print ('Test Accuracy = {}'.format(Accuracy))



def main():
    input_mode=FLAGS.input_mode
    Net=I3D(input_mode,class_num=51,batch_size=28,iters=50000)
    Net.Train()


if __name__=='__main__':
    main()
