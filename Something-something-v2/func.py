import tensorflow as tf

## Input: bxTxWxHxC
def ST_Attention(Net_In,Pooling_size=[1,1,2,2,1]):
    net = tf.layers.conv3d(Net_In, filters=256, kernel_size=[1, 3, 3], strides=[1, 1, 1],padding='SAME', name='Conv_11')
    net = tf.layers.conv3d(net, filters=256, kernel_size=[3, 1, 1], strides=[1, 1, 1], padding='SAME', name='Conv_12')
    net = tf.nn.max_pool3d(net,ksize=[1,1,2,2,1],strides=Pooling_size, padding='SAME',name='pool_1')
    net = tf.layers.conv3d(net, filters=256, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='SAME',name='Conv_21')
    net = tf.layers.conv3d(net, filters=256, kernel_size=[3, 1, 1], strides=[1, 1, 1], padding='SAME', name='Conv_22')
    net = tf.nn.max_pool3d(net, ksize=[1,1,2,2,1], strides=Pooling_size, padding='SAME',name='pool_2')
    net = tf.reduce_mean(net, axis=4)
    Net_In=tf.transpose(Net_In,[4,0,1,2,3])
    Net_Out=Net_In*net
    Net_Out=tf.transpose(Net_Out,[1,2,3,4,0])
    return Net_Out


## Input: bxTxWxHxC
def Motion_Block(Net_In):
    net = tf.layers.conv3d(Net_In, filters=128, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='SAME', name='Conv_1')
    net0 = net[:,1:,:,:,:]-net[:,:-1,:,:,:]
    net1=net[:,0,:,:,:]
    net1=tf.expand_dims(net1,axis=1)
    net_diff = tf.concat([net1,net0],axis=1)
    net = tf.layers.conv3d(net_diff, filters=128, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='SAME', name='Conv_2')
    net0 = net[:, 1:, :, :, :] - net[:, :-1, :, :, :]
    net1 = net[:, 0, :, :, :]
    net1 = tf.expand_dims(net1, axis=1)
    net_diff2 = tf.concat([net1, net0], axis=1)
    net_diff2 = tf.layers.conv3d(net_diff2, filters=128, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='SAME',name='Conv_3')
    Net_Out=net+net_diff+net_diff2
    return Net_Out


## Input: bxTxWxHxC
def TSM_Block(net):
    shape=net.shape
    shape=shape.as_list()
    fold=shape[4]/4
    Net_Out=tf.zeros(shape=shape)
    Net_Out[:,:-1,:,:,fold]=net[:,1:,:,:,:fold]
    Net_Out[:,1:,:,:,fold:2*fold]=net[:,:-1,:,:,fold:2*fold]
    Net_Out[:,:,:,:,2*fold:]=net[:,:,:,:,2*fold:]
    return Net_Out







