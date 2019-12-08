import sonnet as snt
import tensorflow as tf

class Unit3D(snt.AbstractModule):
  def __init__(self, output_channels,
               kernel_shape=(1, 1, 1),
               stride=(1, 1, 1),
               activation_fn=tf.nn.relu,
               use_batch_norm=True,
               use_bias=False,
               name='unit_3d'):
    super(Unit3D, self).__init__(name=name)
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._stride = stride
    self._use_batch_norm = use_batch_norm
    self._activation_fn = activation_fn
    self._use_bias = use_bias

  def _build(self, inputs, is_training):
    net = snt.Conv3D(output_channels=self._output_channels,
                     kernel_shape=self._kernel_shape,
                     stride=self._stride,
                     padding=snt.SAME,
                     use_bias=self._use_bias)(inputs)
    if self._use_batch_norm:
      bn = snt.BatchNorm()
      net = bn(net, is_training=is_training, test_local_stats=False)
      # net=tf.contrib.layers.group_norm(net,groups=32,channels_axis=-1,reduction_axes=(-4,-3,-2))
    if self._activation_fn is not None:
      net = self._activation_fn(net)
    return net



class Res3D(snt.AbstractModule):
  def __init__(self, num_classes=174, spatial_squeeze=True, name='Res3D'):
    super(Res3D, self).__init__(name=name)
    self._num_classes = num_classes
    self._spatial_squeeze = spatial_squeeze

  def _build(self, inputs, is_training, dropout_keep_prob=1.0):
      net = inputs

      stage='stage_1'
      with tf.variable_scope(stage):
          net = Unit3D(output_channels=64, kernel_shape=[1, 7, 7], stride=[1, 2, 2], name='Conv_11')(net,is_training=is_training)
          net = Unit3D(output_channels=64, kernel_shape=[7, 1, 1], stride=[1, 1, 1], name='Conv_12')(net,is_training=is_training)
          net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding=snt.SAME, name='Pooling_1')
          net = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv_2')(net, is_training=is_training)

      stage='stage_2'
      with tf.variable_scope(stage):
          net=self.ConvBlock(net, [64,64,128], strides=[1,1,1,1,1], name=stage+'a', is_training=is_training)
          net=self.IdentityBlock(net, [64,64,128], name=stage+'b', is_training=is_training)
          net = self.IdentityBlock(net, [64, 64, 128], name=stage+'c', is_training=is_training)

      stage = 'stage_3'
      with tf.variable_scope(stage):
          net = self.ConvBlock(net, [128, 128, 256], strides=[1, 1, 2, 2, 1], name=stage + 'a', is_training=is_training)
          net = self.IdentityBlock(net, [128, 128, 256], name=stage + 'b', is_training=is_training)
          net = self.IdentityBlock(net, [128, 128, 256], name=stage + 'c', is_training=is_training)

      stage = 'stage_4'
      with tf.variable_scope(stage):
          net = self.ConvBlock(net, [256, 256, 512], strides=[1, 2, 2, 2, 1], name=stage + 'a', is_training=is_training)
          net = self.IdentityBlock(net, [256, 256, 512], name=stage + 'b', is_training=is_training)
          net = self.IdentityBlock(net, [256, 256, 512], name=stage + 'c', is_training=is_training)
          net = self.IdentityBlock(net, [256, 256, 512], name=stage + 'd', is_training=is_training)
          net = self.IdentityBlock(net, [256, 256, 512], name=stage + 'e', is_training=is_training)

      stage = 'stage_5'
      with tf.variable_scope(stage):
          net = self.ConvBlock(net, [512, 512, 1024], strides=[1, 2, 2, 2, 1], name=stage + 'a', is_training=is_training)
          net = self.IdentityBlock(net, [512, 512, 1024], name=stage + 'b', is_training=is_training)
          net = self.IdentityBlock(net, [512, 512, 1024], name=stage + 'c', is_training=is_training)
          net = tf.nn.avg_pool3d(net, ksize=[1, 3, 7, 7, 1], strides=[1, 1, 1, 1, 1], padding=snt.VALID)
          net = tf.nn.dropout(net, dropout_keep_prob)
          net = tf.squeeze(net, [2, 3])
          net = tf.reduce_mean(net,axis=1)
          logits = tf.layers.dense(net, units=self._num_classes,use_bias=True)
          return logits







  def ConvBlock(self,Input_X,num_channels,strides,name,is_training):
      net = Unit3D(output_channels=num_channels[0], kernel_shape=[1, 3, 3], stride=strides,name=name+'_Conv_11')(Input_X,is_training=is_training)
      # net = Unit3D(output_channels=num_channels[0], kernel_shape=[3, 1, 1], name=name+'_Conv_12')(net,is_training=is_training)
      net = Unit3D(output_channels=num_channels[1], kernel_shape=[1, 3, 3], name=name+'_Conv_21')(net,is_training=is_training)
      # net = Unit3D(output_channels=num_channels[1], kernel_shape=[3, 1, 1], name=name+'_Conv_12')(net,is_training=is_training)
      net = Unit3D(output_channels=num_channels[2], kernel_shape=[1, 3, 3], name=name+'_Conv_31')(net,is_training=is_training)
      # net = Unit3D(output_channels=num_channels[2], kernel_shape=[3, 1, 1], name=name+'_Conv_32')(net,is_training=is_training)
      shortcut=Unit3D(output_channels=num_channels[2], kernel_shape=[1, 1, 1], stride=strides,name=name+'_shortcut')(Input_X,is_training=is_training)
      Output_X = net + shortcut
      return tf.nn.relu(Output_X)


  def IdentityBlock(self,Input_X,num_channels,name,is_training):
      net = Unit3D(output_channels=num_channels[0], kernel_shape=[1, 3, 3], name=name+'_IB_Conv_11')(Input_X,is_training=is_training)
      net = Unit3D(output_channels=num_channels[0], kernel_shape=[3, 1, 1], name=name+'_IB_Conv_12')(net,is_training=is_training)
      net = Unit3D(output_channels=num_channels[1], kernel_shape=[1, 3, 3], name=name+'_IB_Conv_21')(net,is_training=is_training)
      net = Unit3D(output_channels=num_channels[1], kernel_shape=[3, 1, 1], name=name+'_IB_Conv_12')(net,is_training=is_training)
      net = Unit3D(output_channels=num_channels[2], kernel_shape=[1, 3, 3], name=name+'_IB_Conv_31')(net,is_training=is_training)
      net = Unit3D(output_channels=num_channels[2], kernel_shape=[3, 1, 1], name=name+'_IB_Conv_32')(net,is_training=is_training)
      Output_X=net+Input_X
      return tf.nn.relu(Output_X)

