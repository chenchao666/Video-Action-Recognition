from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


class Unit3D(snt.AbstractModule):
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""
  def __init__(self, output_channels,
               kernel_shape=(1, 1, 1),
               stride=(1, 1, 1),
               activation_fn=tf.nn.relu,
               use_batch_norm=True,
               use_bias=False,
               name='unit_3d'):
    """Initializes Unit3D module."""
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
    if self._activation_fn is not None:
      net = self._activation_fn(net)
    return net


class InceptionI3d(snt.AbstractModule):
  VALID_ENDPOINTS = (
      'Conv3d_1a_7x7',
      'MaxPool3d_2a_3x3',
      'Conv3d_2b_1x1',
      'Conv3d_2c_3x3',
      'MaxPool3d_3a_3x3',
      'Mixed_3b',
      'Mixed_3c',
      'MaxPool3d_4a_3x3',
      'Mixed_4b',
      'Mixed_4c',
      'Mixed_4d',
      'Mixed_4e',
      'Mixed_4f',
      'MaxPool3d_5a_2x2',
      'Mixed_5b',
      'Mixed_5c',
      'Logits',
      'Predictions',
  )

  def __init__(self, num_classes=400, spatial_squeeze=True,
               final_endpoint='Logits', name='inception_i3d'):


    if final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError('Unknown final endpoint %s' % final_endpoint)

    super(InceptionI3d, self).__init__(name=name)
    self._num_classes = num_classes
    self._spatial_squeeze = spatial_squeeze
    self._final_endpoint = final_endpoint


  def _build(self, inputs, is_training, dropout_keep_prob=1.0):

    if self._final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

    net = inputs
    end_points = {}
    end_point = 'Conv3d_1a_7x7'
    net = Unit3D(output_channels=64, kernel_shape=[7, 7, 7],
                 stride=[1, 2, 2], name=end_point)(net, is_training=is_training)   #[2,2,2]
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
    end_point = 'MaxPool3d_2a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
    end_point = 'Conv3d_2b_1x1'
    net = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                 name=end_point)(net, is_training=is_training)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
    end_point = 'Conv3d_2c_3x3'
    net = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                 name=end_point)(net, is_training=is_training)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
    end_point = 'MaxPool3d_3a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_3b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=32, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_3c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_4a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],     #[1,2,2,2,1]
                           padding=snt.SAME, name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=208, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=48, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=224, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4d'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=256, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4e'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=144, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=288, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4f'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_5a_2x2'
    net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding=snt.SAME, name=end_point)  #stride [1,2,2,2,1]
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_5b'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0a_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_5c'
    with tf.variable_scope(end_point):
      with tf.variable_scope('Branch_0'):
        branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
      with tf.variable_scope('Branch_1'):
        branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_1,
                                                is_training=is_training)
      with tf.variable_scope('Branch_2'):
        branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                          name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                          name='Conv3d_0b_3x3')(branch_2,
                                                is_training=is_training)
      with tf.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                    strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                    name='MaxPool3d_0a_3x3')
        branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                          name='Conv3d_0b_1x1')(branch_3,
                                                is_training=is_training)
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points


##################################################################################################
    # end_point = 'Logits'
    # with tf.variable_scope(end_point):
    #   net = tf.nn.avg_pool3d(net, ksize=[1, 7, 7, 7, 1],      #[1,2,7,7,1]
    #                          strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    #   net = tf.nn.dropout(net, dropout_keep_prob)
    #   logits = Unit3D(output_channels=self._num_classes,
    #                   kernel_shape=[1, 1, 1],
    #                   activation_fn=None,
    #                   use_batch_norm=False,
    #                   use_bias=True,
    #                   name='Conv3d_0c_1x1')(net, is_training=is_training)
    #   if self._spatial_squeeze:
    #     logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
    # averaged_logits = tf.reduce_mean(logits, axis=1)
#####################################################################################################



#############################################################
    end_point = 'Logits'
    with tf.variable_scope(end_point):
        net = tf.nn.avg_pool3d(net, ksize=[1, 2, 3, 3, 1],strides=[1, 1, 1, 1, 1], padding=snt.VALID)
        net = tf.nn.dropout(net, dropout_keep_prob)
        averaged_logits = self.Bilinear_Pooling(net)
        # averaged_logits=self.Group_BP(net,split=16)


        # ###########################################################################################
        net=tf.nn.max_pool3d(end_points['Mixed_3c'], ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding=snt.SAME)
        net = Unit3D(output_channels=256, kernel_shape=[1, 1, 1])(net, is_training=is_training)
        averaged_logits_3c = self.Bilinear_Pooling(net)
        # averaged_logits_3c = self.Group_BP(net,split=12)


        # ###############################################################################################
        # net = tf.nn.max_pool3d(end_points['Mixed_4b'], ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding=snt.SAME)
        # net = Unit3D(output_channels=192, kernel_shape=[1, 1, 1])(net, is_training=is_training)
        # # averaged_logits_4b = self.Group_BP(net,split=12)
        # averaged_logits_4b = self.Bilinear_Pooling(net)
        #
        # #########################################################################################
        net = tf.nn.max_pool3d(end_points['Mixed_4c'], ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding=snt.SAME)
        net = Unit3D(output_channels=256, kernel_shape=[1, 1, 1])(net, is_training=is_training)
        averaged_logits_4c = self.Bilinear_Pooling(net)
        # averaged_logits_4c =  self.Group_BP(net,split=16)
        #
        # #########################################################################################
        # net = tf.nn.max_pool3d(end_points['Mixed_4d'], ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding=snt.SAME)
        # net = Unit3D(output_channels=256, kernel_shape=[1, 1, 1])(net, is_training=is_training)
        # averaged_logits_4d = self.Bilinear_Pooling(net)
        #
        # #########################################################################################
        net = tf.nn.max_pool3d(end_points['Mixed_4e'], ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding=snt.SAME)
        net = Unit3D(output_channels=256, kernel_shape=[1, 1, 1])(net, is_training=is_training)
        averaged_logits_4e = self.Bilinear_Pooling(net)
        # averaged_logits_4e = self.Group_BP(net,split=24)
        #
        # #########################################################################################
        # net = tf.nn.max_pool3d(end_points['Mixed_4f'], ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding=snt.SAME)
        # net = Unit3D(output_channels=384, kernel_shape=[1, 1, 1])(net, is_training=is_training)
        # averaged_logits_4f = self.Bilinear_Pooling(net)
        #
        # #########################################################################################
        net = tf.nn.max_pool3d(end_points['Mixed_5b'], ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding=snt.SAME)
        net = Unit3D(output_channels=256, kernel_shape=[1, 1, 1])(net, is_training=is_training)
        averaged_logits_5b = self.Bilinear_Pooling(net)
        # averaged_logits_5b = self.Group_BP(net,split=32)

        averaged_logits=tf.concat([averaged_logits,averaged_logits_3c,averaged_logits_4c,averaged_logits_4e,averaged_logits_5b],1)

        # averaged_logits = self.Gate(averaged_logits)
#############################################################

    end_points[end_point] = averaged_logits
    if self._final_endpoint == end_point: return averaged_logits, end_points

    end_point = 'Predictions'
    predictions = tf.nn.softmax(averaged_logits)
    end_points[end_point] = predictions
    return predictions, end_points



  def Bilinear_Pooling(self,net):
      # net=self.SE_Block(net)
      net = tf.reshape(net, [net.shape[0], -1, net.shape[4]])
      mean_value = tf.reduce_mean(net, axis=[2])
      net = tf.transpose(net, [2,0,1])
      net = net - mean_value
      net = tf.transpose(net, [1,2,0])
      Cov = tf.matmul(net, net, transpose_a=True)
      # Cov = self.SqrtRoot(Cov)
      averaged_logits = tf.reshape(Cov, [Cov.shape[0], -1])
      averaged_logits = tf.sign(averaged_logits) * tf.sqrt(tf.abs(averaged_logits) + 1e-12)
      averaged_logits = tf.transpose(tf.transpose(averaged_logits) / (tf.norm(averaged_logits, axis=1)))
      return averaged_logits


  def Group_BP(self,net,split=16):
      net_split=tf.split(net,split,axis=4)
      net_split=tf.convert_to_tensor(net_split)
      net_split = tf.reshape(net_split, [net_split.shape[0],net_split.shape[1], -1, net_split.shape[5]])
      mean_value=tf.reduce_mean(net_split,axis=[3])
      net_split=tf.transpose(net_split,[3,0,1,2])
      net_split=net_split-mean_value
      net_split=tf.transpose(net_split,[1,2,3,0])
      Cov = tf.matmul(net_split, net_split, transpose_a=True)
      Cov=tf.transpose(Cov,[1,2,3,0])
      averaged_logits = tf.reshape(Cov, [Cov.shape[0], -1])
      averaged_logits = tf.sign(averaged_logits) * tf.sqrt(tf.abs(averaged_logits) + 1e-12)
      averaged_logits = tf.transpose(tf.transpose(averaged_logits) / (tf.norm(averaged_logits, axis=1)))
      return averaged_logits


  def SqrtRoot(self,Cov):
      normalize=tf.trace(Cov)
      Cov= tf.transpose(Cov, [1, 2, 0])
      Cov=Cov/normalize
      Cov = tf.transpose(Cov, [2, 0, 1])
      Y0=Cov
      Z0=tf.eye(int(Cov.shape[1]))
      Z0=tf.expand_dims(Z0,0)
      Z0=tf.tile(Z0,multiples=[int(Cov.shape[0]),1,1])
      I=Z0
      Y1=0.5*tf.matmul(Y0,3*I-tf.matmul(Z0,Y0))
      Z1=0.5*tf.matmul(3*I-tf.matmul(Z0,Y0),Z0)
      Y2 = 0.5 * tf.matmul(Y1, 3 * I - tf.matmul(Z1, Y1))
      Z2 = 0.5 * tf.matmul(3 * I - tf.matmul(Z1, Y1), Z1)
      Y3 = 0.5 * tf.matmul(Y2, 3 * I - tf.matmul(Z2, Y2))
      Z3 = 0.5 * tf.matmul(3 * I - tf.matmul(Z2, Y2), Z2)
      Y4 = 0.5 * tf.matmul(Y3, 3 * I - tf.matmul(Z3, Y3))
      Z4 = 0.5 * tf.matmul(3 * I - tf.matmul(Z3, Y3), Z3)
      Y5 = 0.5 * tf.matmul(Y4, 3 * I - tf.matmul(Z4, Y4))
      Z5 = 0.5 * tf.matmul(3 * I - tf.matmul(Z4, Y4), Z4)
      Y6 = 0.5 * tf.matmul(Y5, 3 * I - tf.matmul(Z5, Y5))
      return Y6


  def SE_Block(self,net):
    ratio=32
    shape=net.get_shape()
    net_pool = tf.nn.avg_pool3d(net, ksize=[1, net.shape[1], net.shape[2], net.shape[3], 1], strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    squeeze = tf.squeeze(net_pool,[1,2,3])
    excitation = tf.layers.dense(squeeze,units=int(squeeze.shape[1])/ratio,activation=tf.nn.relu,use_bias=True)
    excitation = tf.layers.dense(excitation, units=int(squeeze.shape[1]),activation=tf.nn.relu,use_bias=True)
    net=tf.reshape(net,[net.shape[0],-1,net.shape[4]])
    net=tf.transpose(net,[1,0,2])
    net=tf.multiply(net,excitation)
    net=tf.transpose(net,[1,0,2])
    net=tf.reshape(net,shape)
    return net


  def Gate(self,net):
      ratio = 32
      gate = tf.layers.dense(net, 1, activation=tf.nn.relu, use_bias=True)
      gate = tf.layers.dense(gate, units=int(net.shape[1]), activation=tf.nn.relu, use_bias=True)
      net=tf.multiply(net,gate)
      net = tf.transpose(tf.transpose(net) / (tf.norm(net, axis=1)))
      return net




