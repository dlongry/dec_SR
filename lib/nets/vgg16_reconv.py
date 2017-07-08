# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg


class vgg16(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)

#-----------keras function definition---------------------------

  def slice_1(self,t):
    return t[:, 0, :, :]

  def slice_2(self,t):
    return t[:, 1:, :, :]

  def slice_3(self,t):
    return t[:, 0, :]

  def slice_4(self,t):
    return t[:, 1:, :]
  
  def slice_3x3(self,t,x,y):
    return t[:,x:x+3,y:y+3,:]

  def single_relation_conv(self,input_map,MLP_units,channel,is_training,reuse_x=None): #fall feature map: slow version
    w,h,c=input_map.shape[1],input_map.shape[2],input_map.shape[3]
    print("input_shape:",w,h,c)
    print("begin get features...")
    all_num = w*h*w*h
    ck=2*int(c)
    features = []
    for k1 in range(w):
      features1 = self.slice_1(input_map)
      pool5 = self.slice_2(input_map) # like cut layer one by one
      for k2 in range(h):
        features2 = self.slice_3(features1)
        features1 = self.slice_4(features1)
        features.append(features2)
    print("beging get relationships")
    relations = []
    for feature1 in features:
      for feature2 in features:
        feature_all = tf.concat([feature1,feature2],1)
            #print(feature_all.shape)
        relations.append(feature_all)
    relations_map=tf.stack(relations,2)
    print(relations_map.shape)
    relations_map = tf.expand_dims(relations_map,3)
    print(relations_map.shape)
    print("beging get  mid relationships")
    mid_relations=slim.conv2d(relations_map, MLP_units, [ck,1], padding="VALID",trainable=is_training, scope="mid_relations1",reuse=reuse_x)
    print("mid_relations:",mid_relations.shape)
    mid_relations=slim.conv2d(mid_relations, MLP_units, [1,1], padding="VALID",trainable=is_training, scope="mid_relations2",reuse=reuse_x)
    print("mid_relations2:",mid_relations.shape)
    mid_relations=slim.conv2d(mid_relations, channel, [1,1], padding="VALID",trainable=is_training, scope="mid_relations3",reuse=reuse_x)
    print("mid_relations3:",mid_relations.shape)
    rn_map = slim.avg_pool2d(mid_relations,[1,all_num])
    print("rn:",rn_map.shape)
    return rn_map

  def relation_conv(self,input_map,MLP_units,out_channel,kernel_size,is_training=True):
    w,h=input_map.shape[1],input_map.shape[2]
    print("relation_conv",w,h,kernel_size)
    w=w-kernel_size
    print("src_w:",w)
    h=h-kernel_size
    h_now=0
    w_now=0
    result_list=[]
    while h_now <= h:
      #begin_x=[0,w_now,h_now,0]
      #size_x=[-1,kernel_size,kernel_size,-1]
      out=self.slice_3x3(input_map,w_now,h_now)
      print("out_shape",out.shape)
      if h_now==0 and w_now ==0:
        mid_result=self.single_relation_conv(out,MLP_units,out_channel,is_training,reuse_x=None) #cut feature map
      else:       
        mid_result=self.single_relation_conv(out,MLP_units,out_channel,is_training,reuse_x=True) #cut feature map
      result_list.append(mid_result)
      if w_now==w:
        w_now=0
        h_now=h_now+1
        print("h_now",h_now)
      else:
        w_now=w_now+1
        print("w_now",w_now)
    w=w+1
    h=h+1
    result_map=tf.stack(result_list)
    print("result_map",result_map.shape)
    result=tf.reshape(result_map,(-1,int(w),int(h),out_channel)) #w,h or h,w?
    print("result_shape",result.shape)
    return result

#----------------------------------------------------------
  def build_network(self, sess, is_training=True):
    with tf.variable_scope('vgg_16', 'vgg_16'):
      # select initializers
      if cfg.TRAIN.TRUNCATED:
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
      else:
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

      net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                        trainable=False, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv3')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')
      self._act_summaries.append(net)
      self._layers['head'] = net
      # build the anchors for the image
      self._anchor_component()

      # rpn
      rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError

      # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net, rois, "pool5")
      else:
        raise NotImplementedError
      #-------------------------------------pool5 here use RN network-------------------------:
      pool5_shapes=pool5.shape
      #rn_map=self.single_relation_conv(pool5,512,512,is_training)
      rn_map = self.relation_conv(pool5,256,256,3,is_training)
      rn_flatten =slim.flatten(rn_map,scope="flatten")
            
      print("rn:",rn_flatten.shape)
      rn = slim.fully_connected(rn_flatten,256,activation_fn=None, scope='MLP_l1')
      rn = slim.batch_norm(rn)
      rn = tf.nn.relu(rn)
      if is_training:
         rn = slim.dropout(rn, keep_prob=0.5, is_training=True, scope='drop1')
  
      rn = slim.fully_connected(rn, 256,weights_initializer=initializer,trainable=is_training,activation_fn=None, scope='MLP_l2')
      rn = slim.batch_norm(rn)
      rn = tf.nn.relu(rn)
      if is_training:
         rn = slim.dropout(rn, keep_prob=0.5, is_training=True, scope='drop2')
      print("RN OK")
      

      ''' pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
      '''


      #------------------------------using RN to get f7-----------------------------
      cls_score = slim.fully_connected(rn, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(rn, self._num_classes * 4, 
                                       weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')

      self._predictions["rpn_cls_score"] = rpn_cls_score
      self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
      self._predictions["rpn_cls_prob"] = rpn_cls_prob
      self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
      self._predictions["cls_score"] = cls_score
      self._predictions["cls_prob"] = cls_prob
      self._predictions["bbox_pred"] = bbox_pred
      self._predictions["rois"] = rois

      self._score_summaries.update(self._predictions)

      return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == 'vgg_16/conv1/conv1_1/weights:0':
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        #fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        #fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({"vgg_16/conv1/conv1_1/weights": conv1_rgb})
        
        #restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv, 
                                     # "vgg_16/fc7/weights": fc7_conv,
                                     # "vgg_16/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        #sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv, 
                            #self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
        #sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv, 
                            #self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))
