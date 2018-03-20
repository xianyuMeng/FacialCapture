from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pickle 
import pprint

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

class Module(object):
    def __init__(self, config, mode):
        assert mode in {'train', 'inference'}
        self.mode = mode
        self.config = config
        self.reader = tf.TFRecordReader()

    def build_bshape(self):
        pkl_file = open(self.config.pkl_fname)
        self.bshape = pickle.load(pkl_file)
        print("Loading pickle file of shape {}".format(self.bshape.shape))
        self.bshape = tf.constant(self.bshape)
        pkl_file.close()
        mean_mesh_file = open(self.config.mean_pkl_fname)
        self.mean_mesh = pickle.load(mean_mesh_file)
        print("Loading mean mesh file of shape {}".format(self.mean_mesh.shape))
        self.mean_mesh = tf.constant(self.mean_mesh)
        mean_mesh_file.close()


    def build_inputs(self):
        if self.mode == 'train':
            data_files = []
            for pattern in self.config.file_pattern.split(','):
                print(tf.gfile.Glob(pattern))
                data_files.extend(tf.gfile.Glob(pattern))
            #data_files = tf.gfile.Glob(self.config.file_pattern)
            print("number of trainig record files : {}".format(len(data_files)))
            filename_queue = tf.train.string_input_producer( \
                    data_files, \
                    shuffle = True, \
                    capacity = 16)
            _, example = self.reader.read(filename_queue)
            proto_value = tf.parse_single_example(\
                    example, \
                    features = { \
                    self.config.image_feature_name: tf.FixedLenFeature([], dtype=tf.string),\
                    self.config.predict_feature_name: tf.FixedLenFeature([self.config.predict_num], dtype=tf.float32) \
                   })
            image = proto_value[self.config.image_feature_name]
            weight = proto_value[self.config.predict_feature_name]
            image = self.process_image(image)
            #weight = self.process_weight(weight)
            self.image, self.weight = tf.train.batch_join([(image, weight)], batch_size = self.config.batch_size)
            print('get image of shape{}'.format(self.image.get_shape()))

        else:
            self.image_feed = tf.placeholder(dtype=tf.string, shape=[], name='image_feed')
            self.image = tf.expand_dims(self.process_image(self.image_feed),0)

    def process_image(self, im_str):
        image = tf.reshape(tf.decode_raw(im_str, out_type=tf.uint8), (self.config.image_width, self.config.image_height, 1))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images( \
                image, \
                size=[self.config.image_height, self.config.image_width],\
                method=tf.image.ResizeMethod.BILINEAR)
        return image
    
    def setup_global_step(self):
        self.global_step = tf.Variable( \
                initial_value=0, \
                name = 'global_step', \
                trainable = False, \
                collections = [tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    def build_model(self):
        with tf.variable_scope('face'):
            weight_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.1)
            # See Section 3.1 of paper ` Production-Level Facial Performance Capture ` for more information
            conv1a = tf.contrib.layers.conv2d( \
                    self.image, \
                    64, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False), \
                   # weights_regularizer = weight_regularizer, \
                    scope = "conv1a" \
                    )
            conv1b = tf.contrib.layers.conv2d( \
                    conv1a, \
                    64, \
                    kernel_size = [3, 3], \
                    stride = 1 , \
                   # weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False), \
                    weights_regularizer = weight_regularizer, \
                    scope = "conv1b"\
                    )
            conv2a = tf.contrib.layers.conv2d( \
                    conv1b, \
                    96, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False),\
                   # weights_regularizer = weight_regularizer, \
                    scope = "conv2a" \
                    )
            conv2b = tf.contrib.layers.conv2d( \
                    conv2a, \
                    96, \
                    kernel_size = [3, 3], \
                    stride = 1, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False),\
                   # weights_regularizer = weight_regularizer, \
                    scope = "conv2b")
            conv3a = tf.contrib.layers.conv2d( \
                    conv2b, \
                    144, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False), \
                   # weights_regularizer = weight_regularizer, \
                    scope = 'conv3a')
            conv3b = tf.contrib.layers.conv2d( \
                    conv3a, \
                    144, \
                    kernel_size = [3, 3], \
                    stride = 1, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False), \
                   # weights_regularizer = weight_regularizer, \
                    scope = 'conv3b')
            print("conv3b shape is {}".format(conv3b.get_shape()))
            conv4a = tf.contrib.layers.conv2d( \
                    conv3b, \
                    216, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                   # weights_regularizer = weight_regularizer, \
                    scope = 'conv4a')
            conv4b = tf.contrib.layers.conv2d( \
                    conv4a, \
                    216, \
                    kernel_size = [3, 3], \
                    stride = 1, \
                   # weights_regularizer = weight_regularizer, \
                    scope = 'conv4b')
            conv5a = tf.contrib.layers.conv2d( \
                    conv4b, \
                    324, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                   # weights_regularizer = weight_regularizer, \
                    scope = 'conv5a')
            conv5b = tf.contrib.layers.conv2d( \
                    conv5a, \
                    324, \
                    kernel_size = [3, 3], \
                    stride = 1, \
                   # weights_regularizer = weight_regularizer, \
                    scope = 'conv5b')
            conv6a = tf.contrib.layers.conv2d( \
                    conv5b, \
                    486, \
                    kernel_size = [3, 3], \
                   # weights_regularizer = weight_regularizer, \
                    stride = 2, \
                    scope = 'conv6a')

            conv6b = tf.contrib.layers.conv2d( \
                    conv6a, \
                    486, \
                    kernel_size = [3, 3], \
                   # weights_regularizer = weight_regularizer, \
                    stride = 1, \
                    scope = 'conv6b')
            print("conv6b size is {}".format(conv6b.get_shape()))
            drop = tf.contrib.layers.dropout(\
                    conv6b, \
                    0.8, \
                    is_training = (self.mode == 'train'), \
                    scope = 'dropout')
            fc = tf.contrib.layers.fully_connected( \
                    inputs = drop, \
                    num_outputs = self.config.pca_num, \
                    activation_fn = None, \
                    scope = 'fully_connected')
            print("fc shape {}".format(fc.get_shape()))
            output = tf.contrib.layers.flatten(fc)
            output = tf.contrib.layers.fully_connected( \
                    inputs = output, \
                    num_outputs = self.config.pca_num, \
                    activation_fn = None, \
                    scope = 'output')
            #self.bshape = tf.reshape(self.bshape, [1, self.config.predict_num])
            self.pca_coef = output
            output = (tf.matmul(output, self.bshape))
       

           # output = tf.contrib.layers.fully_connected( \
           #         inputs = output, \
           #         num_outputs = self.config.predict_num, \
           #         scope = 'output')

            print('shape is {}'.format(output.get_shape()))
            output = tf.add(self.mean_mesh, output)

            if self.mode == 'inference' :
                self.prediction = output
            else :
                losses = tf.square(output - self.weight)
                batch_loss = tf.reduce_sum(losses)
                tf.losses.add_loss(batch_loss)
                tf.summary.scalar('losses/batch_loss', batch_loss)
                self.total_loss = tf.losses.get_total_loss()
                tf.summary.scalar('losses/total_loss', self.total_loss)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)


    def build(self):
        self.build_bshape()
        if self.mode == 'train' :
            self.setup_global_step()
        self.build_inputs()
        self.build_model()




