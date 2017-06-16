# Reference: https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
import os
import tensorflow as tf
import numpy as np
import cv2

vgg19_dir = "vgg19/"
VGG_MEAN = [103.939, 116.779, 123.68]

class VGG19:
    tensor_name_input_image = "images:0"
    arch="VGG19"

    def __init__(self, sess):
        print('Loading VGG19')
        self.data_dict = np.load(open(vgg19_dir + "vgg19.npy", "rb"), encoding="latin1").item()
        print('Finished loading VGG19')

        self.graph = sess.graph

        with self.graph.as_default():
            x = tf.placeholder(tf.float32, [None, None, None, 3], name='images')

            self.conv1_1 = self.conv_layer(x, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
            self.pool3 = self.max_pool(self.conv3_4, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
            self.pool4 = self.max_pool(self.conv4_4, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")

            # Removed FC layers, don't need them in this case

        self.layer_names = [name for name in self.get_all_layer_names() if ('Conv2D') in name]
        self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]
        self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)
        return

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def get_layer_tensors(self, layer_ids):
        return [self.layer_tensors[idx] for idx in layer_ids]

    def get_layer_names(self, layer_ids):
        return [self.layer_names[idx] for idx in layer_ids]

    def get_all_layer_names(self, startswith=None):
        names = [op.name for op in self.graph.get_operations()]
        if startswith is not None:
            names = [name for name in names if name.startswith(startswith)]
        return names

    def create_feed_dict(self, image, expand=True):
        image = np.expand_dims(image, axis=0)
        feed_dict = {self.tensor_name_input_image: image}

        return feed_dict

    def preprocess(self, image, bgr=False):
        bgr_image = image
        if bgr == False:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bgr_image[:,:,0] = bgr_image[:,:,0] - VGG_MEAN[0]
        bgr_image[:,:,1] = bgr_image[:,:,1] - VGG_MEAN[1]
        bgr_image[:,:,2] = bgr_image[:,:,2] - VGG_MEAN[2]

        return image

    def unprocess(self, image):
        unprocessed = image
        #unprocessed[:,:,1] = unprocessed[:,:,0] + VGG_MEAN[0]
        #unprocessed[:,:,1] = unprocessed[:,:,1] + VGG_MEAN[1]
        #unprocessed[:,:,2] = unprocessed[:,:,2] + VGG_MEAN[2]

        return unprocessed

# Test to see if its working
if __name__ == "__main__":
    sess = tf.Session()
    net = VGG19(sess)
    print(net.layer_tensors)
