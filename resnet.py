import numpy as np
import tensorflow as tf
import os

resnet_dir = "resnet/"

class ResNet:
    tensor_name_input_image = "images:0"
    tensor_is_training = "is_training:0"

    # There is no dropout

    def __init__(self, sess, model='ResNet-L152'):
        # Load the meta-data
        print('Loading ' + model + '...')
        net = tf.train.import_meta_graph(resnet_dir + model + '.meta')
        net.restore(sess, resnet_dir + model + '.ckpt')
        print('Finished loading ' + model)

        # Set the graph
        self.graph = sess.graph

        # Get layer names and tensors
        # Only grab b layer from each block since they are 3x3 convolutions
        self.layer_names = [name for name in self.get_all_layer_names() if name.endswith('/b/Conv2D')]
        #self.layer_names = ['scale1/Conv2D'] + self.layer_names
        self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]
        self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

        # For debugging purposes
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print (i)# i.name if you want just a name


        print('Total Number of 3x3 Convolutional Tensors: %d' % len(self.layer_tensors))

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
        is_training = False
        feed_dict = {self.tensor_name_input_image: image,
                     self.tensor_is_training: is_training}

        return feed_dict

# For testing purposes, make sure everything works
if __name__ == "__main__":
    sess = tf.Session()
    net = ResNet(sess)
    print(net.layer_names)
