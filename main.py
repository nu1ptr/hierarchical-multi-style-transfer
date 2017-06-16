##################################################################
# CODE BASED OFF AND HEAVILY MODIFIED FROM SIRAJ RAVAL'S TUTORIAL
##################################################################
import tensorflow as tf
import progressbar
import numpy as np
import sys
import cv2
import time
import glob
import os
import utils
from functools import reduce

# Different Network Architectures
import vgg16
import vgg19
import alexnet
import resnet

# Tensorflow flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('contents', '../data/content/', 'Content Image Directory')
flags.DEFINE_string('styles', '../data/styles/', 'Style Image Directory')
flags.DEFINE_integer('iterations', 100, 'Number iterations')
flags.DEFINE_integer('resize', -1, 'Resize to according to height')
flags.DEFINE_float('weight_denoise', 0.3, 'Weight Denoise')
flags.DEFINE_float('step_size', 10.0, 'Step size of gradient')
flags.DEFINE_string('model', 'VGG19', 'Models: VGG16, AlexNet, ResNet-L152, ResNet-L101, ResNet-L50')

# Get directories of our nets
vgg16.data_dir = 'vgg16/'
vgg19.vgg19_dir = 'vgg19/'
resnet.resnet_dir = 'resnet/'
alexnet.alexnet_dir = 'alexnet/'

# Calculate the mean squared error between content
def mean_squared_error(a,b):
    return tf.reduce_mean(tf.square(a-b))

# Our content loss for some layer in VGG
def create_content_loss(session, model, content_image, layer_ids):
    # Feed our content image as the imput
    feed_dict = model.create_feed_dict(image=model.preprocess(content_image))

    # The layers that we want to run our loss computation on
    layers = model.get_layer_tensors(layer_ids)

    # Values of each of these layers
    values = session.run(layers, feed_dict=feed_dict)

    # Calculate the loss
    with model.graph.as_default():
        layer_losses = []

        for value, layer in zip(values, layers):
            # You're getting the values when you run the content image through here
            value_const = tf.constant(value)

            # Calculate the loss between your value and layer
            # Compare between two feature maps
            loss = mean_squared_error(layer, value_const)
            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

# Tensor is a convolutional layer with 4 dimensions
def gram_matrix(tensor):
    # 4D Tensor W,H,C,N
    shape = tensor.get_shape()

    num_channel = int(shape[3])

    # Flatten
    matrix = tf.reshape(tensor, shape=[-1, num_channel])
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

"""
def create_style_loss(session, model, style_image, layer_ids):
    # Create a feed-dict with the style-image.
    feed_dict = model.create_feed_dict(image=style_image)

    layers = model.get_layer_tensors(layer_ids)
    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]

        values = session.run(gram_layers, feed_dict=feed_dict)

        # Initialize an empty list of loss-functions.
        layer_losses = []

        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(gram_layer, value_const)
            layer_losses.append(loss)
        total_loss = tf.reduce_mean(layer_losses)

    return total_loss
"""

# Style loss
def create_style_loss(session, model, style_image, layer_ids, style_mask=None):
    # define a placeholder within our model
    feed_dict = model.create_feed_dict(image=model.preprocess(style_image))

    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        # Gram Matrix captures second order statistics, toss away everything that is unnecessary
        # and keep style

        # Calculate the values for all layers that we want to get the style of by taking taking the gram matrix
        values = session.run(layers, feed_dict=feed_dict)

        layer_losses = []

        for value, layer in zip(values, layers):
            value = tf.convert_to_tensor(value)

            # Create a tensor mask
            if style_mask is not None:
                _, h,w,d = value.get_shape()
                mask = cv2.resize(style_mask, (w,h), interpolation=cv2.INTER_AREA)
                mask = tf.convert_to_tensor(mask)
                tensors = []
                for _ in range(d.value):
                    tensors.append(mask)
                mask = tf.stack(tensors, axis=2)
                mask = tf.stack(mask, axis=0)
                mask = tf.expand_dims(mask, 0)
                masked_value = tf.multiply(value, mask)
                masked_layer = tf.multiply(layer, mask)
                gram_layer = gram_matrix(masked_layer)
            else:
                gram_layer = gram_matrix(layer)

            loss = mean_squared_error(gram_layer, gram_matrix(value))
            layer_losses.append(loss)

        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

# Denoise Loss
# Kind of acts like a regularizer
# Shift pixels and minimize loss over their differences
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
    return loss

def transform_network(session, model, content_image):
    return

def spatial_multi_style_transfer(session, model, content_images, style_images, content_layer_ids, style_layer_ids, weight_contents,
                        weight_styles, masks=None, weight_denoise=0.3, num_iterations=100, step_size=9.0, style_step_size=10.5):
    assert len(style_images) == len(style_layer_ids), "Mismatch between number of style images and number of subset style layers"

    # Print Layers
    for i, layers in enumerate(content_layer_ids):
        print("Content %d layers:" % i)
        print(model.get_layer_names(layers))

    for i, layers in enumerate(style_layer_ids):
        print("Style %d layers:" % i)
        print(model.get_layer_names(layers))

    # Loss functions
    loss_contents = [ create_content_loss(session=session, model=model, content_image=im, layer_ids=lays)
                    for im, lays in zip(content_images, content_layer_ids) ]

    # Multi Style Loss
    if masks != None:
        loss_styles = [ create_style_loss(session=session, model=model, style_image=im, layer_ids=lays, style_mask=mask)
                        for im, lays, mask in zip(style_images, style_layer_ids, masks) ]
    else:
        loss_styles = [ create_style_loss(session=session, model=model, style_image=im, layer_ids=lays, style_mask=None)
                        for im, lays in zip(style_images, style_layer_ids) ]

    # Denoising Loss
    loss_denoise = create_denoise_loss(model)

    # Adjustment variables
    with model.graph.as_default():
        adj_contents = [tf.Variable(1e-10, name='adj_content' + str(i)) for i in range(len(content_images))]
        adj_styles = [tf.Variable(1e-10, name='adj_style' + str(i)) for i in range(len(style_images))]
        adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Initialize the adjustments
    session.run([adj_denoise.initializer] + \
                [adj_con.initializer for adj_con in adj_contents] + \
                [adj_sty.initializer for adj_sty in adj_styles])

    # Avoid division by zero and get inverse of loss
    update_adj_contents = [ adj_content.assign(1.0 / (loss_content + 1e-10)) for adj_content, loss_content in zip(adj_contents, loss_contents) ]
    update_adj_styles = [ adj_style.assign(1.0 / (loss_style + 1e-10)) for adj_style, loss_style in zip(adj_styles, loss_styles) ]
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    # Combine content losses
    f = lambda a,b: a + b
    loss_contents_combine = reduce(f,[ w*a*l for w,a,l in zip(weight_contents, adj_contents, loss_contents)])
    loss_style_combine = reduce(f,[w*a*l for w,a,l in zip(weight_styles, adj_styles, loss_styles)])

    # Combine all the losses togeter
    loss_combine =  loss_contents_combine + loss_style_combine + weight_denoise * adj_denoise * loss_denoise

    # Minimize loss with random noise image
    gradient = tf.gradients(loss_combine, model.input)

    #run_list = [gradient] + style_gradients + [update_adj_denoise] + update_adj_contents + update_adj_styles
    run_list = [gradient] + [update_adj_denoise] + update_adj_contents + update_adj_styles

    # Initialize a mixed image
    mixed_image = np.random.normal(size=content_images[0].shape, scale=np.std(content_images[0]) * 0.1)
    #mixed_image = np.zeros(content_images[0].shape)

    # Iterate over this mixed image
    start = time.time()
    bar = progressbar.ProgressBar()

    # Pretty much own gradient descent
    for i in bar(range(num_iterations)):
        feed_dict = model.create_feed_dict(image=mixed_image)

        # Run the gradients
        update_values = session.run(run_list, feed_dict=feed_dict)

        # Get update values
        grad = update_values[0]
        adj_denoise_val = update_values[1]
        adj_content_val = update_values[2]
        adj_style_vals = update_values[3:]

        grad = np.squeeze(grad)

        # Masking Layers for each style

        # Learning rate
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update our mixed image
        mixed_image -= (grad)* step_size_scaled

        mixed_image = np.clip(mixed_image, 0.0, 500.0)
    end = time.time()

    # Post processing
    """
    mixed_image = model.unprocess(mixed_image)
    min_b = np.min(mixed_image[:,:,0])
    min_g = np.min(mixed_image[:,:,1])
    min_r = np.min(mixed_image[:,:,2])
    mixed_image[:,:,0] += np.abs(min_b)
    mixed_image[:,:,1] += np.abs(min_g)
    mixed_image[:,:,2] += np.abs(min_r)
    """
    max_b = np.max(mixed_image[:,:,0])
    max_g = np.max(mixed_image[:,:,1])
    max_r = np.max(mixed_image[:,:,2])
    mixed_image[:,:,0] = mixed_image[:,:,0] / max_b
    mixed_image[:,:,1] = mixed_image[:,:,1] / max_g
    mixed_image[:,:,2] = mixed_image[:,:,2] / max_r
    mixed_image *= 255.0
    mixed_image = cv2.cvtColor(mixed_image.astype(np.float32), cv2.COLOR_BGR2YUV)
    mixed_image[:,:,0] = cv2.equalizeHist(mixed_image[:,:,0].astype(np.uint8))
    mixed_image= cv2.cvtColor(mixed_image.astype(np.float32), cv2.COLOR_YUV2BGR)

    print("Computation time: %f" % (end - start))
    return mixed_image

def multi_style_transfer(session, model, content_images, style_images, content_layer_ids, style_layer_ids, weight_contents,
                        weight_styles, weight_denoise=0.3, num_iterations=100, step_size=10.0):
    assert len(style_images) == len(style_layer_ids), "Mismatch between number of style images and number of subset style layers"

    # Print Layers
    for i, layers in enumerate(content_layer_ids):
        print("Content %d layers:" % i)
        print(model.get_layer_names(layers))

    for i, layers in enumerate(style_layer_ids):
        print("Style %d layers:" % i)
        print(model.get_layer_names(layers))

    # Loss functions
    loss_contents = [ create_content_loss(session=session, model=model, content_image=im, layer_ids=lays)
                    for im, lays in zip(content_images, content_layer_ids) ]

    # Multi Style Loss
    loss_styles = [ create_style_loss(session=session, model=model, style_image=im, layer_ids=lays)
                    for im, lays in zip(style_images, style_layer_ids) ]

    # Denoising Loss
    loss_denoise = create_denoise_loss(model)

    # Adjustment variables
    with model.graph.as_default():
        adj_contents = [tf.Variable(1e-10, name='adj_content' + str(i)) for i in range(len(content_images))]
        adj_styles = [tf.Variable(1e-10, name='adj_style' + str(i)) for i in range(len(style_images))]
        adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Initialize the adjustments
    session.run([adj_denoise.initializer] + \
                [adj_con.initializer for adj_con in adj_contents] + \
                [adj_sty.initializer for adj_sty in adj_styles])

    # Avoid division by zero and get inverse of loss
    update_adj_contents = [ adj_content.assign(1.0 / (loss_content + 1e-10)) for adj_content, loss_content in zip(adj_contents, loss_contents) ]
    update_adj_styles = [ adj_style.assign(1.0 / (loss_style + 1e-10)) for adj_style, loss_style in zip(adj_styles, loss_styles) ]
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    # Combine style losses
    f = lambda a,b: a + b
    loss_styles_combine = reduce(f,[ w*a*l for w,a,l in zip(weight_styles, adj_styles, loss_styles)])

    # Combine content losses
    loss_contents_combine = reduce(f,[ w*a*l for w,a,l in zip(weight_contents, adj_contents, loss_contents)])

    # Combine all the losses together
    loss_combine =  loss_contents_combine + loss_styles_combine + \
                    weight_denoise * adj_denoise * loss_denoise

    # Minimize loss with random noise image
    gradient = tf.gradients(loss_combine, model.input)

    run_list = [gradient, update_adj_denoise] + update_adj_contents + update_adj_styles

    # Initialize a mixed image
    mixed_image = np.random.normal(size=content_images[0].shape, scale=np.std(content_images[0]) * 0.1)

    # Iterate over this mixed image
    start = time.time()
    bar = progressbar.ProgressBar()

    # Pretty much own gradient descent
    for i in bar(range(num_iterations)):
        feed_dict = model.create_feed_dict(image=mixed_image)

        # Get update values
        update_values = session.run(run_list, feed_dict=feed_dict)
        grad = update_values[0]
        adj_denoise_val = update_values[1]
        adj_content_val = update_values[2:2+len(content_images)]
        adj_style_vals = update_values[2+len(content_images):]

        grad = np.squeeze(grad)

        # Learning rate
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update our mixed image
        mixed_image -= grad * step_size_scaled

        mixed_image = np.clip(mixed_image, 0.0, 255.0)
    end = time.time()

    print("Computation time: %f" % (end - start))
    return model.unprocess(mixed_image)

def style_transfer(session, model, content_image, style_image, content_layer_ids, style_layer_ids,
                    weight_content=1.5, weight_style=10.0, weight_denoise=0.3,
                    num_iterations=100, step_size=10.0):

    # Print Layers
    print("Content Layers:")
    print(model.get_layer_names(content_layer_ids))
    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))

    # Loss functions
    loss_content = create_content_loss(session=session,model=model, content_image=content_image,
                                        layer_ids=content_layer_ids)
    loss_style = create_style_loss(session=session, model=model, style_image=style_image,
                                        layer_ids=style_layer_ids)
    loss_denoise = create_denoise_loss(model)

    # Adjustment variables
    with model.graph.as_default():
        adj_content = tf.Variable(1e-10, name='adj_content')
        adj_style = tf.Variable(1e-10, name='adj_style')
        adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Initialize the adjustments
    session.run([adj_content.initializer, adj_style.initializer, adj_denoise.initializer])

    # Avoid division by zero and get inverse of loss
    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    # Combine all the losses together
    loss_combine =  weight_content * adj_content * loss_content + \
                    weight_style * adj_style * loss_style + \
                    weight_denoise * adj_denoise * loss_denoise

    # Minimize loss with random noise image
    gradient = tf.gradients(loss_combine, model.input)

    run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

    # Initialize a mixed image
    mixed_image = np.random.rand(*content_image.shape) + 128

    # Iterate over this mixed image
    start = time.time()
    bar = progressbar.ProgressBar()
    for i in bar(range(num_iterations)):
        feed_dict = model.create_feed_dict(image=mixed_image)

        # Get update values
        grad, adj_content_val, adj_style_val, adj_denoise_val = session.run(run_list, feed_dict=feed_dict)
        grad = np.squeeze(grad)

        # Learning rate
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update our mixed image
        mixed_image -= grad * step_size_scaled
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

    end = time.time()

    print("Computation time: %f" % (end - start))

    # Post processing
    """
    mixed_image = model.unprocess(mixed_image)
    min_b = np.min(mixed_image[:,:,0])
    min_g = np.min(mixed_image[:,:,1])
    min_r = np.min(mixed_image[:,:,2])
    mixed_image[:,:,0] += np.abs(min_b)
    mixed_image[:,:,1] += np.abs(min_g)
    mixed_image[:,:,2] += np.abs(min_r)
    max_b = np.max(mixed_image[:,:,0])
    max_g = np.max(mixed_image[:,:,1])
    max_r = np.max(mixed_image[:,:,2])
    mixed_image[:,:,0] = mixed_image[:,:,0] / max_b
    mixed_image[:,:,1] = mixed_image[:,:,1] / max_g
    mixed_image[:,:,2] = mixed_image[:,:,2] / max_r
    mixed_image *= 255.0
    """
    mixed_image = cv2.cvtColor(mixed_image.astype(np.float32), cv2.COLOR_BGR2YUV)
    mixed_image[:,:,0] = cv2.equalizeHist(mixed_image[:,:,0].astype(np.uint8))
    mixed_image= cv2.cvtColor(mixed_image.astype(np.float32), cv2.COLOR_YUV2BGR)

    return mixed_image

# Run Style Transfer
if __name__ == '__main__':
    ##################################################
    # Define parameters for style transfer and images
    ##################################################
    # Image pre-processing
    contents = utils.dir_2_dict(FLAGS.contents)
    styles = utils.dir_2_dict(FLAGS.styles)

    for i, k in enumerate(contents):
        print('Content %d loaded: %s' % (i, k))
    for i, k in enumerate(styles):
        print('Style %d loaded: %s' % (i, k))

    # Define your loss layer locations and styles
    # Messing around here
    content_layers = [[12]]# 0,2,4,8,12
    multi_content = [contents["rick-and-morty"]]
    weight_contents = [10.0]

    style_layers = [ list(range(4)), list(range(4,12)), list(range(12,15))]
    multi_style = [styles["asheville"], styles["spaghetti"], styles["polygon"]]
    weight_styles = [10.0,10.0, 10.0]

    # Resize to first content shape, also resizes style
    if FLAGS.resize > 0:
        ratio = float(multi_content[0].shape[1]) / multi_content[0].shape[0]
        multi_content[0] = cv2.resize(multi_content[0], (int(FLAGS.resize*ratio), FLAGS.resize))
        for i in range(len(multi_content[1:])):
            multi_content[i+1] = cv2.resize(multi_content[i+1], (multi_content[0].shape[1], multi_content[0].shape[0]))

        # This may be an issue with scaling...
        multi_style = [cv2.resize(style, (multi_content[0].shape[1], (multi_content[0].shape[0]))) for style in multi_style]
    else:
        multi_style = [cv2.resize(style, (multi_content[0].shape[1], (multi_content[0].shape[0]))) for style in multi_style]


    """
    mask2 = np.zeros((multi_content[0].shape[0], multi_content[0].shape[1])).astype(np.float32)
    mask2[:,int(mask2.shape[1]/2):] = 1.0
    mask1 = np.zeros((multi_content[0].shape[0], multi_content[0].shape[1])).astype(np.float32)
    mask1[:,:int(mask1.shape[1]/2 - 1)] = 1.0
    """

    ###########################################
    # Instantiate the models in a session
    ###########################################
    if FLAGS.model == 'VGG16':
        model = vgg16.VGG16()
        sess = tf.Session(graph=model.graph)
    elif FLAGS.model == 'VGG19':
        sess = tf.Session()
        model = vgg19.VGG19(sess)
    elif 'ResNet' in FLAGS.model:
        sess = tf.Session()
        model = resnet.ResNet(sess, model=FLAGS.model)
    elif 'AlexNet' == FLAGS.model:
        # WARNING ONLY USE SINGLE STYLE FOR THIS
        sess = tf.Session()
        model = alexnet.AlexNet(sess)
        print('This needs to be changed in code... Don\'t use this for now.')
        # Image preprocessing for AlexNet
        # If possible, try to make AlexNet FC
        content = cv2.resize(content, (227,227)).astype(np.float32)
        content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
        style = cv2.resize(style, (227,227)).astype(np.float32)
        style = cv2.cvtColor(style, cv2.COLOR_RGB2BGR)

    ################################
    # Run style transfer
    ################################
    """
    mixed = multi_style_transfer(sess,  model, multi_content, multi_style, content_layers, style_layers, weight_contents, weight_styles,
                            weight_denoise= FLAGS.weight_denoise,
                            num_iterations= FLAGS.iterations)
    """
    # Input must be RGB images, Output is BGR
    mixed = spatial_multi_style_transfer(sess,  model, multi_content, multi_style, content_layers, style_layers, weight_contents, weight_styles,
                            weight_denoise= FLAGS.weight_denoise,
                            num_iterations= FLAGS.iterations,
                            step_size=      FLAGS.step_size)
    """
    mixed = style_transfer(sess, model, multi_content[0], multi_style[0],content_layers[0], style_layers[0],
                    weight_content=1.5, weight_style=10.0, weight_denoise=0.3,
                    num_iterations=500, step_size=10.0)
    """

    ################################################
    # Display results, make sure in BGR for cv2
    ################################################
    # Post Processing
    #mixed = cv2.bilateralFilter(mixed.astype(np.float32),9,75,75)
    merged = np.hstack(multi_content + multi_style + [mixed])
    file_name =  './images/results/' + str(np.random.randint(3,10000000)) + '.png'
    cv2.imwrite(file_name, cv2.cvtColor(merged.astype(np.float32), cv2.COLOR_BGR2RGB))
    print('Saved to ' + file_name)
