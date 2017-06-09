import tensorflow as tf
import numpy as np
import vgg16 as net
import sys
import cv2

# Get directory of our vgg net
net.data_dir = 'vgg16/'

# Calculate the mean squared error between content
def mean_squared_error(a,b):
    return tf.reduce_mean(tf.square(a-b))

# Our content loss for some layer in VGG
def create_content_loss(session, model, content_image, layer_ids):
    # Feed our content image as the imput
    feed_dict = model.create_feed_dict(image=content_image)

    # The layers that we want to run our loss computation on
    layers = model.get_layer_tensors(layer_ids)

    # Values of each of these layers
    values = session.run(layers, feed_dict=feed_dict)

    # Calculate the loss
    with model.graph.as_default():
        layer_losses = []

        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            value = mean_squared_error(layer, value_const)

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

# Style loss
def create_style_loss(session, model, style_image, layer_ids):
    # define a placeholder within our model
    feed_dict = model.create_feed_dict(image=style_image)

    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        # Gram Matrix captures second order statistics, toss away everything that is unnecessary
        # and keep style
        gram_layers = [gram_matrix(layer) for layer in layers]

        # Calculate the values for all layers that we want to get the style of by taking taking the gram matrix
        values = session.run(gram_layers, feed_dict=feed_dict)

        layer_losses = []

        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(gram_layer, value_const)
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

def style_transfer(content_image, style_image, content_layer_ids, style_layer_ids,
                    weight_content=1.5, weight_style=10.0, weight_denoise=0.3,
                    num_iterations=100, step_size=10.0):
    # Instantiate VGG 16
    model = net.VGG16()
    session = tf.InteractiveSession(graph=model.graph)

    print("Content Layers:")
    print(model.get_layer_names(content_layer_ids))
    print()

    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))
    print()

    # Loss functions
    loss_content = create_content_loss(session=session,model=model, content_image=content_image,
                                        layer_ids=content_layer_ids)
    loss_style = create_style_loss(session=session, model=model, style_image=style_image,
                                        layer_ids=style_layer_ids)
    loss_denoise = create_denoise_loss(model)

    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Initialize the adjustments
    session.run([adj_content.initializer, adj_style.initializer, adj_denoise.initializer])

    # Avoid division by zero and get inverse of loss
    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style+ 1e-10))
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
    for i in range(num_iterations):
        feed_dict = model.create_feed_dict(image=mixed_image)

        # Get values
        grad, adj_content_val, adj_style_val, adj_denoise_val = session.run(run_list, feed_dict=feed_dict)
        grad = np.squeeze(grad)

        # Learning rate
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update our mixed image
        mixed_image -= grad * step_size_scaled

        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        # Display
        if(i % 10 == 0) or (i == num_iterations - 1):
            #cv2.imshow('Intermediate', mixed_image)
            print(i)

    # Close session
    session.close()

    return mixed_image

# Run Style Transfer
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('INVALID INPUT: ./' + sys.argv[0] + ' <content_dir> <style_dir>\n')

    content_filename = sys.argv[1]
    style_filename = sys.argv[2]

    content = cv2.cvtColor(cv2.imread(content_filename), cv2.COLOR_BGR2RGB)
    style = cv2.cvtColor(cv2.imread(style_filename),cv2.COLOR_BGR2RGB)

    # Display content
    #cv2.imshow('a', cv2.cvtColor(content, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(0)

    mixed = style_transfer(content, style, [4], list(range(13)),num_iterations=100)
    cv2.imshow('Mixed', cv2.cvtColor(mixed.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
