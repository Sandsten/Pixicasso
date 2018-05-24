import os
import cv2
import sys
import scipy.io as sio
import scipy.misc
import argparse
import numpy as np
import tensorflow as tf

########## CONSTANTS ##########
MAX_SIZE = 300
ALPHA = 1   # Weight on content loss.
BETA = 100  # Weight on style loss.
GAMMA = 90  # Cross loss weight
ITERATIONS = 200  # Number of iterations to run.

# Path to the deep learning model.
# It can be downloaded from http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

# Mean of the training data used for the VGG. We need to normalize our input images with that values
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


def parse_args():
    parser = argparse.ArgumentParser(description='Style transfer.')
    parser.add_argument('--style_img', type=str,
                        help='Filenames of the style image (example: starry-night.jpg)',
                        default='starry-night.jpg')
    parser.add_argument('--content_img', type=str, default='lion.jpg',
                        help='Filename of the content image (example: lion.jpg)')
    parser.add_argument('--save_intermediate', type=bool, default=False,
                        help="Save intermediate images")
    parser.add_argument('--save_path', type=str, default="result/")

    args = parser.parse_args()
    return args


def read_content_image(path):
    img = scipy.misc.imread(path)

    h, w, d = img.shape
    mx = MAX_SIZE
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)

    # Adding an extra dimension for convnet input
    img = np.reshape(img, ((1,) + img.shape))
    # Input to the VGG model expects the mean to be subtracted.
    img = img - MEAN_VALUES
    return img


def white_noise_start_img(img):

    _, h, w, _ = img.shape

    noise = np.random.rand(1, h, w, 3) * 255
    noise -= MEAN_VALUES

    return noise


def read_style_image(path, content_img):
    img = scipy.misc.imread(path)

    # Resize to have the same dimension as the content image
    _, h, w, _ = content_img.shape
    img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_AREA)
    # Adding an extra dimension for convnet input
    img = np.reshape(img, ((1,) + img.shape))
    # Input to the VGG model expects the mean to be subtracted.
    img = img - MEAN_VALUES
    return img


def save_image(filename, img):
    img = img + MEAN_VALUES
    img = img[0]  # Remove the extra dimension that we added at the beginning
    img = np.clip(img, 0, 255).astype('uint8')
    scipy.misc.imsave(filename, img)


def load_vgg_model(path, input_image):
    vgg = sio.loadmat(path)
    vgg_layers = vgg['layers']

    def get_weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        W = vgg_layers[0][layer][0][0][2][0][0]
        b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]

        assert layer_name == expected_layer_name
        return W, b

    def conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = get_weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.relu(tf.nn.conv2d(
            prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b)

    def avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    _, h, w, d = input_image.shape
    graph['input'] = tf.Variable(np.zeros((1, h, w, d)), dtype='float32')
    graph['conv1_1'] = conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = avgpool(graph['conv1_2'])
    graph['conv2_1'] = conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = avgpool(graph['conv2_2'])
    graph['conv3_1'] = conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = avgpool(graph['conv3_4'])
    graph['conv4_1'] = conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = avgpool(graph['conv4_4'])
    graph['conv5_1'] = conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = avgpool(graph['conv5_4'])
    return graph


def content_loss_func(sess, model):

    # Get the values from said layer of our content image
    p = sess.run(model['conv4_2'])
    # Get the tensor for our generated image at the same layer
    x = model['conv4_2']

    # Using p here since it's a vector already, could probably get the same values from the tensor x too.
    # N is the number of filters (at layer l).
    N = p.shape[3]
    # M is the height times the width of the feature map (at layer l).
    M = p.shape[1] * p.shape[2]

    # Loss function from paper, equation (1)
    # content_loss = 0.5 * tf.reduce_sum(tf.pow(x - p, 2))

    # This normalization constant is supposed to be faster and more commonly
    # used in style loss
    content_loss = (1 / (1 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))

    # Return the tensor for content loss
    return content_loss


# For softer features, increase weight of higher layers
# For sharper features, increase weight of lower layer
# STYLE_LAYERS = [
#     ('conv1_1', 0.5),
#     ('conv2_1', 1.0),
#     ('conv3_1', 1.5),
#     ('conv4_1', 3.0),
#     ('conv5_1', 4.0),
# ]

# Original weights from Gatys
STYLE_LAYERS = [
    ('conv1_1', 1/5),
    ('conv2_1', 1/5),
    ('conv3_1', 1/5),
    ('conv4_1', 1/5),
    ('conv5_1', 1/5),
]


def style_loss_func(sess, model):

    def gram_matrix(F, N, M):
        # The gram matrix, equation (3) in the paper
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    E = []  # E holds all the E_l from the paper, equation (4)
    for layer_name, _ in STYLE_LAYERS:
        # Get the values from said style layer of our content image
        a = sess.run(model[layer_name])
        # Get the tensor for our generated image at the same layer
        x = model[layer_name]

        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image (at layer l).
        A = gram_matrix(a, N, M)
        # G is the style representation of the generated image (at layer l).
        G = gram_matrix(x, N, M)

        # Equation (4)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        E.append(result)

    # Array with all the weights for each layer (how much each layer should contribute)
    W = [w for _, w in STYLE_LAYERS]

    # Array with all the different losses from equation (5), not summed yet
    losses = []
    for l in range(len(STYLE_LAYERS)):
        losses.append(W[l] * E[l])

    # Sum all the losses from equation (5)
    # It doesn't seem to make a difference wether we use sum or tf.reduce_sum
    style_loss = sum(losses)
    # style_loss = tf.reduce_sum(losses)

    return style_loss


def style_transfer(c_path, s_path, iterations):
    content_path = 'content-img/' + c_path + ".jpg"
    content_img = read_content_image(content_path)

    style_path = 'style-img/' + s_path + ".jpg"
    style_img = read_style_image(style_path, content_img)

    input_image = content_img
    # input_image = white_noise_start_img(content_img)

    sess = tf.InteractiveSession()
    model = load_vgg_model(VGG_MODEL, input_image)

    # Construct the style_loss for the content image
    sess.run(model['input'].assign(content_img))
    cross_loss = style_loss_func(sess, model)

    # Construct content_loss for content image
    sess.run(model['input'].assign(content_img))
    content_loss = content_loss_func(sess, model)

    # Construct style_loss using style_image.
    sess.run(model['input'].assign(style_img))
    style_loss = style_loss_func(sess, model)

    # Instantiate equation 7 of the paper.
    # total_loss = ALPHA * content_loss + BETA * style_loss

    # Our own variations of the total_loss function
    total_loss = ALPHA * content_loss + BETA * style_loss - GAMMA * cross_loss

    # Initialize an optimizer that will minimize our loss
    # The content is built from one layer, while the style is from five
    # layers. Then we minimize the total_loss, which is the equation 7.
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(total_loss)

    # The AdamOptimizer adds new variables that has to be initialized
    # Create the opreations for initializing variables
    init_vars = tf.global_variables_initializer()
    # Apply operations to model which applies the variables
    sess.run(init_vars)

    # Do not initialize variable after this line, if we do that
    # the input will be overwritten by the zeros defined when loading the model.
    sess.run(model['input'].assign(input_image))

    # Arrays for storing losses
    total_loss_list = []
    content_loss_list = []
    style_loss_list = []
    cross_loss_list = []
    for it in range(iterations):
        # Perform one epoch of training
        sess.run(train_step)
        total_loss_list.append(sess.run(total_loss))
        content_loss_list.append(sess.run(content_loss))
        style_loss_list.append(sess.run(style_loss))
        cross_loss_list.append(sess.run(cross_loss))
        if it % 100 == 0:
            print(it)
        if it % 100 == 0 and it != 0:
            if args.save_intermediate:
                print("Save!")
                mixed_image = sess.run(model['input'])
                the_path = args.save_path
                filename = the_path + '/' + c_path + \
                    '-' + s_path + '_%d.png' % (it)
                save_image(filename, mixed_image)

    data_list = {"total_loss": total_loss_list, "content_loss": content_loss_list,
                 "style_loss": style_loss_list, "cross_loss": cross_loss_list}

    mixed_image = sess.run(model['input'])
    sess.close()
    return mixed_image, data_list


def compareStyles(c_path, styles_path):
    total_losses = []

    # Create the save path if it doesn't extis
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for s_path in styles_path:
        print(s_path)

        new_image, data_list = style_transfer(
            c_path, s_path, ITERATIONS)

        the_path = args.save_path
        filename = the_path + '/' + c_path + "-" + s_path + '.png'
        save_image(filename, new_image)
        np.savez(the_path + "/data_list_" + s_path, data_list)



########## MAIN FUNCTION ##########
global args
args = parse_args()

# compareStyles("tubingen", ["starry-night", "seated-nude", "kandinsky",
#                            "shipwreck", "the_scream", "woman-with-hat-matisse"])

# compareStyles("tubingen", ["kandinsky"])
# compareStyles("tubingen", ["shipwreck"])
# compareStyles("coast", ["the_scream"])
compareStyles("tubingen", ["seated-nude"])
# compareStyles("coast", ["starry-night"])
# compareStyles("coast", ["woman-with-hat-matisse"])
