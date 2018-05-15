import os
import cv2
import sys
import scipy.io as sio
import scipy.misc
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from skimage.transform import resize

########## CONSTANTS ########## 
MAX_SIZE = 300
ALPHA = 100 # Weight on style loss.
BETA = 5 # Weight on content loss.
ITERATIONS = 500 # Number of iterations to run.

# Path to the deep learning model.
# It can be downloaded from http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

# Mean of the training data used for the VGG. We need to normalize our input images with that values
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


def parse_args():
	parser = argparse.ArgumentParser(description='Style transfer.')
	parser.add_argument('--style_img', type=str, 
		help='Filenames of the style image (example: starry-night.jpg)', 
    	default='starry-night.jpg')
	parser.add_argument('--content_img', type=str, default='lion.jpg',
    	help='Filename of the content image (example: lion.jpg)')
	args = parser.parse_args()
	return args

#def checkImage(img, path):
#	if img is None: sys.exit("Error: Invalid path {}\nExiting the program.\n".format(path))

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
	img = img[0] # Remove the extra dimension that we added at the beginning
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
    _, h, w, d     = input_image.shape
    graph['input']   = tf.Variable(np.zeros((1, h, w, d)), dtype = 'float32')
    graph['conv1_1']  = conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = avgpool(graph['conv1_2'])
    graph['conv2_1']  = conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = avgpool(graph['conv2_2'])
    graph['conv3_1']  = conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = avgpool(graph['conv3_4'])
    graph['conv4_1']  = conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = avgpool(graph['conv4_4'])
    graph['conv5_1']  = conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = avgpool(graph['conv5_4'])
    return graph


def content_loss_func(sess, model):
    """
    Content loss function as defined in the paper.
    """
    def _content_loss(p, x):
        # N is the number of filters (at layer l).
        N = p.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = p.shape[1] * p.shape[2]
        # Interestingly, the paper uses this form instead:
        #
        #   0.5 * tf.reduce_sum(tf.pow(x - p, 2)) 
        #
        # But this form is very slow in "painting" and thus could be missing
        # out some constants (from what I see in other source code), so I'll
        # replicate the same normalization constant as used in style loss.
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

# Layers to use. We will use these layers as advised in the paper.
# To have softer features, increase the weight of the higher layers
# (conv5_1) and decrease the weight of the lower layers (conv1_1).
# To have harder features, decrease the weight of the higher layers
# (conv5_1) and increase the weight of the lower layers (conv1_1).
STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.0),
    ('conv3_1', 1.5),
    ('conv4_1', 3.0),
    ('conv5_1', 4.0),
]

def style_loss_func(sess, model):
    """
    Style loss function as defined in the paper.
    """
    def _gram_matrix(F, N, M):
        """
        The gram matrix G.
        """
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        """
        The style loss calculation.
        """
        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image (at layer l).
        A = _gram_matrix(a, N, M)
        # G is the style representation of the generated image (at layer l).
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    loss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
    return loss



########## MAIN FUNCTION ##########
global args
args = parse_args()

#Read Images
content_path = 'content-img/' + args.content_img
content_img = read_content_image(content_path)

style_path = 'style-img/' + args.style_img
style_img = read_style_image(style_path, content_img)

#input_image = generate_noise_image(content_img)
input_image = content_img

sess = tf.InteractiveSession()
model = load_vgg_model(VGG_MODEL, input_image)

#Display Image (this is just for testing)
#cv2.imshow('image',style_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Do the magic
#sess.run(tf.global_variables_initializer())
# Construct content_loss using content_image.
sess.run(model['input'].assign(content_img))
content_loss = content_loss_func(sess, model)
# Construct style_loss using style_image.
sess.run(model['input'].assign(style_img))
style_loss = style_loss_func(sess, model)
# Instantiate equation 7 of the paper.
total_loss = BETA * content_loss + ALPHA * style_loss

# From the paper: jointly minimize the distance of a white noise image
# from the content representation of the photograph in one layer of
# the neywork and the style representation of the painting in a number
# of layers of the CNN.
#
# The content is built from one layer, while the style is from five
# layers. Then we minimize the total_loss, which is the equation 7.
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(total_loss)

sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(input_image))
for it in range(ITERATIONS):
    sess.run(train_step)
    if it%10 == 0:
        # Print every 10 iteration.
        mixed_image = sess.run(model['input'])
        print('Iteration %d' % (it))
        print('sum : ', sess.run(tf.reduce_sum(mixed_image)))
        print('cost: ', sess.run(total_loss))
		#Save result
        if not os.path.exists('result/'):
            os.mkdir('result/')

        filename = 'result/%d.png' % (it)
        save_image(filename, mixed_image)
