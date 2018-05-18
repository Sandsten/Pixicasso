import cv2
import scipy.io as sio
import scipy.misc
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.models import Model
MAX_SIZE = 300
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

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
# vgg init
# get_weigths
# layers - relu conv pool
# const graph
content_img = read_content_image("content-img/lion.jpg")
vg = VGG19(weights='imagenet', include_top=False, pooling='avg')
layer_name = 'block1_conv1'
intermediate_layer_model = Model(inputs=vg.input,
                                 outputs=vg.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(content_img)
print(intermediate_output[0,0,0,:])
sess = tf.InteractiveSession()
model = load_vgg_model(VGG_MODEL, content_img)
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(content_img))
print(sess.run(model['conv1_1'])[0, 0, 0, :])
# compute losses
# content_loss
# style_loss

# init model

# run the model
