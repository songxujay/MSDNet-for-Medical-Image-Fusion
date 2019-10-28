import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

WEIGHT_INIT_STDDEV = 0.1
DENSE_layers = 3
DECAY = .9
EPSILON = 1e-8

class Encoder(object):
    def __init__(self, model_pre_path):
        self.weight_vars = []
        self.model_pre_path = model_pre_path

        with tf.variable_scope('encoder'):
            self.weight_vars.append(self._create_variables(1, 16, 3, scope='conv1'))

            self.weight_vars.append(self._create_variables(16, 16, 3, scope='dense_block_conv2'))
            self.weight_vars.append(self._create_variables(32, 16, 3, scope='dense_block_conv3'))
            self.weight_vars.append(self._create_variables(48, 16, 3, scope='dense_block_conv4'))

            self.weight_vars.append(self._create_variables(64, 64, 1, scope='conv5_1'))
            self.weight_vars.append(self._create_variables(64, 64, 3, scope='conv5_2'))
            self.weight_vars.append(self._create_variables(64, 64, 5, scope='conv5_3'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        if self.model_pre_path:
            reader = pywrap_tensorflow.NewCheckpointReader(self.model_pre_path)
            with tf.variable_scope(scope):
                kernel = tf.Variable(reader.get_tensor('encoder/' + scope + '/kernel'), name='kernel')
                bias = tf.Variable(reader.get_tensor('encoder/' + scope + '/bias'), name='bias')
        else:
            with tf.variable_scope(scope):
                kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
                bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def encode(self, image):
        dense_indices = (1, 2, 3)

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            if i in dense_indices:
                out = conv2d_dense(out, kernel, bias, use_relu=True)

            elif i == 4:
                out1 = conv2d(out, kernel, bias, use_relu=True)
            elif i == 5:
                out2 = conv2d(out, kernel, bias, use_relu=True)
            elif i == 6:
                out = conv2d_incep(out, out1, out2, kernel, bias, use_relu=True)

            else:
                out = conv2d(out, kernel, bias, use_relu=True)
        return out


def conv2d(x, kernel, bias, use_relu=True):

    if kernel.shape[0] == 5:
        x_padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
    elif kernel.shape[0] == 3:
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    else:
        x_padded = x

    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.relu(out)

    return out


def conv2d_dense(x, kernel, bias, use_relu=True):
    if kernel.shape[0] == 5:
        x_padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
    elif kernel.shape[0] == 3:
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    else:
        x_padded = x

    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.relu(out)
    # concatenate
    out = tf.concat([out, x], 3)

    return out

def conv2d_incep (x, x1, x2, kernel, bias, use_relu=True):
    if kernel.shape[0] == 5:
        x_padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
    elif kernel.shape[0] == 3:
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    else:
        x_padded = x

    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    out = tf.nn.relu(out)
    # concatenate
    temp = tf.concat([x1, x2], 3)
    out = tf.concat([temp, out], 3)

    return out


def transition_block(x, kernel, bias):

    num_maps = x.shape[3]
    out = __batch_normalize(x, num_maps)
    out = tf.nn.relu(out)
    out = conv2d(out, kernel, bias, use_relu=False)

    return out


def __batch_normalize(inputs, num_maps, is_training=True):
    # Trainable variables for scaling and offsetting our inputs
    # scale = tf.Variable(tf.ones([num_maps], dtype=tf.float32))
    # offset = tf.Variable(tf.zeros([num_maps], dtype=tf.float32))

    # Mean and variances related to our current batch
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

    # # Create an optimizer to maintain a 'moving average'
    # ema = tf.train.ExponentialMovingAverage(decay=DECAY)
    #
    # def ema_retrieve():
    #     return ema.average(batch_mean), ema.average(batch_var)
    #
    # # If the net is being trained, update the average every training step
    # def ema_update():
    #     ema_apply = ema.apply([batch_mean, batch_var])
    #
    #     # Make sure to compute the new means and variances prior to returning their values
    #     with tf.control_dependencies([ema_apply]):
    #         return tf.identity(batch_mean), tf.identity(batch_var)
    #
    # # Retrieve the means and variances and apply the BN transformation
    # mean, var = tf.cond(tf.equal(is_training, True), ema_update, ema_retrieve)
    bn_inputs = tf.nn.batch_normalization(inputs, batch_mean, batch_var, None, None, EPSILON)

    return bn_inputs