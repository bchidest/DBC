import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, s):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='VALID')


def variable_summaries(var, name, visualize=False):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)
    if visualize:
        var2 = tf.transpose(var, perm=[3, 0, 1, 2])
        for i in range(var2.get_shape()[-1]):
            tf.summary.image(name="first_layer_filters_" + str(i),
                             tensor=tf.expand_dims(var2[..., i], -1),
                             max_outputs=var2.get_shape()[0])


def inference_efficient(x, params, input_size, is_training):
    layer_in = [x]
    layer_ind = 0
    n_filters_previous_layer = 2
    shapes = []
    conv_remainders = params.calculate_conv_remainders(input_size)
    t = []
    for layer in params.conv_layers:
        layer_name = 'conv' + str(layer_ind)
        with tf.name_scope(layer_name):
            weights = weight_variable([layer.filter_size,
                                       layer.filter_size,
                                       n_filters_previous_layer,
                                       layer.n_filters])
            biases = bias_variable([layer.n_filters])
            layer_out = []
            if layer.convolution_stride > 1:
                temp_out = conv2d(layer_in[0], weights, 1) + biases
                #with tf.variable_scope(layer_name + '_bn') as scope:
                #    temp_out = tf.contrib.layers.batch_norm(temp_out,
                #                                    center=True, scale=True,
                #                                    is_training=is_training)
                temp_out = tf.nn.relu(temp_out)
                if conv_remainders[layer_ind] > 0:
                    for s1 in range(layer.convolution_stride):
                        for s2 in range(layer.convolution_stride):
                            layer_out.append(temp_out[:,
                                s1:-conv_remainders[layer_ind]:layer.convolution_stride,
                                s2:-conv_remainders[layer_ind]:layer.convolution_stride, :])
                else:
                    for s1 in range(layer.convolution_stride):
                        for s2 in range(layer.convolution_stride):
                            layer_out.append(temp_out[:,
                                s1::layer.convolution_stride,
                                s2::layer.convolution_stride, :])
            else:
                temp_out = conv2d(layer_in[0], weights, 1) + biases
                #with tf.variable_scope(layer_name + '_bn') as scope:
                #    temp_out = tf.contrib.layers.batch_norm(temp_out,
                #                                    center=True, scale=True,
                #                                    is_training=is_training)
                layer_out.append(tf.nn.relu(temp_out))
                t.append(tf.nn.relu(temp_out))
            for i in range(1, len(layer_in)):
                if layer.convolution_stride > 1:
                    temp_out = conv2d(layer_in[i], weights, 1) + biases
                    #with tf.variable_scope(layer_name + '_bn', reuse=True) as scope:
                    #    temp_out = tf.contrib.layers.batch_norm(temp_out,
                    #                                    center=True, scale=True,
                    #                                    is_training=is_training)
                    temp_out = tf.nn.relu(temp_out)
                    if conv_remainders[layer_ind] > 0:
                        for s1 in range(layer.convolution_stride):
                            for s2 in range(layer.convolution_stride):
                                layer_out.append(temp_out[:,
                                    s1:-conv_remainders[layer_ind]:layer.convolution_stride,
                                    s2:-conv_remainders[layer_ind]:layer.convolution_stride, :])
                    else:
                        for s1 in range(layer.convolution_stride):
                            for s2 in range(layer.convolution_stride):
                                layer_out.append(temp_out[:,
                                    s1::layer.convolution_stride,
                                    s2::layer.convolution_stride, :])
                else:
                    temp_out = conv2d(layer_in[i], weights, 1) + biases
                    #with tf.variable_scope(layer_name + '_bn', reuse=True) as scope:
                    #    temp_out = tf.contrib.layers.batch_norm(temp_out,
                    #                                    center=True, scale=True,
                    #                                    is_training=is_training)
                    layer_out.append(tf.nn.relu(temp_out))
            shapes.append(tf.shape(layer_out[0]))
            layer_in = layer_out
            layer_ind += 1
            n_filters_previous_layer = layer.n_filters

    # Transition layer from conv to fully-connected is special
    layer_in_n_nodes = params.calculate_n_output_conv_nodes(input_size)
    layer = params.fully_connected_layers[0]
    layer_ind = 0
    layer_name = 'full' + str(layer_ind)
    with tf.name_scope(layer_name):
        weights = weight_variable([layer_in_n_nodes, layer.n_nodes])
        # TODO: This is the confusing part... I need to actually check
        #       that the weights are being reshaped correctly. I think
        #       I have the dimensionality of the weight matrix correct,
        #       but are the coefficients being mapped to the correct
        #       place in that matrix by tf.reshape()?
        filter_size = int(np.sqrt(layer_in_n_nodes/n_filters_previous_layer))
        weights = tf.reshape(weights, [filter_size,
                                       filter_size,
                                       n_filters_previous_layer,
                                       layer.n_nodes])
        bias = bias_variable([layer.n_nodes])
        temp = conv2d(layer_in[0], weights, 1)
        temp = tf.reshape(temp, [-1, layer.n_nodes])
        temp = temp + bias
        t.append(temp)
        with tf.variable_scope(layer_name + '_bn') as scope:
            temp2 = tf.contrib.layers.batch_norm(temp[0:10, :],
                                        center=True, scale=True,
                                        is_training=is_training)
        with tf.variable_scope(layer_name + '_bn', reuse=True) as scope:
            temp3 = tf.contrib.layers.batch_norm(temp[0:100, :],
                                        center=True, scale=True,
                                        is_training=is_training)
        with tf.variable_scope(layer_name + '_bn', reuse=True) as scope:
            temp = tf.contrib.layers.batch_norm(temp,
                                        center=True, scale=True,
                                        is_training=is_training)
        layer_in[0] = tf.nn.relu(temp)
        for i in range(1, len(layer_in)):
            temp = conv2d(layer_in[i], weights, 1)
            temp = tf.reshape(temp, [-1, layer.n_nodes])
            temp = temp + bias
            with tf.variable_scope(layer_name + '_bn', reuse=True) as scope:
                temp = tf.contrib.layers.batch_norm(temp,
                                            center=True, scale=True,
                                            is_training=is_training)
            layer_in[i] = tf.nn.relu(temp)
        t.append(layer_in[0])
        layer_ind += 1
        layer_in_n_nodes = layer.n_nodes
    shapes.append(tf.shape(layer_in[0]))

    # Normal fully-connected layers
    for layer in params.fully_connected_layers[1:]:
        layer_name = 'full' + str(layer_ind)
        with tf.name_scope(layer_name):
            weights = weight_variable([layer_in_n_nodes, layer.n_nodes])
            bias = bias_variable([layer.n_nodes])
            temp_out = tf.matmul(layer_in[0], weights) + bias
            with tf.variable_scope(layer_name + '_bn'):
                temp_out = tf.contrib.layers.batch_norm(temp_out,
                                                center=True, scale=True,
                                                is_training=is_training)
            layer_in[0] = tf.nn.relu(temp_out)
            for i in range(1, len(layer_in)):
                temp_out = tf.matmul(layer_in[i], weights) + bias
                with tf.variable_scope(layer_name + '_bn', reuse=True) as scope:
                    temp_out = tf.contrib.layers.batch_norm(temp_out,
                                                    center=True, scale=True,
                                                    is_training=is_training)
                layer_in[i] = tf.nn.relu(temp_out)
            layer_ind += 1
            layer_in_n_nodes = layer.n_nodes
            shapes.append(tf.shape(layer_in[0]))
    layer_name = 'output_layer'
    with tf.name_scope(layer_name):
        weights = weight_variable([layer_in_n_nodes, 1])
        bias = bias_variable([1])
        y = []
        temp_out = tf.matmul(layer_in[0], weights) + bias
        with tf.variable_scope(layer_name + '_bn'):
            temp_out = tf.contrib.layers.batch_norm(temp_out,
                                            center=True, scale=True,
                                            is_training=is_training)
        t.append(temp_out)
        y.append(tf.nn.sigmoid(temp_out))
        for i in range(1, len(layer_in)):
            temp_out = tf.matmul(layer_in[i], weights) + bias
            with tf.variable_scope(layer_name + '_bn', reuse=True) as scope:
                temp_out = tf.contrib.layers.batch_norm(temp_out,
                                                center=True, scale=True,
                                                is_training=is_training)
            y.append(tf.nn.sigmoid(temp_out))
        shapes.append(tf.shape(y[0]))
    return y, t, temp2, temp3


def inference(x, params, input_size, is_training):
    layer_in = x
    layer_ind = 0
    n_filters_previous_layer = 2
    for layer in params.conv_layers:
        layer_name = 'conv' + str(layer_ind)
        with tf.name_scope(layer_name):
            weights = weight_variable([layer.filter_size,
                                       layer.filter_size,
                                       n_filters_previous_layer,
                                       layer.n_filters])
            if n_filters_previous_layer == 2:
                variable_summaries(weights, layer_name + '/weights', True)
            else:
                variable_summaries(weights, layer_name + '/weights')
            biases = bias_variable([layer.n_filters])
            variable_summaries(biases, layer_name + '/biases')
            layer_in = conv2d(layer_in, weights, layer.convolution_stride) + biases
            #with tf.variable_scope(layer_name + '_bn'):
            #    layer_in = tf.contrib.layers.batch_norm(layer_in,
            #                                            center=True, scale=True,
            #                                            is_training=is_training)
            layer_in = tf.nn.relu(layer_in)
            tf.summary.histogram(layer_name + '/activations',
                                 layer_in)
            layer_ind += 1
            n_filters_previous_layer = layer.n_filters

    layer_in = tf.reshape(layer_in, [params.batch_size, -1])
    layer_in_n_nodes = params.calculate_n_output_conv_nodes(input_size)
    layer_ind = 0
    for layer in params.fully_connected_layers:
        layer_name = 'full' + str(layer_ind)
        with tf.name_scope(layer_name):
            weights = weight_variable([layer_in_n_nodes, layer.n_nodes])
            variable_summaries(weights, layer_name + '/weights')
            bias = bias_variable([layer.n_nodes])
            variable_summaries(bias, layer_name + '/biases')
            layer_in = tf.matmul(layer_in, weights) + bias
            b = layer_in
            with tf.variable_scope(layer_name + '_bn'):
                layer_in = tf.contrib.layers.batch_norm(layer_in,
                                                        center=True, scale=True,
                                                        is_training=is_training)
                tf.summary.histogram(layer_name + '_bn',
                                     layer_in)
            relu = tf.nn.relu(layer_in)
            tf.summary.histogram(layer_name + '/activations', relu)
            #layer_in = tf.nn.dropout(relu, params.dropout_prob)
            layer_in = relu
            layer_ind += 1
            layer_in_n_nodes = layer.n_nodes
    layer_name = 'output_layer'
    #a = layer_in[0]
    with tf.name_scope(layer_name):
        #weights = weight_variable([layer_in_n_nodes,
        #                           params.n_classes])
        weights = weight_variable([layer_in_n_nodes,
                                   1])
        variable_summaries(weights, layer_name + '/weights')
        #bias = bias_variable([params.n_classes])
        bias = bias_variable([1])
        variable_summaries(bias, layer_name + '/biases')
        layer_in = tf.matmul(layer_in, weights) + bias
        with tf.variable_scope(layer_name + '_bn'):
            layer_in = tf.contrib.layers.batch_norm(layer_in,
                                                    center=True, scale=True,
                                                    is_training=is_training)
        abda = layer_in
        y = tf.nn.sigmoid(layer_in)
        #y = tf.matmul(layer_in, weights) + bias
        #y = tf.nn.softmax(tf.matmul(layer_in, weights) + bias)[:, 0]
        tf.summary.histogram(layer_name + '/activations', y)
    return y, abda, b


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def loss_sq_diff(logits, regres):
    mean_sq_error = tf.reduce_mean(tf.squared_difference(logits, regres), name='sq_error')
    tf.summary.scalar('mean_sq_error', mean_sq_error)
    tf.add_to_collection('losses', mean_sq_error)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def prob(logits):
    return tf.nn.softmax(logits)


def training(loss, global_step, learning_rate):
    #lr = tf.train.exponential_decay(learning_rate, global_step, 5000, 0.96, staircase=True)
    lr = learning_rate
    tf.summary.scalar('learning_rate', lr)
    # train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    return train_step


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def evaluation_sq_error(logits, labels, regres, n_classes, n_bins, batch_size):
    h = []
    dif = (logits - regres)
    for i in range(n_classes):
        #ind = tf.where(tf.equal(labels, tf.constant(i, dtype=tf.int32, shape=[batch_size])))
        ind = tf.equal(labels, tf.constant(i, dtype=tf.int32))
        dims = ind.get_shape()
        assert dims.dims[0] == batch_size
        #ind = tf.equal(labels, i)
        #ind = labels == i
        h.append(tf.histogram_fixed_width(tf.boolean_mask(dif, ind),
                                        [-1.0, 1.0], n_bins))
        #h.append(tf.histogram_fixed_width(dif[ind],
        #                                [-1, 1], n_bins))
    return h
