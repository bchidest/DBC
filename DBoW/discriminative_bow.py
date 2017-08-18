import tensorflow as tf


def variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)


def codeword_representation(x, ind, n_features, n_codewords, n_nodes):
    n_nodes_previous = n_features
    layer_in = x[ind]
    for i in range(len(n_nodes)):
        weights = tf.get_variable('weights' + str(i), [n_nodes_previous, n_nodes[i]],
                                  initializer=tf.random_normal_initializer())
        variable_summaries(weights, 'weights' + str(i))
        biases = tf.get_variable('biases' + str(i), [n_nodes[i]],
                                 initializer=tf.constant_initializer(0.0))
        variable_summaries(biases, 'biases' + str(i))
        layer_in = tf.nn.relu(tf.matmul(layer_in, weights) + biases)
        tf.summary.histogram('activations' + str(i),
                             layer_in)
        n_nodes_previous = n_nodes[i]
    weights = tf.get_variable('output_weights', [n_nodes_previous, n_codewords],
                              initializer=tf.random_normal_initializer())
    variable_summaries(weights, 'output_weights')
    biases = tf.get_variable('output_biases', [n_codewords],
                             initializer=tf.constant_initializer(0.0))
    variable_summaries(biases, 'output_biases')
    c = tf.nn.softmax(tf.matmul(layer_in, weights) + biases)
    tf.summary.histogram('output_activations', c)
    return c


def bow(c):
    # Assume c is NxC (N=number of nuclei, C=Number of codewords)
    bow = tf.reduce_sum(c, axis=0)
    return bow


def bow_classifier(bow, n_classes, n_codewords, n_nodes):
    weights = tf.get_variable('fc_weights', [n_codewords, n_nodes],
                              initializer=tf.random_normal_initializer())
    variable_summaries(weights, 'fc_weights')
    biases = tf.get_variable('fc_biases', [n_nodes],
                             initializer=tf.constant_initializer(0.0))
    variable_summaries(biases, 'fc_biases')
    post_weights = tf.matmul(bow, weights)
    bow_hidden = tf.nn.relu(tf.matmul(bow, weights) + biases)
    tf.summary.histogram('fc_activations', bow_hidden)

    weights = tf.get_variable('output_weights', [n_nodes, n_classes],
                              initializer=tf.random_normal_initializer())
    variable_summaries(weights, 'output_weights')
    biases = tf.get_variable('output_biases', [n_classes],
                             initializer=tf.constant_initializer(0.0))
    variable_summaries(biases, 'output_biases')
    y = tf.matmul(bow_hidden, weights) + biases
    tf.summary.histogram('output_activations', y)
    return y, bow_hidden, post_weights


def bow_classifier_simple_2_class(bow):
    weights = tf.constant([[0, 1], [0, 0]], dtype=tf.float32, name='weights')
    bias = tf.get_variable('output_biases', [2],
                           initializer=tf.constant_initializer(0.0))
    y_pre_bias = tf.matmul(bow, weights)
    y = tf.matmul(bow, weights) + bias
    return y, y_pre_bias


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(loss, global_step, learning_rate):
    lr = tf.train.exponential_decay(learning_rate, global_step, 400, 0.96, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    return train_step


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def codeword_representation_fake(x, ind, n_features, mu_0, mu_1):
    layer_in = x[ind]
    weights = tf.ones([n_features, 2], dtype=tf.float32, name='weights')
    inverter = tf.constant([[-1, 0], [0, 1]], dtype=tf.float32, name='inverter')
    weights = tf.matmul(weights, inverter)
    bias = tf.constant([n_features*((mu_1 - mu_0)/2 + mu_0), -n_features*((mu_1 - mu_0)/2 + mu_0)],
                       dtype=tf.float32, name='biases')
    c = tf.nn.sigmoid(tf.matmul(layer_in, weights) + bias)
    return c


def bow_classifier_fake(bow):
    weights = tf.constant([[0], [1]], dtype=tf.float32, name='weights')
    bias = tf.constant([-400], dtype=tf.float32, name='biases')
    y = tf.sigmoid(tf.matmul(bow, weights) + bias)
    return y
