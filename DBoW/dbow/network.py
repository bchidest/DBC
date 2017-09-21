import tensorflow as tf


class DBoWParams(object):
    def __init__(self, n_codewords=8, n_nodes_codeword=[25, 15],
                 n_nodes_bow=6, learning_rate=0.001, max_steps=5000,
                 batch_size=30):
        self.n_codewords = n_codewords
        self.n_nodes_codeword = n_nodes_codeword
        self.n_nodes_bow = n_nodes_bow
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.batch_size = batch_size

    @classmethod
    def load_from_file(cls, param_filename):
        '''
        Example format:
        n_codewords
        n_nodes_codeword
        n_nodes_bow
        learning_rate
        max_steps
        batch_size
        '''
        f_param = open(param_filename, 'r')
        n_codewords = int(f_param.readline())
        string_temp = f_param.readline().split(',')
        n_nodes_codeword = [int(s) for s in string_temp]
        n_nodes_bow = int(f_param.readline())
        learning_rate = float(f_param.readline())
        max_steps = int(f_param.readline())
        batch_size = int(f_param.readline())
        dbow_params = cls(n_codewords, n_nodes_codeword, n_nodes_bow,
                          learning_rate, max_steps, batch_size)
        return dbow_params


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
    # Assume c is NxC (N=number of objects, C=Number of codewords)
    bow = tf.reduce_sum(c, axis=0)
    return bow


def bow_classifier(bow, n_classes, n_codewords, n_nodes):
    weights = tf.get_variable('fc_weights', [n_codewords, n_nodes],
                              initializer=tf.random_normal_initializer())
    variable_summaries(weights, 'fc_weights')
    biases = tf.get_variable('fc_biases', [n_nodes],
                             initializer=tf.constant_initializer(0.0))
    variable_summaries(biases, 'fc_biases')
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
    return y


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
