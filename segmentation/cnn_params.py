import math
import csv


class Params:

    def __init__(self, learning_rate, conv_layers,
                 fully_connected_layers, batch_size,
                 dropout_prob, n_classes, max_steps,
                 param_file_line_number):
        """
        Order:
            learning_rate - float
            layer_list - list
                layer_type - {"fully", "conv"}
                n_nodes - int


        """
        self.learning_rate = learning_rate
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.n_classes = n_classes
        self.max_steps = max_steps
        self.param_file_line_number = param_file_line_number

    def calculate_conv_output_sizes(self, input_size):
        sizes = []
        for layer in self.conv_layers:
            input_size = layer.calculate_n_output_size(input_size)
            sizes.append(input_size)
        return sizes

    def calculate_conv_remainders(self, input_size):
        # TODO: This is hardoded for a stride of 2 only!
        remainders = []
        for layer in self.conv_layers:
            if (input_size % 2 == 0) and (layer.convolution_stride == 2):
                remainders.append(1)
            else:
                remainders.append(0)
            input_size = layer.calculate_n_output_size(input_size)
        return remainders

    def calculate_n_output_conv_nodes(self, input_size):
        for layer in self.conv_layers:
            input_size = layer.calculate_n_output_size(input_size)
        return input_size**2 * layer.n_filters


class FullyConnectedLayer:

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes


class ConvolutionalLayer:
    def __init__(self, n_filters, filter_size, pooling_support,
                 convolution_stride):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pooling_support = pooling_support
        self.convolution_stride = convolution_stride

    def calculate_n_output_size(self, input_size):
        after_conv_size = int(math.ceil((input_size - self.filter_size + 1) /
                              float(self.convolution_stride)))
        return after_conv_size


def load_params(param_filename):
    param_file = open(param_filename, "r")
    param_csv = csv.reader(param_file, delimiter=",")
    param = []
    position = "learning_rate"
    conv_layers = []
    full_layers = []
    line_number = 0
    for line in param_csv:
        line_number = line_number + 1
        if position == "learning_rate":
            param_file_line_number = line_number
            learning_rate = float(line[0])
            position = "n_classes"
            continue
        elif position == "n_classes":
            n_classes = int(line[0])
            position = "batch_size"
            continue
        elif position == "batch_size":
            batch_size = int(line[0])
            position = "dropout_prob"
            continue
        elif position == "dropout_prob":
            dropout_prob = float(line[0])
            position = "max_steps"
            continue
        elif position == "max_steps":
            max_steps = int(line[0])
            position = "layer"
        elif position == "layer":
            if not line:
                param.append(Params(learning_rate, conv_layers,
                                    full_layers, batch_size, dropout_prob,
                                    n_classes, max_steps,
                                    param_file_line_number))
                conv_layers = []
                full_layers = []
                position = "learning_rate"
                continue
            if line[0] == "conv":
                conv_layers.append(ConvolutionalLayer(int(line[1]),
                                                      int(line[2]),
                                                      int(line[3]),
                                                      int(line[4])))
            elif line[0] == "full":
                full_layers.append(FullyConnectedLayer(int(line[1])))

    param.append(Params(learning_rate, conv_layers,
                        full_layers, batch_size, dropout_prob,
                        n_classes, max_steps, param_file_line_number))

    return param

