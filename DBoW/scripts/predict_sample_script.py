import argparse

from dbow import predictor
from dbow import dataset


parser = argparse.ArgumentParser(description='Evaluate a DBoNW model with '
                                 'given parameters on a data set.')
parser.add_argument('model_filename')
parser.add_argument('data_set_filename',
                    help='Pickle file of BowDataSet.')
parser.add_argument('sample_filename')
parser.add_argument('--features_filename_suffix', default='',
        type=str)
parser.add_argument('--labels_filename', default=[])
# TODO: need to make a params file from which to read these params
parser.add_argument('--max_n_objects',
                    type=int)
parser.add_argument('--n_codewords',
                    type=int,
                    default=8)
parser.add_argument('--n_nodes_codeword',
                    help='List of number of nodes for each hidden layer in '
                    + 'codeword network.',
                    nargs='+',
                    type=int,
                    default=[25, 15])
parser.add_argument('--n_nodes_bow',
                    help='Number of nodes for hidden layer in '
                    + 'BoW network.',
                    type=int,
                    default=6)
args = parser.parse_args()

data_set = dataset.DBoWDataSet.load_from_pkl(args.data_set_filename)
dbow_pred = predictor.DBoWPredictor(args.model_filename, data_set,
                                    args.labels_filename,
                                    features_filename_suffix=args.features_filename_suffix)
y_hat, y = dbow_pred.predict(args.sample_filename)
print('Predicted label = %d' % y_hat)
print('True label = %d' % y)
