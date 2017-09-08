import argparse

from ..run_discriminative_bow import run_training
from .. import bow_dataset


parser = argparse.ArgumentParser(description='Train a DBoNW model with given '
                                 'parameters.')
parser.add_argument('model_prefix')
parser.add_argument('model_dir')
parser.add_argument('summary_dir')
parser.add_argument('train_data_set_filename',
                    help='Pickle file of training BowDataSet.')
parser.add_argument('validation_data_set_filename',
                    help='Pickle file of validation BowDataSet.')
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
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001)
parser.add_argument('--max_steps',
                    type=int,
                    default=5000)
parser.add_argument('--batch_size',
                    type=int,
                    default=30)
args = parser.parse_args()


# Load data sets from pkl files
train_data_set =\
        bow_dataset.DBoWDataSet.load_from_pkl(args.train_data_set_filename)
validation_data_set =\
        bow_dataset.DBoWDataSet.load_from_pkl(args.validation_data_set_filename)

# Train the model
run_training(args.model_prefix, args.model_dir, args.summary_dir,
             train_data_set, validation_data_set, args.n_codewords,
             args.n_nodes_codeword, args.n_nodes_bow, args.learning_rate,
             args.max_steps, args.batch_size)
