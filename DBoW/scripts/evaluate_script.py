import sys
import os
import argparse


sys.path.append(os.path.abspath("../"))

from run_discriminative_bow import evaluate_on_dataset
import bow_dataset


parser = argparse.ArgumentParser(description='Evaluate a DBoNW model with '
                                 'given parameters on a data set.')
parser.add_argument('model_filename')
parser.add_argument('data_set_filename',
                    help='Pickle file of BowDataSet.')
parser.add_argument('reference_data_set_filename',
                    help='Pickle file of reference BowDataSet (needed for normalization).')
# TODO: need to make a params file from which to read these params
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


# Load data set from pkl files
data_set =\
        bow_dataset.DBoWDataSet.load_from_pkl(args.data_set_filename)
reference_data_set =\
        bow_dataset.DBoWDataSet.load_from_pkl(args.reference_data_set_filename)

evaluate_on_dataset(args.model_filename, data_set, reference_data_set.max_n_objects,
                    args.n_codewords, args.n_nodes_codeword, args.n_nodes_bow)
