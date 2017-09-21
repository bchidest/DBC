import argparse

from dbow.run import evaluate_on_dataset
from dbow import dataset


parser = argparse.ArgumentParser(description='Evaluate a DBoNW model with '
                                 'given parameters on a data set.')
parser.add_argument('model_filename')
parser.add_argument('param_filename',
                    help='Parameter file for DBoW.')
parser.add_argument('data_set_filename',
                    help='Pickle file of BowDataSet.')
args = parser.parse_args()


# Load data set from pkl files
data_set =\
        dataset.DBoWDataSet.load_from_pkl(args.data_set_filename)
reference_data_set =\
        dataset.DBoWDataSet.load_from_pkl(args.reference_data_set_filename)

evaluate_on_dataset(args.model_filename, data_set, reference_data_set.max_n_objects,
                    args.param_filename)
