import argparse

from dbow.run import run_training
from dbow import dataset

parser = argparse.ArgumentParser(description='Train a DBoNW model with given '
                                 'parameters.')
parser.add_argument('model_dir')
parser.add_argument('summary_dir')
parser.add_argument('train_data_set_filename',
                    help='Pickle file of training BowDataSet.')
parser.add_argument('validation_data_set_filename',
                    help='Pickle file of validation BowDataSet.')
parser.add_argument('param_filename',
                    help='Parameter file for DBoW.')
parser.add_argument('--model_prefix', default='')
args = parser.parse_args()


# Load data sets from pkl files
train_data_set =\
        dataset.DBoWDataSet.load_from_pkl(args.train_data_set_filename)
validation_data_set =\
        dataset.DBoWDataSet.load_from_pkl(args.validation_data_set_filename)

# Train the model
run_training(args.param_filename, args.model_dir, args.summary_dir,
             train_data_set, validation_data_set, args.model_prefix)
