import argparse

from dbow import predictor
from dbow import dataset


parser = argparse.ArgumentParser(description='Evaluate a DBoNW model with '
                                 'given parameters on a data set.')
parser.add_argument('model_filename')
parser.add_argument('param_filename',
                    help='Parameter file for DBoW.')
parser.add_argument('reference_data_set_filename',
                    help='Pickle file of reference BowDataSet (needed for normalization).')
parser.add_argument('sample_filename')
parser.add_argument('--features_filename_suffix', default='',
        type=str)
parser.add_argument('--labels_filename', default=[])
args = parser.parse_args()

reference_data_set = dataset.DBoWDataSet.load_from_pkl(args.reference_data_set_filename)
dbow_pred = predictor.DBoWPredictor(args.model_filename, reference_data_set,
                                    args.param_filename,
                                    labels_filename=args.labels_filename,
                                    features_filename_suffix=args.features_filename_suffix)
y_hat, y = dbow_pred.predict(args.sample_filename)
print('Predicted label = %d' % y_hat)
print('True label = %d' % y)
