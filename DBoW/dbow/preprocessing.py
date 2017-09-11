import os
from glob import glob
import pandas as pd


def load_feature_list_file(feature_list_file):
    '''
    Helper function for feature_set_modification() to load a
    file of features.
    '''
    f = open(feature_list_file, 'r')
    feature_list = []
    for line in f:
        if line[0] == "#":
            continue
        else:
            feature_list.append(line.rstrip())
    return feature_list


def load_feature_merge_list_file(feature_list_file):
    '''
    Helper function for feature_set_modification() to load a
    file of features to merge.
    '''
    if feature_list_file is None:
        return None
    f = open(feature_list_file, 'r')
    feature_list = []
    feature_group = []
    for line in f:
        if not line.rstrip():
            if feature_group:
                feature_list.append(feature_group)
                feature_group = []
        else:
            feature_group.append(line.rstrip())
    if feature_group:
        feature_list.append(feature_group)
    return feature_list


def feature_set_modification(data_orig_dir, data_processed_dir,
                             feature_list_file, feature_avg_list_file,
                             feature_max_list_file,
                             feature_filename_suffix=''):
    '''
    Modifies the set of features across all samples by removing or
    merging (either by taking the max or average) features.

    A new copy of the CSV file of the modified features for each sample
    is made in data_processed_dir.

    Parameters:
        data_orig_dir : string
            Directory of CSV files of samples.
        data_processed_dir : string
            Directory in which to save new CSV files for samples.
        feature_list_file : string
            Filename of a list of features to keep. Each feature should
            be on a separate line, with no empty lines. Commented
            lines are ignored to allow for easy manipulation of the list
            without losing track of the names of original features.
        feature_avg_list_file : string
            Filename of a list of lists of features to average. A group
            of features to average should be on separate, continuous
            lines with empty lines between sets of features to average.
            It is assumed that features in groups will have the same name
            except differing only in the suffix (for example, for
            CellProfiler, texture features are taken at evenly-spaced
            rotations, and the angle in degrees for each rotation is
            appended to the end of each feature).
        feature_max_list_file : string
            Similar to feature_avg_list_file, but features in groups are
            merged by taking the max
        feature_filename_suffix : string
            A suffix string that designates that a CSV file in
            data_orig_dir is the feature file of a sample in the data set.
    '''
    feature_files = glob(os.path.join(data_orig_dir, '*' +
                                      feature_filename_suffix + '.csv'))
    features_list = load_feature_list_file(feature_list_file)
    features_avg = load_feature_merge_list_file(feature_avg_list_file)
    features_max = load_feature_merge_list_file(feature_max_list_file)

    n_samples = len(feature_files)
    temp = pd.read_csv(feature_files[0])
    for i in range(n_samples):
        temp_basename = os.path.split(feature_files[i])[1]
        print(temp_basename)
        temp = pd.read_csv(feature_files[i])
        if features_list:
            temp = temp[features_list]
        for feature_avg in features_avg:
            column_avg = pd.DataFrame(temp[feature_avg].mean(axis=1))
            column_avg.columns = [feature_avg[0].rsplit('_', 1)[0]]
            temp = pd.concat([temp, column_avg], axis=1)
            temp = temp.drop(feature_avg, axis=1)
        for feature_max in features_max:
            column_max = pd.DataFrame(temp[feature_max].max(axis=1))
            column_max.columns = [feature_max[0].rsplit('_', 1)[0]]
            temp = pd.concat([temp, column_max], axis=1)
            temp = temp.drop(feature_max, axis=1)
        temp.to_csv(os.path.join(data_processed_dir, temp_basename), index=False)
