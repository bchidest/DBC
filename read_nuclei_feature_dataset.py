import numpy as np
import os
from glob import glob
import pandas as pd
import random
import sys
import time

sys.path.append(os.path.abspath("./DBoW/"))

import bow_dataset


def load_feature_list_file(feature_list_file):
    f = open(feature_list_file, 'r')
    feature_list = []
    for line in f:
        if line[0] == "#":
            continue
        else:
            feature_list.append(line.rstrip())
    return feature_list


def load_feature_merge_list_file(feature_list_file):
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


def preprocess_data_sets(data_orig_dir, data_processed_dir, feature_list_file,
                         feature_avg_list_file, feature_max_list_file):
    feature_files = glob(os.path.join(data_orig_dir, "*Nuclei.csv"))

    features_list = load_feature_list_file(feature_list_file)
    features_avg = load_feature_merge_list_file(feature_avg_list_file)
    features_max = load_feature_merge_list_file(feature_max_list_file)

    n_samples = len(feature_files)
    temp = pd.read_csv(feature_files[0])
    for i in range(n_samples):
        temp_basename = os.path.split(feature_files[i])[1]
        print(temp_basename)
        temp = pd.read_csv(feature_files[i])
        temp.drop(['ImageNumber', 'ObjectNumber'], axis=1)
        if features_list:
            temp = temp[features_list]
        for feature_avg in features_avg:
            column_avg = pd.DataFrame(temp[feature_avg].mean(axis=1))
            column_avg.columns = [feature_avg[0].rsplit('_', 1)[0]]
            #print([feature_avg[0].rsplit('_', 1)[0]])
            temp = pd.concat([temp, column_avg], axis=1)
            #print(temp.columns)
            temp = temp.drop(feature_avg, axis=1)
        for feature_max in features_max:
            column_max = pd.DataFrame(temp[feature_max].max(axis=1))
            column_max.columns = [feature_max[0].rsplit('_', 1)[0]]
            #print([feature_max[0].rsplit('_', 1)[0]])
            temp = pd.concat([temp, column_max], axis=1)
            temp = temp.drop(feature_max, axis=1)
        temp.to_csv(os.path.join(data_processed_dir, temp_basename), index=False)


def normalize_features(features, feature_mean, feature_var):
    return (features - feature_mean)/(2*np.sqrt(feature_var)) + 0.5


def read_sample(data_dir, sample_filename, reference_data_set=[],
                label='Basal', label_type='subtype', load_label=True):
    if load_label:
        if label_type == 'mutation':
            labels_file = os.path.join(data_dir, "labels.csv")
            labels_df = pd.read_csv(labels_file, delimiter='\t', index_col=0)
        elif label_type == 'subtype':
            labels_file = os.path.join(data_dir, "tcgaSubtype.csv")
            labels_df = pd.read_csv(labels_file, delimiter=',', index_col=0)
        if label_type == 'mutation':
            tcga_pid = os.path.splitext(os.path.split(sample_filename)[1])[0][0:15]
            tcga_pid = tcga_pid.replace('-', '.')
        elif label_type == 'subtype':
            tcga_pid = os.path.splitext(os.path.split(sample_filename)[1])[0][0:12]
        if tcga_pid in labels_df.index:
            if label_type == 'mutation':
                if pd.isnull(labels_df.loc[tcga_pid, label]):
                    y = 0
                else:
                    y = 1
            elif label_type == 'subtype':
                if labels_df.loc[tcga_pid, labels_df.columns[0]] == label:
                    y = 1
                else:
                    y = 0
        else:
            print('Sample not found in labels file!')
            return [], []
    features = pd.read_csv(sample_filename)
    if reference_data_set:
        features = features[reference_data_set.feature_names]
        features = normalize_features(features,
                                      reference_data_set.feature_mean,
                                      reference_data_set.feature_var)
    if load_label:
        return features, y
    else:
        return features


def read_data_sets(data_dir, label='Basal', label_type='subtype', percent_train=0.5,
                   percent_validation=0.1, balance_classes=True, min_variance=0.001,
                   normalize=True, save_data_sets=True, output_dir='./'):
    # TODO: should remove these checks for different label types and just
    #       require uniform formatting of label file
    feature_files = glob(os.path.join(data_dir, "*Nuclei.csv"))
    if label_type == 'mutation':
        labels_file = os.path.join(data_dir, "labels.csv")
        labels_df = pd.read_csv(labels_file, delimiter='\t', index_col=0)
    elif label_type == 'subtype':
        labels_file = os.path.join(data_dir, "tcgaSubtype.csv")
        labels_df = pd.read_csv(labels_file, delimiter=',', index_col=0)
#    if label_type == 'mutation':
#        labels_df = pd.read_csv(labels_file, delimiter='\t', index_col=0)
#    elif label_type == 'subtype':
#        labels_df = pd.read_csv(labels_file, delimiter=',', index_col=0)
    n_samples = len(feature_files)

    ind = [[], []]
    for i in range(n_samples):
        if label_type == 'mutation':
            tcga_pid = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:15]
            tcga_pid = tcga_pid.replace('-', '.')
        elif label_type == 'subtype':
            tcga_pid = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:12]
        if tcga_pid in labels_df.index:
            if label_type == 'mutation':
                if pd.isnull(labels_df.loc[tcga_pid, label]):
                    ind[0].append(i)
                else:
                    ind[1].append(i)
            elif label_type == 'subtype':
                if labels_df.loc[tcga_pid, labels_df.columns[0]] == label:
                    ind[1].append(i)
                else:
                    ind[0].append(i)

    n_samples = [len(i) for i in ind]
    print("Number of samples per label:")
    print(n_samples)
    n_samples_total = 0
    for i in range(len(ind)):
        n_samples_total += n_samples[i]
    for i in range(len(ind)):
        random.shuffle(ind[i])
    temp = pd.read_csv(feature_files[0])
    feature_names = temp.columns
    train_samples = []
    train_object_features = []
    validation_object_features = []
    test_object_features = []
    tcga_pids_train = []
    tcga_pids_validation = []
    tcga_pids_test = []
    feature_mean = pd.Series(data=np.zeros(len(feature_names),), index=feature_names)
    feature_var = pd.Series(data=np.zeros(len(feature_names),), index=feature_names)
    # Add training patients and calculate average mean and variance of features
    print("Training patients:")
    n_samples_train = 0
    for j in range(len(ind)):
        n_samples_train += int(n_samples[j]*percent_train)
        for i in ind[j][0: int(n_samples[j]*percent_train)]:
            if label_type == 'mutation':
                tcga_pid = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:15]
                tcga_pid = tcga_pid.replace('-', '.')
            elif label_type == 'subtype':
                tcga_pid = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:12]
            print(tcga_pid)
            tcga_pids_train.append(tcga_pid)
            temp = pd.read_csv(feature_files[i])
            feature_mean += temp.mean()
            if normalize:
                feature_var += temp.var()
            train_samples.append(temp)
    feature_mean /= n_samples_train
    feature_var /= n_samples_train
    valid_feature_ind = feature_var > min_variance
    feature_names = list(feature_names[valid_feature_ind])
    for j in range(len(ind)):
#        for i in range(int(n_samples[j]*percent_train)):
        for i in ind[j][0: int(n_samples[j]*percent_train)]:
            temp = pd.read_csv(feature_files[i])
            if normalize:
                temp = normalize_features(temp, feature_mean, feature_var)
#            if normalize:
#                temp = (train_samples[i] - feature_mean)/(2*np.sqrt(feature_var)) + 0.5
#            else:
#                temp = train_samples[i]
            train_object_features.append(temp.loc[:, valid_feature_ind].values)
    print("Validation patients:")
    for j in range(len(ind)):
        for i in ind[j][int(n_samples[j]*percent_train):int(n_samples[j]*(percent_train + percent_validation))]:
            if label_type == 'mutation':
                tcga_pid = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:15]
                tcga_pid = tcga_pid.replace('-', '.')
            elif label_type == 'subtype':
                tcga_pid = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:12]
            print(tcga_pid)
            tcga_pids_validation.append(tcga_pid)
            temp = pd.read_csv(feature_files[i])
            if normalize:
                temp = normalize_features(temp, feature_mean, feature_var)
            validation_object_features.append(temp.loc[:, valid_feature_ind].values)
    print("Testing patients:")
    for j in range(len(ind)):
        for i in ind[j][int(n_samples[j]*(percent_train + percent_validation)):]:
            if label_type == 'mutation':
                tcga_pid = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:15]
                tcga_pid = tcga_pid.replace('-', '.')
            elif label_type == 'subtype':
                tcga_pid = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:12]
            print(tcga_pid)
            tcga_pids_test.append(tcga_pid)
            temp = pd.read_csv(feature_files[i])
            if normalize:
                temp = normalize_features(temp, feature_mean, feature_var)
            test_object_features.append(temp.loc[:, valid_feature_ind].values)
    train_labels_df = labels_df.loc[tcga_pids_train]
    validation_labels_df = labels_df.loc[tcga_pids_validation]
    test_labels_df = labels_df.loc[tcga_pids_test]
    if label_type == 'mutation':
        train_labels = pd.isnull(train_labels_df[label]).values.astype('uint8')
        validation_labels = pd.isnull(validation_labels_df[label]).values.astype('uint8')
        test_labels = pd.isnull(test_labels_df[label]).values.astype('uint8')
    elif label_type == 'subtype':
        train_labels = (train_labels_df == label).values.astype('uint8').ravel()
        validation_labels = (validation_labels_df == label).values.astype('uint8').ravel()
        test_labels = (test_labels_df == label).values.astype('uint8').ravel()

    train_data_set = bow_dataset.DBowDataSet(train_object_features, train_labels,
                             tcga_pids_train, feature_names, balance_classes)
    validation_data_set = bow_dataset.DBowDataSet(validation_object_features, validation_labels,
                                  tcga_pids_validation, feature_names, balance_classes)
    test_data_set = bow_dataset.DBowDataSet(test_object_features, test_labels,
                            tcga_pids_test, feature_names, balance_classes)

    if save_data_sets:
        # Save the necessary contents for each data set to a pkl file
        time_str = ''
        train_filename = os.path.join(output_dir, label + '_train.pkl')
        # If data set pkl already exists, append time to name of new pkl
        if os.path.isfile(train_filename):
            time_str = '_' + str(int(time.time()))
        train_data_set.save_to_pkl(os.path.join(output_dir, label + '_train_' +
                                                time_str + '.pkl'))
        train_data_set.save_without_features_to_pkl(os.path.join(output_dir,
                                                                 label +
                                                                 '_train_no_features_'
                                                                 + time_str + '.pkl'))
        validation_data_set.save_to_pkl(os.path.join(output_dir, label + '_validation_' +
                                                     time_str + '.pkl'))
        test_data_set.save_to_pkl(os.path.join(output_dir, label + '_test_' +
                                               time_str + '.pkl'))
#                                     train_object_features,
#                                     train_labels,
#                                     tcga_pids_train, feature_names)
#    bow_dataset.save_data_set_to_pkl(os.path.join(output_dir, label + '_validation_' +
#                                     time_str + '.pkl'),
#                                     validation_object_features,
#                                     validation_labels,
#                                     tcga_pids_validation, feature_names)
#    bow_dataset.save_data_set_to_pkl(os.path.join(output_dir, label + '_test_' +
#                                     time_str + '.pkl'),
#                                     test_object_features,
#                                     test_labels,
#                                     tcga_pids_test, feature_names)

    return train_data_set, validation_data_set, test_data_set
