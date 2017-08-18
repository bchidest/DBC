import numpy as np
import os
from glob import glob
import pandas as pd
import random
import sys

sys.path.append(os.path.abspath("./DBoW/"))

import bow_dataset

#
#
#class DataSet(object):
#
#    def __init__(self, data_dir, object_features, labels, pids,
#                 feature_names, balance_classes):
#        """Construct a DataSet.
#        one_hot arg is used only if fake_data is true.  `dtype` can be either
#        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
#        `[0, 1]`.
#        Load_chunk -> how many steps' data to preload during training. Preferred between 10 and 30.
#        """
#        self.data_dir = data_dir
#        self.object_features = object_features
#        self.labels = labels
#        self.pids = pids
#        self.feature_names = feature_names
#        self.n_features = len(feature_names)
#        self.balance_classes = balance_classes
#
#        self.epochs_completed = -1
#        self.index_in_epoch = np.inf
#        self.index_in_no_repeats = np.inf
#        self.num_in_epoch = 0
#        self.num_examples = len(object_features)
#
#        self.max_n_objects = 0
#        for i in range(self.num_examples):
#            if object_features[i].shape[0] > self.max_n_objects:
#                self.max_n_objects = object_features[i].shape[0]
#
#        self.labels_list = list(set(labels))
#        self.n_labels = len(self.labels_list)
#        self.n_samples_per_label = [np.sum(labels == l) for l in self.labels_list]
#        self.ind_max_label = np.argmax(np.array(self.n_samples_per_label))
#        self.max_n_samples = self.n_samples_per_label[self.ind_max_label]
#        self.num_examples_in_epoch = self.max_n_samples*self.n_labels
#        self.n_repeats = [self.max_n_samples / n for n in self.n_samples_per_label]
#        self.n_remainder = [self.max_n_samples % n for n in self.n_samples_per_label]
#        self.ind = [list(np.where(labels == l)[0]) for l in self.labels_list]
#
#        self.perm = []
#        self.shuffle_data()
#
#    #@property
#    #def num_examples(self):
#    #    return self.num_examples
#
#    #@property
#    #def epochs_completed(self):
#    #    return self.epochs_completed
#
#    def shuffle_data(self):
#        self.perm = []
#        for i in range(self.n_labels):
#            self.perm += self.ind[i]*self.n_repeats[i]
#            self.perm += random.sample(self.ind[i], self.n_remainder[i])
#        random.shuffle(self.perm)
#
#    def reset_epoch(self):
#        self.index_in_epoch = np.inf
#        #self.shuffle_data()
#
#    def reset_no_repeats(self):
#        self.index_in_no_repeats = np.inf
#
#    def next_batch(self, batch_size):
#
#        """Return the next `batch_size` examples from this data set."""
#
#        start = self.index_in_epoch
#        self.index_in_epoch += batch_size
#        self.num_in_epoch += 1
#
#        # Finished epoch
#        if self.index_in_epoch > self.num_examples_in_epoch:
#            self.shuffle_data()
#            self.epochs_completed += 1
#            self.index_in_epoch = batch_size
#            # Shuffle the data
#            start = 0
#        end = self.index_in_epoch
#        samples = [self.object_features[self.perm[i]] for i in range(start, end)]
#        labels = [self.labels[self.perm[i]] for i in range(start, end)]
#        return samples, labels
#
#    def next_batch_no_repeats(self, batch_size):
#        start = self.index_in_no_repeats
#        self.index_in_no_repeats += batch_size
#
#        # Processed all samples
#        if self.index_in_no_repeats > self.num_examples:
#            self.index_in_no_repeats = batch_size
#            # Shuffle the data
#            start = 0
#        end = self.index_in_no_repeats
#        samples = [self.object_features[i] for i in range(start, end)]
#        labels = [self.labels[i] for i in range(start, end)]
#        return samples, labels
#


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


def read_data_sets(data_dir, label='TP53', label_type='mutation', percent_train=0.5,
                   percent_validation=0.1, balance_classes=True, min_variance=0.001,
                   normalize=True):
    feature_files = glob(os.path.join(data_dir, "features", "*Nuclei.csv"))
    #feature_files = glob(os.path.join(data_dir, "features", "*Nuclei.csv"))[0:400]
    if label_type == 'mutation':
        labels_file = os.path.join(data_dir, "labels.csv")
    elif label_type == 'subtype':
        labels_file = os.path.join(data_dir, "tcgaSubtype.csv")

    # Check that sample is in both labels and features
    if label_type == 'mutation':
        labels_df = pd.read_csv(labels_file, delimiter='\t', index_col=0)
    elif label_type == 'subtype':
        labels_df = pd.read_csv(labels_file, delimiter=',', index_col=0)
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
                temp = (temp - feature_mean)/(2*np.sqrt(feature_var)) + 0.5
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
                temp = (temp - feature_mean)/(2*np.sqrt(feature_var)) + 0.5
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
                temp = (temp - feature_mean)/(2*np.sqrt(feature_var)) + 0.5
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

    train_data_set = bow_dataset.BowDataSet(data_dir, train_object_features, train_labels,
                             tcga_pids_train, feature_names, balance_classes)
    validation_data_set = bow_dataset.BowDataSet(data_dir, validation_object_features, validation_labels,
                                  tcga_pids_validation, feature_names, balance_classes)
    test_data_set = bow_dataset.BowDataSet(data_dir, test_object_features, test_labels,
                            tcga_pids_test, feature_names, balance_classes)
    return train_data_set, validation_data_set, test_data_set

