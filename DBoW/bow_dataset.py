import numpy as np
import random
import pandas as pd
import cPickle as pkl
import os
from glob import glob
import time


def generate_fake_sample(n_objects_range, n_features, percent_objects):
    '''
    Generate a sample of the fake dataset.

    Each feature in each object is generated according to one of two
    Gaussian distributions, with mean m_0 or m_1.

    Parameters:
        n_objects_range: 2-element list of ints
            The number of objects in a particular sample is picked
            uniformly at random from this range.
        n_features : int
            The dimensionality of each object in a sample.
        percent_objects : float [0-1]
            The balance of the two objects for the mixed class.

    Returns:
        features : numpy.array (n_objects, n_features)
            The generated sample.
     '''
    mu_0 = 0.3
    mu_1 = 0.6
    s = 0.05
    n_objects = int(np.random.uniform(n_objects_range[0], n_objects_range[1]))
    features = np.zeros((n_objects, n_features), dtype='float32')
    ind = range(n_objects)
    random.shuffle(ind)
    for i in ind[0:int(n_objects*percent_objects)]:
        features[i, :] = np.random.normal(mu_1, s, n_features)
    for i in ind[int(n_objects*percent_objects):]:
        features[i, :] = np.random.normal(mu_0, s, n_features)
    return features


class DBoWDataSet(object):
    def __init__(self, object_features=[], labels=[], sample_ids=[],
                 feature_names=[], feature_means=[], feature_vars=[],
                 max_n_objects=[], balance_classes=True):
        '''
        Class to store a BoW data set.

        There are two ways to fetch batches from this data set class.
        The first, next_batch(), should be considered the default for
        training, as it will obey the balance_classes parameter.
        If balance_classes is True, each epoch will consist of sufficient
        repeats of samples to balance the classes. When training, it may
        be desirable to have balanced classes so that the network equally
        weights each class.

        (In the future, this should be modified to allow for modified
        weightings that are not necessarily equal. This would allow for
        training for a particular FPR and FNR.)

        The second, next_batch_no_repeats(), can be considered as a
        separate, concurrent epoch that contains no repeats, regardless
        of balance_classes. This is often used for evaluating a trained
        model on the data set, since class-balanced metrics can easily
        be computed after-the-fact.
        Parameters:
            object_features : list of numpy.arrays (n_objects, n_features)
                Each element of the list is a sample, which is a numpy
                array of size (n_objects, n_features), where n_objects
                can vary from sample to sample.
            labels : numpy.array (n_samples), dtype='uint8'
                The label associated each sample in object_features.
            sample_ids : list, len=n_samples
                IDs of the the samples for later reference.
            feature_names : list, len=n_features
                Names of features for later reference.
            balance_classes : bool
                If the classes have an unequal number of samples, balance
                them by repeating samples of the class of less samples.
        '''
        self.object_features = object_features
        self.labels = labels
        self.sample_ids = sample_ids
        self.feature_names = feature_names
        self.feature_means = feature_means
        self.feature_vars = feature_vars
        self.n_features = len(feature_names)
        self.balance_classes = balance_classes

        self.epochs_completed = -1
        self.index_in_epoch = np.inf
        self.index_in_no_repeats = np.inf
        self.num_in_epoch = 0
        self.num_examples = len(object_features)

        # Used to normalize BoW histograms
        self.max_n_objects = 0
        if not object_features:
            self.max_n_objects = max_n_objects
        else:
            for i in range(self.num_examples):
                if object_features[i].shape[0] > self.max_n_objects:
                    self.max_n_objects = object_features[i].shape[0]

        self.labels_list = list(set(labels))
        self.n_labels = len(self.labels_list)
        self.n_samples_per_label = [np.sum(labels == l) for l in self.labels_list]
        self.ind_max_label = np.argmax(np.array(self.n_samples_per_label))
        self.max_n_samples = self.n_samples_per_label[self.ind_max_label]
        self.num_examples_in_epoch = self.max_n_samples*self.n_labels
        # How many times must all samples of under-represented classes
        # be resampled
        self.n_repeats = [self.max_n_samples / n for n in self.n_samples_per_label]
        # How many additional samples of under-represented classes must be
        # generated
        self.n_remainder = [self.max_n_samples % n for n in self.n_samples_per_label]
        self.ind = [list(np.where(labels == l)[0]) for l in self.labels_list]

        # Random ordering of samples
        self.perm = []

        self.shuffle_data()

    @classmethod
    def load_from_pkl(cls, data_set_filename):
        '''
        Attempts to load a DBoWDataSet from a pkl file.

        The presence of all necessary variables is checked and then passed
        to the DBoWDataSet constructor. Variables are assumed to be preceded
        in the pkl file by a string their name.
        '''
        f_data_set = open(data_set_filename, 'rb')
        data_set_info = pkl.load(f_data_set)
        dbow_data_set_members = {
                'object_features': [],
                'labels': [],
                'sample_ids': [],
                'feature_names': [],
                'feature_means': [],
                'feature_vars': [],
                'max_n_objects': [],
                }
        for i in range(0, len(data_set_info), 2):
            dbow_data_set_members[data_set_info[i]] = data_set_info[i+1]
        data_set = cls(dbow_data_set_members['object_features'],
                       dbow_data_set_members['labels'],
                       dbow_data_set_members['sample_ids'],
                       dbow_data_set_members['feature_names'],
                       dbow_data_set_members['feature_means'],
                       dbow_data_set_members['feature_vars'],
                       dbow_data_set_members['max_n_objects'])
        f_data_set.close()
        return data_set

    def shuffle_data(self):
        '''Shuffle the ordering of the samples.'''
        self.perm = []
        for i in range(self.n_labels):
            self.perm += self.ind[i]*self.n_repeats[i]
            self.perm += random.sample(self.ind[i], self.n_remainder[i])
        random.shuffle(self.perm)

    def reset_epoch(self):
        '''Reset the epoch counter.'''
        self.index_in_epoch = np.inf

    def reset_no_repeats(self):
        '''Reset the no_repeats epoch counter.'''
        self.index_in_no_repeats = np.inf

    def next_batch(self, batch_size):
        '''Return the next batch.'''
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        self.num_in_epoch += 1

        # Finished epoch
        if self.index_in_epoch > self.num_examples_in_epoch:
            self.shuffle_data()
            self.epochs_completed += 1
            self.index_in_epoch = batch_size
            start = 0
        end = self.index_in_epoch
        samples = [self.object_features[self.perm[i]] for i in range(start, end)]
        labels = [self.labels[self.perm[i]] for i in range(start, end)]
        return samples, labels

    def next_batch_no_repeats(self, batch_size):
        '''Return the next batch without repeats.'''
        start = self.index_in_no_repeats
        self.index_in_no_repeats += batch_size

        # Processed all unique samples
        if self.index_in_no_repeats > self.num_examples:
            self.index_in_no_repeats = batch_size
            start = 0
        end = self.index_in_no_repeats
        samples = [self.object_features[i] for i in range(start, end)]
        labels = [self.labels[i] for i in range(start, end)]
        return samples, labels

    def save_to_pkl(self, data_set_filename_prefix=[]):
        '''
        Allows you to save the important member variables, from which a
        DBoWDataSet instance can be created, into a pkl file. This exists
        mostly to allow backward compatability in the future in case the
        DBoWDataSet class changes, you still have access to the raw data and
        variables.

        The name of each variable is saved in the index preceding the
        variable itself for later reference, especially in the case that
        this order must be changed for unforseen reasons in future versions.
        '''
        if not data_set_filename_prefix:
            data_set_filename_prefix = 'data_set'
        f_data_set = open(data_set_filename_prefix + '.pkl', 'wb')
        pkl.dump(['object_features', self.object_features,
                  'labels', self.labels,
                  'sample_ids', self.sample_ids,
                  'feature_names', self.feature_names,
                  'feature_means', self.feature_means,
                  'feature_vars', self.feature_vars,
                  'max_n_objects', self.max_n_objects],
                 f_data_set)
        f_data_set.close()

    def save_without_features_to_pkl(self, data_set_filename_prefix=[]):
        '''
        Allows you to save all important variables except the actual object
        features to a pkl file. This is important to having normalization info,
        including the feature means and variances and the max_n_objects so that
        new data can be properly normalized.
        '''
        if not data_set_filename_prefix:
            data_set_filename_prefix = 'data_set'
        f_data_set = open(data_set_filename_prefix + '_no_features.pkl', 'wb')
        pkl.dump(['object_features', [],
                  'labels', self.labels,
                  'sample_ids', self.sample_ids,
                  'feature_names', self.feature_names,
                  'feature_means', self.feature_means,
                  'feature_vars', self.feature_vars,
                  'max_n_objects', self.max_n_objects],
                 f_data_set)
        f_data_set.close()


def save_data_set_to_pkl(data_set_file, object_features, labels,
                         sample_ids, feature_names):
    f_data_set = open(data_set_file, 'wb')
    pkl.dump([object_features, labels, sample_ids, feature_names],
             f_data_set)
    f_data_set.close()


def load_data_set_from_pkl(data_set_file):
    f_data_set = open(data_set_file, 'rb')
    data_set_info = pkl.load(f_data_set)
    data_set = DBoWDataSet(data_set_info[0], data_set_info[1],
                           data_set_info[2], data_set_info[3])
    f_data_set.close()
    return data_set


def normalize_features(features, feature_mean, feature_var):
    return (features - feature_mean)/(2*np.sqrt(feature_var)) + 0.5


def read_sample(sample_filename, labels_file=[], reference_data_set=[],
                label='Basal'):
    if labels_file:
        labels_df = pd.read_csv(labels_file, delimiter=',', index_col=0)
        sample_id = os.path.splitext(os.path.split(sample_filename)[1])[0][0:12]
        if sample_id in labels_df.index:
            if labels_df.loc[sample_id, labels_df.columns[0]] == label:
                y = 1
            else:
                y = 0
        else:
            print('Sample not found in labels file! Returning features only.')
            y = []
    features = pd.read_csv(sample_filename)
    if reference_data_set:
        features = features[reference_data_set.feature_names]
        features = normalize_features(features,
                                      reference_data_set.feature_means,
                                      reference_data_set.feature_vars)
    features = features.values
    if labels_file:
        return features, y
    else:
        return features


def read_data_sets(data_dir, labels_file,
                   label='Basal', percent_train=0.5,
                   percent_validation=0.1, balance_classes=True, min_variance=0.001,
                   normalize=True, save_data_sets=True, output_dir='./',
                   features_file_suffix=''):
    '''
    Feature files should be saved as CSV.
    No other CSV files with the same features_file_suffix should be in the
    data_dir.
    '''
    feature_files = glob(os.path.join(data_dir, '*' + features_file_suffix +
                                      '.csv'))
    labels_df = pd.read_csv(labels_file, delimiter=',', index_col=0)
    n_samples = len(feature_files)

    ind = [[], []]
    for i in range(n_samples):
        sample_id = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:12]
        if sample_id in labels_df.index:
            if labels_df.loc[sample_id, labels_df.columns[0]] == label:
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
    sample_ids_train = []
    sample_ids_validation = []
    sample_ids_test = []
    feature_mean = pd.Series(data=np.zeros(len(feature_names),), index=feature_names)
    feature_var = pd.Series(data=np.zeros(len(feature_names),), index=feature_names)
    # Add training patients and calculate average mean and variance of features
    print("Training patients:")
    n_samples_train = 0
    for j in range(len(ind)):
        n_samples_train += int(n_samples[j]*percent_train)
        for i in ind[j][0: int(n_samples[j]*percent_train)]:
            sample_id = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:12]
            print(sample_id)
            sample_ids_train.append(sample_id)
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
        for i in ind[j][0: int(n_samples[j]*percent_train)]:
            temp = pd.read_csv(feature_files[i])
            if normalize:
                temp = normalize_features(temp, feature_mean, feature_var)
            train_object_features.append(temp.loc[:, valid_feature_ind].values)
    print("Validation patients:")
    for j in range(len(ind)):
        for i in ind[j][int(n_samples[j]*percent_train):int(n_samples[j]*(percent_train + percent_validation))]:
            sample_id = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:12]
            print(sample_id)
            sample_ids_validation.append(sample_id)
            temp = pd.read_csv(feature_files[i])
            if normalize:
                temp = normalize_features(temp, feature_mean, feature_var)
            validation_object_features.append(temp.loc[:, valid_feature_ind].values)
    print("Testing patients:")
    for j in range(len(ind)):
        for i in ind[j][int(n_samples[j]*(percent_train + percent_validation)):]:
            sample_id = os.path.splitext(os.path.split(feature_files[i])[1])[0][0:12]
            print(sample_id)
            sample_ids_test.append(sample_id)
            temp = pd.read_csv(feature_files[i])
            if normalize:
                temp = normalize_features(temp, feature_mean, feature_var)
            test_object_features.append(temp.loc[:, valid_feature_ind].values)
    train_labels_df = labels_df.loc[sample_ids_train]
    validation_labels_df = labels_df.loc[sample_ids_validation]
    test_labels_df = labels_df.loc[sample_ids_test]
    train_labels = (train_labels_df == label).values.astype('uint8').ravel()
    validation_labels = (validation_labels_df == label).values.astype('uint8').ravel()
    test_labels = (test_labels_df == label).values.astype('uint8').ravel()

    train_data_set = DBoWDataSet(train_object_features, train_labels,
                             sample_ids_train, feature_names, balance_classes)
    validation_data_set = DBoWDataSet(validation_object_features, validation_labels,
                                  sample_ids_validation, feature_names, balance_classes)
    test_data_set = DBoWDataSet(test_object_features, test_labels,
                            sample_ids_test, feature_names, balance_classes)

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

    return train_data_set, validation_data_set, test_data_set


def load_fake_data_sets(n_samples=120, n_features=10,
                        n_objects_range=[1000, 1500],
                        percent_objects=0.5, percent_train=0.5,
                        percent_validate=0.1):
    '''
    Generate a fake BoW data set of training, validation, and testing.

    A BoW data set is composed of samples, which are themselves composed of
    objects. In the fake data set, there are two types of possible objects,
    and one class of samples is composed of both types, while the other class
    is composed only of one type. The learned DBoW classifier learns to
    distinguish these two types.

    Parameters:
        n_samples : int
            The total number of samples in the dataset.
        n_features : int
            The dimensionality of each object in a sample.
        n_objects_range: 2-element list of ints
            The number of objects in a particular sample is picked
            uniformly at random from this range.
        percent_objects : float [0-1]
            The balance of the two objects for the mixed class.
        percent_train : float [0-1]
            The percent of samples used for training.
        percent_validate : float [0-1]
            The percent of samples used for validation. (The remaining are
            used for testing.)

    Returns:
        test_data_set, validate_data_set, test_data_set: BoWDataSet
    '''
    feature_names = ['feature' + str(i) for i in range(n_features)]
    n_train = int(n_samples*percent_train)
    n_validate = int(n_samples*percent_validate)
    n_test = n_samples - n_validate - n_train
    train_object_features = []
    validate_object_features = []
    test_object_features = []
    print("Training samples:")
    for i in range(n_train / 2):
        train_object_features.append(generate_fake_sample(n_objects_range,
                                                          n_features,
                                                          0.0))
    for i in range(n_train - n_train / 2):
        train_object_features.append(generate_fake_sample(n_objects_range,
                                                          n_features,
                                                          percent_objects))
    print("Validation samples:")
    for i in range(n_validate / 2):
        validate_object_features.append(generate_fake_sample(n_objects_range,
                                                             n_features,
                                                             0.0))
    for i in range(n_validate - n_validate / 2):
        validate_object_features.append(generate_fake_sample(n_objects_range,
                                                             n_features,
                                                             percent_objects))
    print("Testing samples:")
    for i in range(n_test / 2):
        test_object_features.append(generate_fake_sample(n_objects_range,
                                                         n_features,
                                                         0.0))
    for i in range(n_test - n_test / 2):
        test_object_features.append(generate_fake_sample(n_objects_range,
                                                         n_features,
                                                         percent_objects))
    train_labels = np.zeros((n_train,), dtype='uint8')
    train_labels[n_train / 2:] = 1
    validate_labels = np.zeros((n_validate,), dtype='uint8')
    validate_labels[n_validate / 2:] = 1
    test_labels = np.zeros((n_test,), dtype='uint8')
    test_labels[n_test / 2:] = 1
    train_data_set = DBoWDataSet("fake_data", train_object_features,
                                 train_labels, [], feature_names,
                                 balance_classes=False)
    validate_data_set = DBoWDataSet("fake_data", validate_object_features,
                                    validate_labels, [], feature_names,
                                    balance_classes=False)
    test_data_set = DBoWDataSet("fake_data", test_object_features,
                                test_labels, [], feature_names,
                                balance_classes=False)
    return train_data_set, validate_data_set, test_data_set
