import numpy as np
import random


class BoWDataSet(object):

    def __init__(self, data_dir, object_features, labels, sample_ids=[],
                 feature_names=[], balance_classes=True):
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
            data_dir : string
                Directory where data set is stored.
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
        self.data_dir = data_dir
        self.object_features = object_features
        self.labels = labels
        self.sample_ids = sample_ids
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.balance_classes = balance_classes

        self.epochs_completed = -1
        self.index_in_epoch = np.inf
        self.index_in_no_repeats = np.inf
        self.num_in_epoch = 0
        self.num_examples = len(object_features)

        # Used to normalize BoW histograms
        self.max_n_objects = 0
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


def read_fake_data_sets(n_samples=120, n_features=10,
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
    train_data_set = BoWDataSet("fake_data", train_object_features,
                                train_labels, [], feature_names,
                                balance_classes=False)
    validate_data_set = BoWDataSet("fake_data", validate_object_features,
                                   validate_labels, [], feature_names,
                                   balance_classes=False)
    test_data_set = BoWDataSet("fake_data", test_object_features,
                               test_labels, [], feature_names,
                               balance_classes=False)
    return train_data_set, validate_data_set, test_data_set
