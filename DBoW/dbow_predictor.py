import run_discriminative_bow as rdb
import bow_dataset


class DBoWPredictor(object):
    def __init__(self, model_filename, reference_data_set, labels_filename=[],
                 label_names=[]):
        '''
        Class to store a BoW data set.
        '''
        self.model_filename = model_filename
        self.reference_data_set = reference_data_set
        self.labels_filename = labels_filename
        self.label_names = label_names

    def predict(self, sample_filename):
        # TODO: this will predict from filename, so it will handle reading
        #       and preprocessing file
        x, y = bow_dataset.read_sample(sample_filename, self.labels_filename,
                                       self.reference_data_set)
#                                       self.label_names[0])
        if self.label_names:
            y_hat, label_hat = self.predict_raw(x)
            return y_hat, label_hat, y
        else:
            y_hat = self.predict_raw(x)
            return y_hat, y

    def predict_raw(self, sample):
        y_hat = rdb.predict_samples(self.model_filename,
                                    [sample],
                                    self.reference_data_set.max_n_objects)
        if self.label_names:
            label_hat = [self.label_names[y] for y in y_hat]
            return y_hat[0], label_hat[0]
        else:
            return y_hat[0]
