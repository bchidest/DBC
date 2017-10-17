'''
The code to load datasets while training. This assumes that the data contains masks data. That is, the data was generated using gen_training_masks_cw.py. For data generated using gen_training_data_cw.py use read_(??)
'''

'''
TODO:
    - just make image_shape a large number, and call image_temp image_buffer,
      so we can load images of any size (different datasets have differently
      size images
'''

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
import csv
import glob
import pandas as pd
from scipy.misc import imread
from PIL import Image

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#flags = tf.app.flags
#FLAGS = flags.FLAGS
#flags.DEFINE_float('percent_train', 0.7, 'percent train.')
#flags.DEFINE_float('percent_valid', 0.1, 'percent valid.')
window_radius = 25

image_folder = '/home/ben/datasets/Nuclei/JPath_2017/he'
percent_train = 0.7
#percent_valid = 0.1


suffix_dict = {0:   "0_0_h.png", 1:   "0_0_e.png", 2:  "90_0_h.png", 3: "90_0_e.png",
               4: "180_0_h.png", 5: "180_0_e.png", 6: "270_0_h.png", 7:"270_0_e.png",
               8:   "0_1_h.png", 9:   "0_1_e.png", 10:  "0_2_h.png", 11: "0_2_e.png"}

rot_map = ['0', '90', '180', '270']


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def create_rot_LUT(dim):
    # NOTE: y=vertical, x=horizontal
    y, x = np.meshgrid(np.arange(dim), np.arange(dim))
    rot_LUT_x = np.zeros((4, dim, dim), dtype='int16')
    rot_LUT_y = np.zeros((4, dim, dim), dtype='int16')
    rot_LUT_x[0, :, :] = x
    rot_LUT_y[0, :, :] = y
    for i in range(1, 4):
        rot_LUT_x[i, :, :] = np.rot90(x, -i)
        rot_LUT_y[i, :, :] = np.rot90(y, -i)
    return rot_LUT_y, rot_LUT_x


def choice_with_repeats(n, size, dtype='int16'):
    """Samples without replacement in the case that size is
    bigger than n.
    """
    s = np.zeros((size,), dtype=dtype)
    n_repeats = size / n
    n_remainder = size % n
    ind = np.arange(n, dtype=dtype)
    for i in range(n_repeats):
        np.random.shuffle(ind)
        s[i*n: (i+1)*n] = ind
    s[n_repeats*n:] = np.random.choice(n, n_remainder, replace=False)
    return s


class DataSet(object):

    def __init__(self, data_dir,
                 batch_size=100, image_buffer_shape=[4000, 4000], one_hot=False,
                 n_nuclei_per_image=[100, 10], n_pixels_per_nucleus=5,
                 dtype='float32', is_regression=True, chunk_size=486,
                 percent_train=0.9, n_partitions=2):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
    '"""
        # NOTE: If we don't precompute rotated and flipped x,y locations,
        # we need to know the size of each image. This could be stored in
        # each ImageOfNuclei instance. However, currently, we use a LUT
        # to determine the proper x,y locations, and we'd need a separte
        # LUT for each image size. So long as there aren't too many
        # different image sizes, this is probably okay, but needs to be
        # considered. It should still be worth the increased memory to
        # eliminate computing the modified x,y for each patch.

        self.dtype = tf.as_dtype(dtype).base_dtype
        if self.dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                        self.dtype)

        self.data_dir = data_dir
        # Find all images in data_dir
        img_dirs = glob.glob(os.path.join(data_dir, "*"))
        self.images = []
        labels_list = []
        for i, d in zip(range(len(img_dirs)), img_dirs):
            print(d)
            pixels_df = pd.read_pickle(os.path.join(d, "labels.pkl"))
            #r_LUTs = create_rot_LUT(2000)
            #img_instance = ImageOfNuclei(img_basename, pixels_df, r_LUTs, percent_train)
            img_instance = ImageOfNuclei(d, pixels_df, percent_train)
            print(img_instance.n_nuclei)
            self.images.append(img_instance)
            labels_list += list(set(pixels_df['Class']))

        self.image_names = []
        self.image_rotated_names_list = []
        image_shapes = []
        for image in self.images:
            self.image_names.append(image.image_name)
            image_shapes.append(image.shape)
            for r in rot_map:
                self.image_rotated_names_list.append(os.path.join(image.image_name, r))
        self.image_rotated_names = {self.image_rotated_names_list[i]: []
                                    for i in range(len(self.image_rotated_names_list))}
        image_shapes = list(set(image_shapes))

        # Fill rot_LUTs of images and their nuclei
        rot_LUTs = []
        for image_shape in image_shapes:
            rot_LUTs.append(create_rot_LUT(image_shape[0]))
        self.rot_LUTs = {s[0]: r for s, r in zip(image_shapes, rot_LUTs)}
        for image in self.images:
            image.fill_rot_LUTs(self.rot_LUTs[image.shape[0]])

        self.n_images = len(self.images)
        self.labels_list = list(set(labels_list))
        self.n_labels = len(self.labels_list)
        # TODO: currently, ImageofNuclei class is hardcoded for 2 partitions
        assert n_partitions == 2
        self.n_partitions = n_partitions
        # NOTE: could take median number of nuclei per image each epoch
        #self.n_nuclei_per_image = data_set_summary['n_nuclei'].median()
        self.n_nuclei_per_image = n_nuclei_per_image
        # NOTE: is it better to only take as many pixels as available, or to
        # resample?
        self.n_pixels_per_nucleus = n_pixels_per_nucleus
        # TODO: want to use class resampling, but currently not implemented
        self.percent_pixels_per_label = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.n_pixels = [self.n_images*self.n_labels *
                         n*self.n_pixels_per_nucleus for n in
                         n_nuclei_per_image]
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.is_regression = is_regression

        # TODO: this is hardcoded for 6 classes...
        self.regress_map = np.array([0., 0.2, 0.9, 0.93, 0.96, 1.0])

        self._epochs_completed = [-1] * self.n_partitions
        self._index_in_epoch = [np.inf] * self.n_partitions
        self._num_in_epoch = [-1] * self.n_partitions
        self._num_examples = self.n_pixels
        for n in self._num_examples:
            assert self.chunk_size * self.batch_size <= n

        # Epoch
        self.pixels = [np.zeros((n, 2), dtype='int16') for n in self.n_pixels]
        self.pixel_image_names = [[] for n in range(self.n_partitions)]
        # NOTE: For now, labels don't actually change, due to the way I create
        #       each in next_batch() (iterate over image, then over labels).
        #       Sampling indices with self.perm incurs necessary randomness.
        #self.labels = np.zeros((self.n_pixels), dtype='uint8')
        # NOTE: Again, here we assume that all nucleu have same label set!
        self.labels = [np.tile(np.repeat(self.labels_list, n*self.n_pixels_per_nucleus),
                               self.n_images) for n in self.n_nuclei_per_image]
        self.regress = []
        for i in range(self.n_partitions):
            temp_array = np.empty((self._num_examples[i], 1), dtype='float32')
            #temp_array = np.empty((self._num_examples[i], ), dtype='float32')
            temp_array[:, 0] = [self.regress_map[l] for l in self.labels[i]]
            #temp_array[:] = [self.regress_map[l] for l in self.labels[i]]
            self.regress.append(temp_array)
        self.perm = [np.arange(n, dtype='uint32') for n in self.n_pixels]
        for i in range(self.n_partitions):
            np.random.shuffle(self.perm[i])

        # Chunk
        self.image_chunk = [np.zeros((chunk_size*batch_size,
                                     2*window_radius + 1,
                                     2*window_radius + 1, 2), dtype=dtype)
                            for i in range(self.n_partitions)]
        self._num_in_chunk = [-1] * self.n_partitions

        # Temporary buffers
        self.image_buffer_shape = image_buffer_shape
        self.image_buffer = np.zeros((self.image_buffer_shape[0],
                                      self.image_buffer_shape[1], 2), dtype='uint8')

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, partition):

        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch[partition]
        self._index_in_epoch[partition] += self.batch_size
        self._num_in_epoch[partition] += 1
        self._num_in_chunk[partition] += 1
        fresh_start = False

        # Finished epoch
        if self._index_in_epoch[partition] > self._num_examples[partition]:
            self._epochs_completed[partition] += 1
            self._num_in_epoch[partition] = 0
            # Load new pixel locations
            self.pixel_image_names[partition] = []
            pixel_ind = 0
            for i in range(self.n_images):
                ion = self.images[i]
                # NOTE: To change weighting of labels, change must be
                #       made here.
                for k in range(self.n_labels):
                    pixel_ind_next = pixel_ind + self.n_nuclei_per_image[partition] *\
                                            self.n_pixels_per_nucleus
                    ion.get_random_nuclei_pixels(self.pixels[partition][
                                                     pixel_ind:pixel_ind_next, :],
                                                 self.pixel_image_names[partition],
                                                 self.n_nuclei_per_image[partition],
                                                 self.n_pixels_per_nucleus,
                                                 self.labels_list[k])
                    pixel_ind = pixel_ind_next

            # Shuffle the data
            np.random.shuffle(self.perm[partition])
            # NOTE: AVOID THE FOLLOWING!!:
            #       self.pixels = self.pixels[:]
            #       Since I don't think this can be done in place, I expect it
            #       to require a full temp copy.
            #       Instead, we'll just shuffle perm and acces self.pixels as
            #       needed when loading the next chunk of images.

            # Start next epoch
            print('Next epoch! Partition=' + str(partition))
            fresh_start = True
            start = 0
            self._index_in_epoch[partition] = self.batch_size
        end = self._index_in_epoch[partition]

        if self._num_in_chunk[partition] == self.chunk_size or fresh_start is True:
            # NOTE: We really should check that the epoch is a multiple of
            #       chunk_size*batch_size
            print('Loading more data! Partition=' + str(partition))
            self._num_in_chunk[partition] = 0

            self.image_rotated_names = {self.image_rotated_names.keys()[i]: []
                                        for i in range(len(self.image_rotated_names_list))}
            #chunk_image_names = \
            #        self.pixel_image_names[start:
            #                               start+self.chunk_size*self._batch_size]
            for i in range(self.chunk_size*self.batch_size):
                self.image_rotated_names[self.pixel_image_names[partition]
                        [self.perm[partition][start + i]]].append(i)
            for image_rotated_name in self.image_rotated_names.keys():
                ind_list = self.image_rotated_names[image_rotated_name]
                if not ind_list:
                    continue
                full_image_name = os.path.join(self.data_dir, image_rotated_name)
                # NOTE: Does this copy the image data twice? Is opencv.imread better?
                # NOTE: Is using a buffer actually less efficient? I think it
                #       might be, so I'm not using it for now...
                #self.image_buffer[:, :, 0] = np.array(Image.open(full_image_name + "_0_e.png"))
                #self.image_buffer[:, :, 1] = np.array(Image.open(full_image_name + "_0_h.png"))
                image_buffer_e = np.array(Image.open(full_image_name + "_0_e.png"))
                image_buffer_h = np.array(Image.open(full_image_name + "_0_h.png"))
                # NOTE: It would be more efficient to save images on disk as
                #       floats, but not sure if it's worth the effort...
                for i in range(len(ind_list)):
                    ind = ind_list[i]
                    y = self.pixels[partition][self.perm[partition][start + ind], 0]
                    x = self.pixels[partition][self.perm[partition][start + ind], 1]
                    self.image_chunk[partition][ind, :, :, 0] =\
                        np.multiply(image_buffer_e[y-window_radius:y+window_radius+1,
                            x-window_radius:x+window_radius+1],
                                    1/255.0)
                    self.image_chunk[partition][ind, :, :, 1] =\
                        np.multiply(image_buffer_h[y-window_radius:y+window_radius+1,
                            x-window_radius:x+window_radius+1],
                                    1/255.0)
            print('Chunk loaded!')

        image_batch = self.image_chunk[partition][self.batch_size*self._num_in_chunk[partition]: self.batch_size*self._num_in_chunk[partition] + self.batch_size, :]
        # NOTE: This could be sped up - I think this creates an extra copy of
        #       the labels of the batch. It could be sped up by passing a
        #       buffer for labels to next_batch(). I'll try without and see how
        #       bad it is.
        label_batch = self.labels[partition][self.perm[partition][self.batch_size*self._num_in_epoch[partition]: self.batch_size*self._num_in_epoch[partition] + self.batch_size]]
        regress_batch = self.regress[partition][self.perm[partition][self.batch_size*self._num_in_epoch[partition]: self.batch_size*self._num_in_epoch[partition] + self.batch_size]]

        if self.is_regression:
            return image_batch, label_batch, regress_batch
        else:
            return image_batch, label_batch


class ImageOfNuclei(object):

    def __init__(self, image_filename, pixel_df, percent_train):
        image_name = os.path.split(image_filename)[1]
        self.image_name = image_name
        self.n_nuclei_total = max(pixel_df['Nucleus_ID'])
        #self.nuclei = [Nucleus() for i in range(self.n_nuclei)]
        self.nuclei = []
        # Nuclei are indexed starting at 1!
        for i in range(1, self.n_nuclei_total + 1):
            pixel_ind = pixel_df['Nucleus_ID'] == i
            self.nuclei.append(Nucleus(image_name, i, pixel_df.loc[pixel_ind]))
        self.labels = list(set(pixel_df['Class']))
        self.n_labels = len(self.labels)

        # Assign nuclei to training and validation sets
        nucleus_ids = np.arange(1, self.n_nuclei_total + 1)
        np.random.shuffle(nucleus_ids)
        self.n_nuclei = []
        self.n_nuclei.append(int(percent_train*self.n_nuclei_total))
        self.n_nuclei.append(self.n_nuclei_total - self.n_nuclei[0])
        self.nucleus_ids = []
        self.nucleus_ids.append(nucleus_ids[0:self.n_nuclei[0]])
        self.nucleus_ids.append(nucleus_ids[self.n_nuclei[0]:])
        non_nuclei_pixels = pixel_df.loc[pixel_df['Nucleus_ID'] == 0, ['Y', 'X']].values
        self.n_non_nuclei_total = non_nuclei_pixels.shape[0]
        # Non_nucleus pixels have label 0 (i.e. stromal pixels)
        self.n_non_nuclei = []
        self.n_non_nuclei.append(int(percent_train*self.n_non_nuclei_total))
        self.n_non_nuclei.append(self.n_non_nuclei_total - self.n_non_nuclei[0])
        perm = np.arange(self.n_non_nuclei_total)
        np.random.shuffle(perm)
        self.non_nuclei_pixels = []
        self.non_nuclei_pixels.append(non_nuclei_pixels[perm[0:self.n_non_nuclei[0]]])
        self.non_nuclei_pixels.append(non_nuclei_pixels[perm[self.n_non_nuclei[0]:]])
        image = Image.open(os.path.join(image_filename, '0_0_e.png'))
        self.shape = image.size

    def fill_rot_LUTs(self, rot_LUTs):
        self.rot_LUTs = rot_LUTs
        for i in range(0, self.n_nuclei_total):
            self.nuclei[i].fill_rot_LUTs(rot_LUTs)

    def get_random_nuclei_pixels(self, ar, image_name_list, n_nuclei, n_pixels, label, partition=0,
                                 repeat_nuclei_if_needed=True,
                                 repeat_pixels_if_needed=True):
        # NOTE: might consider taking in a list of labels instead of only one
        # NOTE: trying to do this copy in-place using ar

        #if n_pixels != rotations.shape[0]:
        #    raise ValueError('The number of requested pixels does not'
        #                     + ' match the number of entries in the '
        #                     + 'rotation array')
        if label == 0:
            if repeat_nuclei_if_needed:
                n_pixels = n_nuclei*n_pixels
                ind = choice_with_repeats(self.n_non_nuclei[partition],
                                          n_pixels)
            else:
                # Only return as many pixels as possible
                raise ValueError('Implementation can only handle set number '
                                 + 'of nuclei and therefore must repeat!')
                #n_pixels = min(n_nuclei*n_pixels, self.n_non_nuclei[partition])
                #ind = np.random.choice(self.n_non_nuclei[partition], n_pixels,
                #                       replace=False)
            rotations = np.random.choice(4, n_pixels)
            ar[:, 0] = self.rot_LUTs[1][rotations,
                                     self.non_nuclei_pixels[partition][ind, 0],
                                     self.non_nuclei_pixels[partition][ind, 1]]
            ar[:, 1] = self.rot_LUTs[0][rotations,
                                     self.non_nuclei_pixels[partition][ind, 0],
                                     self.non_nuclei_pixels[partition][ind, 1]]
            for r in range(n_pixels):
                image_name_list.append(os.path.join(self.image_name, rot_map[rotations[r]]))
            return
        else:
            if repeat_nuclei_if_needed:
                ind = choice_with_repeats(self.n_nuclei[partition],
                                          n_nuclei)
                nucleus_ids = self.nucleus_ids[partition][ind]
                for i in range(n_nuclei):
                    rotations = self.nuclei[nucleus_ids[i] - 1].\
                            get_random_pixels(ar[i*n_pixels: (i+1)*n_pixels],
                                              n_pixels, label,
                                              repeat_pixels_if_needed)
                    for r in range(n_pixels):
                        image_name_list.append(os.path.join(self.image_name, rot_map[rotations[r]]))
                return
            else:
                raise ValueError('Implementation can only handle set number '
                                 + 'of nuclei and therefore must repeat!')
                # Only return as many nuclei as possible
                #n_nuclei = min(self.n_nuclei[partition], n_nuclei)
                #ind = np.random.choice(self.n_nuclei[partition], n_nuclei,
                #                       replace=False)
                #nucleus_ids = self.nucleus_ids[partition][ind]

                #for i in range(n_nuclei):
                #    pixel_locations = np.zeros((n_nuclei*n_pixels, 2), dtype='int16')
                #for i in range(n_nuclei):
                #    pixel_locations.append(self.nuclei[nucleus_ids[i]].
                #                           get_random_pixels(n_pixels, label,
                #                                             repeat_pixels_if_needed))
                #return pixel_locations


class Nucleus(object):

    def __init__(self, image_name, nucleus_id, pixel_nucleus_df):
        # NOTE: Not sure if it matters pixel_locations are stored separately
        #       by label or if they should be stored together. But storing
        #       separately avoids the need to determine at each call to
        #       get_random_pixels() which pixels belong to which label
        self.image_name = image_name
        self.nucleus_id = nucleus_id
        # NOTE: I'm assuming that this is just a reference to the original
        #       rot_LUT and not a copy. If it's a copy, I'll have to pass
        #       it to get_random_pixels() to avoid copying.
        self.labels = list(set(pixel_nucleus_df['Class']))
        self.n_labels = len(self.labels)
        self.pixel_locations = []
        self.n_pixels_per_label = []
        for i in range(self.n_labels):
            ind = pixel_nucleus_df['Class'] == self.labels[i]
            self.n_pixels_per_label.append(np.sum(ind))
            self.pixel_locations.append(pixel_nucleus_df.loc[ind, ['Y', 'X']].values)

    def fill_rot_LUTs(self, rot_LUTs):
        self.rot_LUTs = rot_LUTs

    def get_random_pixels(self, ar, n_pixels, label,
                          repeat_pixels_if_needed=True):
        if label not in self.labels:
            raise ValueError('Label ' + str(label) + ' does not exist for nucleus '
                             + str(self.nucleus_id) + ' in image '
                             + self.image_name + '.')
        label_ind = self.labels.index(label)
        if repeat_pixels_if_needed:
            location_ind = choice_with_repeats(self.n_pixels_per_label[label_ind],
                                               n_pixels)
        else:
            raise ValueError('Implementation can only handle set number '
                             + 'of pixels and therefore must repeat!')
            # Only return as many pixels as possible
            #n_pixels = min(n_pixels, self.n_pixels_per_label[label_ind])
            #location_ind = np.random.choice(self.n_pixels_per_label[label_ind], n_pixels,
            #                                replace=False)
        rotations = np.random.choice(4, n_pixels)
        pixel_locations = self.pixel_locations[label_ind][location_ind, :]
        ar[:, 0] = self.rot_LUTs[1][rotations, pixel_locations[:, 0],
                                    pixel_locations[:, 1]]
        ar[:, 1] = self.rot_LUTs[0][rotations, pixel_locations[:, 0],
                                    pixel_locations[:, 1]]
        return rotations


def read_data_sets(data_dir, is_regression=False, percent_train=0.9):

    '''Read datasets.
    Reads all directoriees and creates the four lists.
    Shuffles the lists. And then picks training and validation
    datasets.
    '''
    # TODO: Later, I will want to take in a list of data_set_dirs, and for
    #       each data set, give the option to split into training/validation,
    #       or use solely for one or the other.
    '''
    NOTE: For the problem of nuclear segmentation, there are several senses
    of 'test' and 'validation'. We can test at pixel level (how well do we
    classify each patch?) and at the image level (what is the precision/recall
    score and how good is the boundary of each nucleus?). At the pixel level,
    we can test at the pixel level, the nuclear level, and the image level.
    Do we want to know how well the algorithm generalizes to other pixels,
    possibly within the same nucleus, or to other nuclei, possibly within the
    same image, or to other images? We want our code to provide modularity
    and generalizability to each of these different scenarios.

    Right now, we focus just on testing at the pixel level (for this code)
    and then at nucleus level. The pixel level is not that useful to know.
    Validating at the image level, allowing the user to set aside a set of
    images, perhaps from a separate data set, will be the next step.
    '''

#    class DataSets(object):
#        pass
#    data_sets = DataSets()
    img_dirs = glob.glob(os.path.join(data_dir, "*"))

#    training_data = {}
#    validation_data = {}
    # First split the image directories into training-validation and testing sets.

#    perm = np.arange(len(dirs))
#    np.random.shuffle(perm)
#    train_dirs = [dirs[perm[p]] for p in range(int((percent_train+percent_valid)*len(dirs)))]
#    test_dirs = [dirs[perm[p]] for p in range(int((percent_train+percent_valid)*len(dirs)), len(dirs))]

    images = []
    image_names = []
    summary_ar = np.zeros((len(img_dirs), 2), dtype='int32')

    for i, d in zip(range(len(img_dirs)), img_dirs):
        print(d)

        img_basename = os.path.split(d)[1]
        pixels_df = pd.read_pickle(os.path.join(d, "labels.pkl"))
#        n_patches = pixels_df.shape[0]
#        n_nuclei = max(pixels_df['Nucleus_ID'])
#
#        nucleus_ids = np.arange(1, n_nuclei + 1)
#        np.random.shuffle(nucleus_ids)
#
#        train_nucleus_ids = nucleus_ids[0:int(percent_train*n_nuclei)]
#        validate_nucleus_ids = nucleus_ids[int(percent_train*n_nuclei):]
#
        r_LUTs = create_rot_LUT(2000)
        img_instance = ImageOfNuclei(img_basename, pixels_df, r_LUTs, percent_train)
        images.append(img_instance)
        summary_ar[i, 0] = sum(img_instance.n_nuclei)
        summary_ar[i, 1] = sum(img_instance.n_non_nuclei)
        image_names.append(img_instance.image_name)

    labels_list = list(set(pixels_df['Class']))
    data_set_summary = pd.DataFrame(data=summary_ar, index=image_names,
                                    columns=['n_nuclei', 'n_non_nuclei'])
    data_set = DataSet(data_dir, images, data_set_summary, labels_list,
                       is_regression=is_regression)
    return data_set


#        train_image_dict = {'stroma': []}
#        valid_image_dict = {'stroma': []}
#
#        # the above excludes the negative stroma examples we add.
#
#
#        #train_nuclei_id = perm[:int((percent_train/(percent_train+percent_valid))*n_nuclei)]
#        #valid_nuclei_id = perm[int((percent_train/(percent_train+percent_valid))*n_nuclei):]
#
#        for _ in xrange(n_patches):
#            if( in train_nuclei_id.tolist()):
#                if (int(str_line[1]) not in train_image_dict.keys()):
#                    train_image_dict[int(str_line[1])] = {'pos': [], 'neg':[], 'int':[]}
#                current_nuclei_id = int(str_line[1])
#                if (str_line[2] is '3'):
#                    train_image_dict[current_nuclei_id]['pos'].append(str_line)
#                elif (str_line[2] is '2'):
#                    train_image_dict[current_nuclei_id]['int'].append(str_line)
#                elif (str_line[2] is '1'):
#                    train_image_dict[current_nuclei_id]['neg'].append(str_line)
#            elif (int(str_line[1]) in valid_nuclei_id.tolist()):
#                if (int(str_line[1]) not in valid_image_dict.keys()):
#                    valid_image_dict[int(str_line[1])] = {'pos': [], 'neg':[], 'int':[]}
#                current_nuclei_id = int(str_line[1])
#                if (str_line[2] is '3'):
#                    valid_image_dict[current_nuclei_id]['pos'].append(str_line)
#                elif (str_line[2] is '2'):
#                    valid_image_dict[current_nuclei_id]['int'].append(str_line)
#                elif (str_line[2] is '1'):
#                    valid_image_dict[current_nuclei_id]['neg'].append(str_line)
#            else:
#                valid_train_toss = np.random.choice([0,1], size =(1,), p=[(percent_train/(percent_train+percent_valid)), (percent_valid/(percent_train+percent_valid))])
#                if(valid_train_toss==0):
#                    train_image_dict['stroma'].append(str_line)
#                else:
#                    valid_image_dict['stroma'].append(str_line)
#
#        training_data[d] = train_image_dict
#        validation_data[d] = valid_image_dict
#
#    data_sets.train = DataSet(training_data)
#    data_sets.validation = DataSet(validation_data) #valid_pos_lines, valid_neg_lines, valid_on_lines, valid_stroma_lines)
#
#    print('Datsets returned')
#    return data_sets
