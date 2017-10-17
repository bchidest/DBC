
"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import tensorflow as tf
import numpy as np
from numpy import matlib
from scipy.misc import imsave
from scipy.misc import imread
from scipy.misc import imresize
from PIL import Image
import matplotlib.pyplot as plt

import nuclei_stride
import read_masks_data
import cnn_params

import time
import os
import sys
import glob

from six.moves import xrange  # pylint: disable=redefined-builtin

sys.path.insert(0, os.path.abspath('./processing'))
import he_processing as he


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 51, 51, 2))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, keep_prob, is_training, images_pl, labels_pl,
                   regres_pl, keep_prob_pl, is_training_pl,
                   partition):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    if regres_pl is not None:
        images_feed, labels_feed, regres_feed = data_set.next_batch(partition)
        feed_dict = {
            images_pl: images_feed,
            labels_pl: labels_feed,
            regres_pl: regres_feed,
            keep_prob_pl: keep_prob,
            is_training_pl: is_training,
        }
    else:
        images_feed, labels_feed = data_set.next_batch(partition)
        feed_dict = {
            images_pl: images_feed,
            labels_pl: labels_feed,
            keep_prob_pl: keep_prob,
            is_training_pl: is_training,
        }
    return feed_dict


def predict_full(model_filename, patch_dir, output_dir, param_filename, param_ind, img_file_rule="*.png"):
    #print(patch_dir)
    #print(os.path.join(patch_dir, img_file_rule))
    img_filenames = glob.glob(os.path.join(patch_dir, img_file_rule))
    #print(img_filenames)
    img = np.array(Image.open(img_filenames[0]))
    M, N, D = img.shape

    param_list = cnn_params.load_params(param_filename)
    params = param_list[param_ind]
    with tf.Graph().as_default():
        keep_prob_placeholder = tf.placeholder(tf.float32)
        images_placeholder, labels_placeholder = placeholder_inputs(5000)
        #conv_shape_placeholder = tf.placeholder(tf.float32)
        #conv = nuclei_stride.conv_layer(images_placeholder)
        #conv_reshape = nuclei_stride.reshape_conv_output(conv)
        #logits = nuclei_stride.fully_connected_layers(conv_reshape,
        #                                              keep_prob_placeholder)
        logits = nuclei_stride.inference(images_placeholder, params, 51)

        saver = tf.train.Saver()

        sess = tf.Session()

        saver.restore(sess, model_filename)
        save_value = model_filename.split('-')[-1]
        print(save_value)
        for img_filename in img_filenames:
            print("Predicting image file: " + img_filename)
            a = predict_sample_full(sess, logits, images_placeholder,
                           keep_prob_placeholder,
                           img_filename, output_dir, save_value)
        return a


def predict_sample_full(sess, logits, img_placeholder,
                   keep_prob_placeholder, img_filename,
                   output_dir, save_value , patch_radius=25, prob_map=False, upsample=False):
    batch_size = 5000
    patch_size = 2*patch_radius + 1

    #probabs = nuclei_stride.prob(logits)
    pred_filename = "nuclei_pixels_" + os.path.split(os.path.splitext(img_filename)[0])[1] + save_value + ".png"
    #prob_filename = os.path.split(os.path.splitext(img_filename)[0])[1] + "_tf_prob.png"

    img_crop = np.array(Image.open(img_filename))
    img_crop = img_crop[500:1500, 500:1500, :]
    if (upsample):
        img_crop = imresize(img_crop, 200)
    M_orig, N_orig, _ = img_crop.shape
    img_crop = np.lib.pad(img_crop, [(patch_radius, patch_radius),
                                     (patch_radius, patch_radius),
                                     (0, 0)], 'symmetric')
    _, sample_h, sample_e = he.stain_normalization(img_crop)
    M, N = sample_h.shape
    sample = np.zeros((1, M, N, 2))
    sample[0, :, :, 0] = sample_h
    sample[0, :, :, 1] = sample_e
    #pred = np.zeros((M_orig, N_orig), dtype='uint8')
    patches = np.zeros((batch_size, 51, 51, 2), dtype='float')
    p = np.zeros((M_orig*N_orig), dtype=np.float32)
    #prob = np.zeros((M_orig*N_orig, 2))

    k = 0
    l = 0
    for i in xrange(M - 2*patch_radius):
        print('Row ' + str(i))
        for j in xrange(N - 2*patch_radius):
           # print('Rescaling intensities')
            patches[k, :, :, :] = sample[0, i:i+51, j:j+51, :] * (1.0 / 255.0)
            if (k + 1) % batch_size == 0:
                feed_dict = {
                    img_placeholder: patches,
                    keep_prob_placeholder: 1.0
                }
                #prob[l:l+batch_size], p[l:l+batch_size] = sess.run([probabs, logits], feed_dict=feed_dict)
                #p[l:l+batch_size] = sess.run(logits, feed_dict=feed_dict)[:, 0]
                logits_eval = sess.run(logits, feed_dict=feed_dict)[:, 0]
                p[l:l+batch_size] = logits_eval
                k = 0
                l = l + batch_size
            else:
                k += 1
    if k != 0:
        feed_dict = {
            img_placeholder: patches,
            keep_prob_placeholder: 1.0
        }
        #temp_p, temp = sess.run([probabs, logits], feed_dict=feed_dict)
        temp = sess.run(logits, feed_dict=feed_dict)
        p[l:] = temp[0][0:k]
        #prob[l:] = temp_p[0:k]
    #prob = softmax(p)
    #prob = prob[:, 1]
    #prob = (255*np.reshape(prob, (M_orig, N_orig))).astype("uint8")
    #print(p)
    #p = np.argmax(p, axis=1)
    p = (np.reshape(255*p, (M_orig, N_orig))).astype("uint8")
    print(p)
    #pred = p[patch_radius:-patch_radius, patch_radius:-patch_radius]

    if (upsample):
        p = imresize(p, 50, interp="nearest")
    plt.imshow(p)
    plt.show()
    (Image.fromarray(p)).save(os.path.join(output_dir, pred_filename))
    #Image.fromarray(40*p).save(os.path.join(output_dir, os.path.splitext(pred_filename)[0] + "_scaled.png"))
    #imsave(os.path.join(output_dir, prob_filename), prob)
    return p


def predict(img, model_filename, param_filename, param_ind=0, patch_radius=25,
            unmix=True):
    M_orig, N_orig, D = img.shape
    img_pad = np.lib.pad(img, [(patch_radius, patch_radius),
                               (patch_radius, patch_radius),
                               (0, 0)], 'symmetric')
    if unmix:
        _, sample_h, sample_e = he.stain_normalization(img_pad)
        M, N = sample_h.shape
        sample = np.zeros((1, M, N, 2), dtype='float')
        sample[0, :, :, 1] = sample_h / 255.0
        sample[0, :, :, 0] = sample_e / 255.0
        print('Sample unmixed...')
    else:
        sample = img_pad

    p = np.zeros((M_orig, N_orig), dtype='float32')

    param_list = cnn_params.load_params(param_filename)
    params = param_list[param_ind]
    #with tf.Graph().as_default():
    #    images_placeholder = tf.placeholder(tf.float32)
    #    is_training_placeholder = tf.placeholder(tf.bool)

    #    # Predict non-efficient
    #    logits2 = nuclei_stride.inference(images_placeholder, params,
    #                                     2*patch_radius + 1,
    #                                     is_training_placeholder)
    #    saver = tf.train.Saver()
    #    sess = tf.Session()
    #    saver.restore(sess, model_filename)
    #    sample2 = np.zeros((2, 51, 51, 2), dtype='float')
    #    sample2[0, :, :, :] = sample[:, 0:51, 0:51, :]
    #    sample2[1, :, :, :] = sample[:, 0:51, 1:52, :]

    #    y_hat, t4, t5 = sess.run(logits2, feed_dict={
    #                           images_placeholder: sample2,
    #                           is_training_placeholder: False})

    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(tf.float32)
        is_training_placeholder = tf.placeholder(tf.bool)
        # Predict efficient
        logits = nuclei_stride.inference_efficient(images_placeholder, params,
                                                   2*patch_radius + 1,
                                                   is_training_placeholder)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, model_filename)
        y_hat, t, t2, t3 = sess.run(logits, feed_dict={
                                images_placeholder: sample,
                                is_training_placeholder: False})
        total_stride = int(np.sqrt(len(y_hat)))
        m = int(M_orig / total_stride)
        n = int(N_orig / total_stride)
        n_strides = int(np.log2(total_stride))
        for i in range(total_stride):
            for j in range(total_stride):
                i_prime = i
                j_prime = j
                ind = 0
                for x in reversed(range(0, n_strides)):
                    a = (i_prime % 2)*(2**x)**2
                    b = (j_prime % 2)*(2**x)**2
                    ind = ind + 2*a + b
                    i_prime = int((i_prime - i_prime % 2) / 2)
                    j_prime = int((j_prime - j_prime % 2) / 2)
                p[i::total_stride, j::total_stride] = y_hat[ind].reshape((m, n))

    #return p, t, t2, t3, t4, t5
    return p


def predict_directory(img_dir, img_out_dir, model_filename, param_filename,
                      param_ind=0, patch_radius=25, unmix=True,
                      img_rule='*.png',
                      img_out_suffix='_nuclear_regression.png'):
    img_filename_list = glob.glob(os.path.join(img_dir, img_rule))
    t = np.zeros(len(img_filename_list))
    for i, img_filename in zip(range(len(img_filename_list)), img_filename_list):
        print('Predicting image ' + img_filename)
        img = np.array(Image.open(img_filename))[0:1000, 0:1000, :]
        img_out_filename = os.path.join(img_out_dir,
                os.path.split(os.path.splitext(img_filename)[0])[1] + img_out_suffix)
        t0 = time.time()
        img_p = predict(img, model_filename, param_filename, param_ind,
                        patch_radius, unmix)
        t1 = time.time()
        t[i] = t1 - t0
        print(t1 - t0)
        Image.fromarray((img_p * 255).astype('uint8')).save(img_out_filename)
    print('Average time = %f' % np.mean(t[1:]))


def do_eval(sess,
            evaluation,
            images_placeholder,
            labels_placeholder,
            regres_placeholder,
            keep_prob_placeholder,
            is_training_placeholder,
            data_set,
            batch_size, num_steps_training,
            is_training,
            partition,
            n_bins=[], n_classes=[]):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        evaluation: The Tensor that returns the number of correct predictions.
        mages_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    sq_error_hist = [np.zeros((n_bins,), dtype='int64') for i in range(n_classes)]
    steps_per_epoch = data_set._num_examples[partition] // batch_size
    num_examples = steps_per_epoch * batch_size
    #for step in xrange(int(num_steps_training / 2)):
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   1.0,
                                   is_training,
                                   images_placeholder,
                                   labels_placeholder,
                                   regres_placeholder,
                                   keep_prob_placeholder,
                                   is_training_placeholder,
                                   partition)
        if regres_placeholder is not None:
            temp_hist = sess.run(evaluation, feed_dict=feed_dict)
            for i in range(n_classes):
                sq_error_hist[i] += temp_hist[i]
        else:
            true_count += sess.run(evaluation, feed_dict=feed_dict)
#    precision = true_count / num_examples
    if regres_placeholder is not None:
        for i in range(n_classes):
            print(sq_error_hist[i])
        return sq_error_hist
    else:
        print(true_count)
        print(step * batch_size)
        precision = true_count * 1. / (step * batch_size)
        return precision


def run_training(param_filename, model_dir, summary_dir, train_dir,
                 sq_diff_loss=True, n_bins=10):

    filename_prefix = os.path.splitext(os.path.split(param_filename)[1])[0]
    data_set = read_masks_data.DataSet(train_dir, is_regression=sq_diff_loss)
    n_classes = data_set.n_labels
    param_list = cnn_params.load_params(param_filename)

    for params in param_list:
        with tf.Graph().as_default():
            # TODO: CHANGE THIS!!!!!!!!!!!!!
            global_step = tf.Variable(0, trainable=False)
            keep_prob_placeholder = tf.placeholder(tf.float32)
            is_training_placeholder = tf.placeholder(tf.bool)
            images_placeholder, labels_placeholder, =\
                placeholder_inputs(params.batch_size)
            logits = nuclei_stride.inference(images_placeholder, params, 51,
                                             is_training_placeholder)

            if sq_diff_loss:
                regres_placeholder = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
                #regres_placeholder = tf.placeholder(tf.float32, shape=(params.batch_size))
                loss = nuclei_stride.loss_sq_diff(logits[0], regres_placeholder)
                evaluation = nuclei_stride.evaluation_sq_error(logits[0], labels_placeholder,
                                                               regres_placeholder, n_classes, n_bins, params.batch_size)
            else:
                regres_placeholder = None
                loss = nuclei_stride.loss(logits, labels_placeholder)
                evaluation = nuclei_stride.evaluation(logits, labels_placeholder)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = nuclei_stride.training(loss, global_step,
                                                  params.learning_rate)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter(
                    os.path.join(summary_dir, filename_prefix +
                                 str(params.param_file_line_number)),
                    sess.graph)

            for step in xrange(params.max_steps):
                start_time = time.time()
                feed_dict = fill_feed_dict(data_set,
                                           1.0,
                                           True,
                                           images_placeholder,
                                           labels_placeholder,
                                           regres_placeholder,
                                           keep_prob_placeholder,
                                           is_training_placeholder,
                                           partition=0)
                #print(feed_dict[images_placeholder])
                #print(feed_dict[labels_placeholder])

                #y = sess.run(logits, feed_dict=feed_dict)
                #print(y)
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                duration = time.time() - start_time

                if np.isnan(loss_value):
                    print(loss_value)
                    print('Model diverged with loss = NaN')
                    return

                #if (step + 1) % 500 == 0:
                if (step + 1) % 10 == 0:
                    num_examples_per_step = params.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                          examples_per_sec, sec_per_batch))

                    y, a, b = sess.run(logits, feed_dict=feed_dict)
                    #print(feed_dict[images_placeholder][0:3, :, :, :])
                    #print(feed_dict[images_placeholder].shape)
                    print(b)
                    print(a)
                    for i in range(20):
                        print(y[i], feed_dict[regres_placeholder][i], feed_dict[labels_placeholder][i])
                        #print(y[1][i][0])
                        #print(y[2][i])
                        #print(y[1][i])
                        #print(feed_dict[images_placeholder][i,0:2,0:2,1])
                        #plt.imshow(feed_dict[images_placeholder][i,:,:,1], cmap=plt.cm.gray)
                        #plt.show()
                    #print(np.mean((y - feed_dict[regres_placeholder])**2))

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    saver.save(sess, os.path.join(model_dir, filename_prefix +
                                                  str(params.param_file_line_number)),
                               global_step=step)

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % (3*3600) == 0 or (step + 1) == params.max_steps:
                    saver.save(sess, os.path.join(model_dir, filename_prefix +
                                                  str(params.param_file_line_number)),
                               global_step=step)
                    # Evaluate against the training set.
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    precision = do_eval(sess,
                                        evaluation,
                                        images_placeholder,
                                        labels_placeholder,
                                        regres_placeholder,
                                        keep_prob_placeholder,
                                        is_training_placeholder,
                                        data_set,
                                        params.batch_size, 2000,
                                        is_training=False,
                                        partition=1,
                                        n_bins=n_bins,
                                        n_classes=n_classes)
                    if not sq_diff_loss:
                        print("Step: " + str(step) + ' vld accuracy now is %0.04f' % (precision))
#                    if(precision<0.50):
#                       print('ERROR!!'#)
#                       break

                if (step + 1) % (3*3600) == 0 or (step + 1) == params.max_steps:
                    print('Training Data Eval:')
                    precision = do_eval(sess,
                                        evaluation,
                                        images_placeholder,
                                        labels_placeholder,
                                        regres_placeholder,
                                        keep_prob_placeholder,
                                        is_training_placeholder,
                                        data_set,
                                        params.batch_size, 2000,
                                        is_training=False,
                                        partition=0,
                                        n_bins=n_bins,
                                        n_classes=n_classes)
                    if not sq_diff_loss:
                        print("Step: " + str(step) + ' trn accuracy now is %0.04f' % (precision))


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()

