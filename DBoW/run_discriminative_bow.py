import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix
import random
from PIL import Image
import cv2

import os
from glob import glob
import time
from datetime import datetime
import cPickle as pkl

import discriminative_bow as dbow
import bow_dataset


def fill_feed_dict(data_set, sample_pl, labels_pl, batch_size, zero_pad=0,
                   repeats=True):
    '''
    Fill the feed_dict.

    zero_pad : int
        Last zero_pad elements of batch_size will be zero.
    repeats : bool
        Fetch batch from repeated epoch or not.
    '''
    if zero_pad > 0:
        if repeats:
            sample_feed, labels_feed_temp = data_set.next_batch(batch_size)
        else:
            sample_feed, labels_feed_temp = data_set.next_batch_no_repeats(batch_size)
        for i in range(zero_pad):
            sample_feed.append(np.zeros((1, data_set.n_features), dtype='float32'))
        labels_feed = np.zeros((batch_size + zero_pad,), dtype='uint8')
        labels_feed[0: batch_size] = labels_feed_temp
    else:
        if repeats:
            sample_feed, labels_feed = data_set.next_batch(batch_size)
        else:
            sample_feed, labels_feed = data_set.next_batch_no_repeats(batch_size)
    feed_dict = {tuple(sample_pl): tuple(sample_feed)}
    feed_dict.update({labels_pl: labels_feed})
    return feed_dict


def do_eval(sess,
            logits,
            nuclei_placeholder,
            labels_placeholder,
            data_set,
            batch_size,
            repeats=False,
            threshold_offset=0.0):
    '''
    Evaluate the model on a data set.

    repeats : bool
        Fetch batch from repeated epoch or not.
    threshold_offset : float [-1.0, 0.0]
        Threshold for trade-off of FPR and FNR.
    '''
    if repeats:
        data_set.reset_epoch()
        num_examples = data_set.num_examples_in_epoch
    else:
        data_set.reset_no_repeats()
        num_examples = data_set.num_examples
    steps_per_epoch = num_examples // batch_size
    remainder_step = num_examples % batch_size

    y_hat = np.zeros((num_examples,), dtype='uint8')
    y = np.zeros((num_examples,), dtype='uint8')
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   nuclei_placeholder,
                                   labels_placeholder,
                                   batch_size,
                                   repeats=repeats)

        logits_eval = sess.run(logits, feed_dict=feed_dict)
        y_hat[step*batch_size: (step+1)*batch_size] = \
            (logits_eval[:, 0] < (logits_eval[:, 1] + threshold_offset)).astype('uint8')
        y[step*batch_size: (step+1)*batch_size] = feed_dict[labels_placeholder]
    feed_dict = fill_feed_dict(data_set,
                               nuclei_placeholder,
                               labels_placeholder,
                               batch_size=remainder_step,
                               zero_pad=batch_size-remainder_step,
                               repeats=repeats)
    logits_eval = sess.run(logits, feed_dict=feed_dict)
    y_hat[(step+1)*batch_size:] = \
        (logits_eval[:, 0] < (logits_eval[:, 1] + threshold_offset)).astype('uint8')[0:remainder_step]
    y[(step+1)*batch_size:] = feed_dict[labels_placeholder][0:remainder_step]

    C = confusion_matrix(y, y_hat, labels=[0, 1])
    precision = C[1, 1] / float(C[1, 1] + C[0, 1])
    recall = C[1, 1] / float(C[1, 1] + C[1, 0])
    f1 = 2*(precision * recall) / (precision + recall)
    specificity = C[0, 0] / float(C[0, 0] + C[0, 1])
    accuracy = (recall + specificity) / 2
    fpr = C[0, 1] / float(C[0, 0] + C[0, 1])
    print(C)
    print('Weighted accuracy = %.3f\n' % accuracy)
    print('               F1 = %.3f' % f1)
    print('        Precision = %.3f' % precision)
    print('           Recall = %.3f' % recall)
    print('      Specificity = %.3f' % specificity)
    print('              FPR = %.3f' % fpr)
    return([accuracy, f1, precision, recall, specificity, fpr])


def run_training(filename_prefix, model_dir, summary_dir, train,
                 validate,
                 n_codewords=8, n_nodes_codeword=[25, 15], n_nodes_bow=6,
                 learning_rate=0.001, max_steps=5000,
                 batch_size=30):

    n_features = train.n_features
    print("Number of training patients = ("
          + str(train.n_samples_per_label[0]) + ", "
          + str(train.n_samples_per_label[1]) + ")")
    print("Number of validation patients = ("
          + str(validate.n_samples_per_label[0]) + ", "
          + str(validate.n_samples_per_label[1]) + ")")

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        nuclei_placeholder = [tf.placeholder(tf.float32) for _ in range(batch_size)]
        labels_placeholder = tf.placeholder(tf.int32)

        with tf.variable_scope("codeword_network") as scope:
            codewords1 = dbow.codeword_representation(nuclei_placeholder,
                                                     0,
                                                     n_features,
                                                     n_codewords,
                                                     n_nodes_codeword)
            bow = tf.expand_dims(dbow.bow(codewords1), axis=0)
            scope.reuse_variables()
            for i in range(1, batch_size):
                codewords = dbow.codeword_representation(nuclei_placeholder,
                                                         i,
                                                         n_features,
                                                         n_codewords,
                                                         n_nodes_codeword)
                bow = tf.concat([bow, tf.expand_dims(dbow.bow(codewords), axis=0)], axis=0)
        # Normalize BOW by maximum number of objects for any sample
        bow = bow / float(train.max_n_objects)
        with tf.name_scope('bow'):
            tf.summary.histogram('activations', bow)
        with tf.variable_scope("bow_classifier") as scope:
            logits = dbow.bow_classifier(bow, train.n_labels,
                                         n_codewords, n_nodes_bow)
            #logits = dbow.bow_classifier_simple_2_class(bow)
        loss = dbow.loss(logits, labels_placeholder)
        train_op = dbow.training(loss, global_step, learning_rate)

        saver = tf.train.Saver(max_to_keep=0)
        summary_op = tf.summary.merge_all()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(
                os.path.join(summary_dir, filename_prefix),
                sess.graph)

        for step in xrange(max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(train,
                                       nuclei_placeholder,
                                       labels_placeholder,
                                       batch_size)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if np.isnan(loss_value):
                print(loss_value)
                print('Model diverged with loss = NaN')
                return

            if (step + 1) % 2 == 0:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                      examples_per_sec, sec_per_batch))

                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 10 == 0 or (step + 1) == max_steps:
                # Evaluate against the validation set.
                print('Train Data Eval:')
                do_eval(sess,
                        logits,
                        nuclei_placeholder,
                        labels_placeholder,
                        train,
                        batch_size)
                print("Step: " + str(step))
            if (step + 1) % 10 == 0 or (step + 1) == max_steps:
                # Evaluate against the training set.
                print('Validate Data Eval:')
                do_eval(sess,
                        logits,
                        nuclei_placeholder,
                        labels_placeholder,
                        validate,
                        batch_size)
                print("Step: " + str(step))
            if (step + 1) % 20 == 0 or (step + 1) == max_steps:
                saver.save(sess, os.path.join(model_dir, filename_prefix),
                           global_step=step)

            if (step + 1) % 100 == 0 or (step + 1) == max_steps:
                bow_eval, y = sess.run([bow, logits], feed_dict=feed_dict)
                c_eval = sess.run(codewords1, feed_dict=feed_dict)
                print("Example Codewords Representations:")
                for i in range(10):
                    print(c_eval[i, :], feed_dict[tuple(nuclei_placeholder)][0][i, 0])
                print("Example BOW:")
                for i in range(5):
                    print(bow_eval[i], feed_dict[labels_placeholder][i])
                    print(y[i], feed_dict[labels_placeholder][i])


def predict_samples(model_filename, samples, max_n_objects,
                    n_codewords=8, n_nodes_codeword=[25, 15],
                    n_nodes_bow=6,
                    threshold_offset=0.0,
                    n_labels=2):
    '''
    samples : list of numpy.arrays
    '''
    n_samples = len(samples)
    n_features = samples[0].shape[1]

    with tf.Graph().as_default():
        nuclei_placeholder = [tf.placeholder(tf.float32) for _ in range(n_samples)]
        with tf.variable_scope("codeword_network") as scope:
            codewords1 = dbow.codeword_representation(nuclei_placeholder,
                                                     0,
                                                     n_features,
                                                     n_codewords,
                                                     n_nodes_codeword)
            bow = tf.expand_dims(dbow.bow(codewords1), axis=0)
            scope.reuse_variables()
            for i in range(1, n_samples):
                codewords = dbow.codeword_representation(nuclei_placeholder,
                                                         i,
                                                         n_features,
                                                         n_codewords,
                                                         n_nodes_codeword)
                bow = tf.concat([bow, tf.expand_dims(dbow.bow(codewords), axis=0)], axis=0)
        # Normalize BOW by maximum number of objects for any sample
        bow = bow / float(max_n_objects)
        with tf.variable_scope("bow_classifier") as scope:
            logits = dbow.bow_classifier(bow, n_labels,
                                         n_codewords, n_nodes_bow)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, model_filename)
        feed_dict = {tuple(nuclei_placeholder): tuple(samples)}
        logits_eval = sess.run(logits, feed_dict=feed_dict)
        y_hat = np.argmax(logits_eval, axis=1)
        return y_hat


def evaluate_on_dataset(model_filename, data_set, max_n_objects,
                        n_codewords=8, n_nodes_codeword=[25, 15],
                        n_nodes_bow=6, batch_size=30, repeats=False,
                        threshold_offset=0.0):
    n_features = data_set.n_features
    print("Number of samples = ("
          + str(data_set.n_samples_per_label[0]) + ", "
          + str(data_set.n_samples_per_label[1]) + ")")

    with tf.Graph().as_default():
        nuclei_placeholder = [tf.placeholder(tf.float32) for _ in range(batch_size)]
        labels_placeholder = tf.placeholder(tf.int32)
        with tf.variable_scope("codeword_network") as scope:
            codewords1 = dbow.codeword_representation(nuclei_placeholder,
                                                     0,
                                                     n_features,
                                                     n_codewords,
                                                     n_nodes_codeword)
            bow = tf.expand_dims(dbow.bow(codewords1), axis=0)
            scope.reuse_variables()
            for i in range(1, batch_size):
                codewords = dbow.codeword_representation(nuclei_placeholder,
                                                         i,
                                                         n_features,
                                                         n_codewords,
                                                         n_nodes_codeword)
                bow = tf.concat([bow, tf.expand_dims(dbow.bow(codewords), axis=0)], axis=0)
        # Normalize BOW by maximum number of objects for any sample
        bow = bow / float(max_n_objects)
        with tf.variable_scope("bow_classifier") as scope:
            logits = dbow.bow_classifier(bow, data_set.n_labels,
                                         n_codewords, n_nodes_bow)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, model_filename)
        print('DataSet Eval:')
        return do_eval(sess,
                       logits,
                       nuclei_placeholder,
                       labels_placeholder,
                       data_set,
                       batch_size,
                       repeats,
                       threshold_offset)


def evaluate_ROC(model_filename, data_set, max_n_objects,
                 output_dir='./',
                 n_codewords=8, n_nodes_codeword=[25, 15],
                 n_nodes_bow=6, batch_size=30, repeats=False,
                 thresholds=[-1.0, 0.0, 1.0]):

    tpr = [0.0]
    fpr = [0.0]
    for t in thresholds:
        metrics = evaluate_on_dataset(model_filename, data_set, max_n_objects,
                                      n_codewords=n_codewords,
                                      n_nodes_codeword=n_nodes_codeword,
                                      n_nodes_bow=n_nodes_bow,
                                      batch_size=batch_size, repeats=repeats,
                                      threshold_offset=t)
        tpr.append(metrics[3])
        fpr.append(metrics[5])
    tpr.append(1.0)
    fpr.append(1.0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0.0, 1.0], [0.0, 1.0], 'b--', fpr, tpr, 'r')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('False Positive Rate', fontsize=20)
    tick = list(np.arange(0.0, 1.1, 0.2))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    ax.set_xticklabels([str(t) for t in tick], fontsize=16)
    ax.set_yticklabels([str(t) for t in tick], fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    fig.savefig(os.path.join(output_dir, 'ROC.png'))
    fig.clf()


def generate_example_codewords(data_dir, model_filename, data_set, max_n_objects,
                               n_codewords=8, n_nodes_codeword=[25, 15],
                               n_nodes_bow=6, n_example_nuclei_per_sample=5,
                               n_example_nuclei=10, n_example_bow_per_label=5,
                               save_data=False):
    n_features = data_set.n_features
    print("Number of patients = ("
          + str(data_set.n_samples_per_label[0]) + ", "
          + str(data_set.n_samples_per_label[1]) + ")")

    max_codeword_count = 0
    bow_dataset = np.zeros((data_set.num_examples, n_codewords), dtype='float32')
    labels_ind = [[], []]
    y_hat = np.zeros((data_set.num_examples,), dtype='uint8')
    with tf.Graph().as_default():
        nuclei_placeholder = [tf.placeholder(tf.float32)]
        labels_placeholder = tf.placeholder(tf.int32)
        with tf.variable_scope("codeword_network") as scope:
            codewords = dbow.codeword_representation(nuclei_placeholder,
                                                     0,
                                                     n_features,
                                                     n_codewords,
                                                     n_nodes_codeword)
            bow = tf.expand_dims(dbow.bow(codewords), axis=0)
            bow = bow / float(max_n_objects)
        with tf.variable_scope("bow_classifier") as scope:
            # Normalize BOW by maximum number of objects for any sample
            logits = dbow.bow_classifier(bow, data_set.n_labels,
                                         n_codewords, n_nodes_bow)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, model_filename)
        nuclei = [[] for _ in range(n_codewords)]
        for i in range(data_set.num_examples):
            feed_dict = fill_feed_dict(data_set,
                                       nuclei_placeholder,
                                       labels_placeholder,
                                       batch_size=1,
                                       repeats=False)
            logits_eval, bow_dataset[i, :], codewords_eval =\
                sess.run([logits, bow, codewords], feed_dict=feed_dict)
            codeword_ind = np.argmax(codewords_eval, axis=1)
            for k in range(n_codewords):
                nuclei_ind = np.where(codeword_ind == k)[0]
                if len(nuclei_ind) > n_example_nuclei_per_sample:
                    nuclei_ind = random.sample(nuclei_ind, n_example_nuclei_per_sample)
                for j in range(len(nuclei_ind)):
                    nuclei[k].append((i, nuclei_ind[j]))
            if (feed_dict[labels_placeholder][0] == 0):
                labels_ind[0].append(i)
            else:
                labels_ind[1].append(i)
            y_hat[i] = np.argmax(logits_eval)
    random.shuffle(labels_ind[0])
    random.shuffle(labels_ind[1])

    y_hat_out = []
    y_hat_out.append(y_hat[labels_ind[0][0:n_example_bow_per_label]])
    y_hat_out.append(y_hat[labels_ind[1][0:n_example_bow_per_label]])
    bow_out = []
    bow_out.append(bow_dataset[labels_ind[0][0:n_example_bow_per_label]])
    bow_out.append(bow_dataset[labels_ind[1][0:n_example_bow_per_label]])
    nuclei_out = []
    pids = []
    pids.append([data_set.pids[labels_ind[0][i]] for i in range(n_example_bow_per_label)])
    pids.append([data_set.pids[labels_ind[1][i]] for i in range(n_example_bow_per_label)])
    for k in range(n_codewords):
        if nuclei[k]:
            if len(nuclei[k]) > n_example_nuclei:
                perm = range(len(nuclei[k]))
                random.shuffle(perm)
                nuclei_out.append([nuclei[k][perm[i]] for i in range(n_example_nuclei)])
            else:
                nuclei_out.append([nuclei[k][i] for i in range(len(nuclei[k]))])
        else:
            nuclei_out.append([])
    if save_data:
        f_out = open('Nuclei_examples.plk', 'wb')
        pkl.dump([nuclei_out, y_hat_out, bow_out], f_out)
        f_out.close()

    # Determine Y lim for BoW histograms
    for i in range(n_example_bow_per_label):
        for j in range(2):
            temp_max_count = np.max(bow_out[j][i, :])
            if temp_max_count > max_codeword_count:
                max_codeword_count = temp_max_count
    # Round to the nearest 5
    max_codeword_count = (int(100*max_codeword_count) / 5 + 1)*0.05
    # Plot BoW histograms
    fig, axarr = plt.subplots(n_example_bow_per_label, 6, figsize=(11, 6))
    colors = [(255, 139, 139), (139, 139, 255)]
    for i in range(n_example_bow_per_label):
        for j in range(2):
            pos = np.arange(n_codewords) + 0.5
            axarr[i, 3*j+1].bar(pos, bow_out[j][i, :], align='center')
            axarr[i, 3*j+1].set_ylim([0.0, max_codeword_count])
            axarr[i, 3*j+1].set_yticks(list(np.arange(0, max_codeword_count + 0.05, 0.05)))
            axarr[i, 3*j+1].set_yticklabels(['0.00'] + ['']*(int(100*max_codeword_count)/5-1)
                    + ['%.2f' % max_codeword_count], fontsize=8)
            axarr[i, 3*j+1].set_xticks(pos)
            axarr[i, 3*j+1].set_xticklabels(range(1, n_codewords + 1), fontsize=8)
            patch_full_filename = os.path.join('/home/ben/datasets/Nuclei_features/TCGA_BRCA/images/bchidest_tmp/drive_1/TCGA_15_patches', pids[j][i]+'*DX1*', 'patch*.png')
            patch_full_filename = glob(patch_full_filename)[0]
            patch_image = np.array(Image.open(patch_full_filename))[1250:1750, 1250:1750]
            axarr[i, 3*j].imshow(patch_image)
            axarr[i, 3*j].set_xticklabels([])
            axarr[i, 3*j].set_yticklabels([])
            axarr[i, 3*j].grid(False)
            axarr[i, 3*j].axis('off')
            circle_patch = 255*np.ones((450, 450, 3), dtype='uint8')
            cv2.circle(circle_patch, (225, 225), 180, colors[y_hat_out[j][i]], thickness=-1)
            cv2.circle(circle_patch, (225, 225), 180, colors[j], thickness=50)
            axarr[i, 3*j+2].imshow(circle_patch)
            axarr[i, 3*j+2].set_xticklabels([])
            axarr[i, 3*j+2].set_yticklabels([])
            axarr[i, 3*j+2].grid(False)
            axarr[i, 3*j+2].axis('off')
    axarr[0, 1].set_title('Non-Basal')
    axarr[0, 4].set_title('Basal')
    axarr[n_example_bow_per_label - 1, 1].set_xlabel('Codeword')
    axarr[n_example_bow_per_label - 1, 4].set_xlabel('Codeword')
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    fig.savefig('paper/figures/bar.png', bbox_inches='tight', pad_inches=0.0)
    fig.clf()

    # Plot example nuclei codewords
    n_i = 5
    n_j = int(np.ceil(n_example_nuclei / n_i))
    for k in range(n_codewords):
        fig = plt.figure()
        for i in range(len(nuclei_out[k])):
            plt_i = i % n_i
            plt_j = i / n_i
            a = fig.add_subplot(n_i, n_j, i+1)
            temp_pid = data_set.pids[nuclei_out[k][i][0]]
            feature_filename = os.path.join(data_dir, temp_pid + '-01_tf_Nuclei.csv')
            feature_df = pd.read_csv(feature_filename)
            # Need to offset by 1!
            image_number = int(feature_df.loc[nuclei_out[k][i][1], 'ImageNumber']) - 1
            x = int(feature_df.loc[nuclei_out[k][i][1], 'AreaShape_Center_X'])
            y = int(feature_df.loc[nuclei_out[k][i][1], 'AreaShape_Center_Y'])
            image_filename = os.path.join(data_dir, temp_pid + '-01_tf_Image.csv')
            image_df = pd.read_csv(image_filename)
            patch_filename = os.path.split(image_df.loc[image_number, 'URL_patch_orig'])[1]
            #patch_full_filename = temp_pid + '/' + patch_filename
            patch_full_filename = os.path.join('/home/ben/datasets/Nuclei_features/TCGA_BRCA/images/bchidest_tmp/drive_1/TCGA_15_patches', temp_pid +'*DX1*', patch_filename)
            patch_full_filename = glob(patch_full_filename)[0]
            print(patch_full_filename, x, y)
            patch_image = np.array(Image.open(patch_full_filename))
            nucleus_patch = patch_image[1000+y-25:1000+y+25, 1000+x-25:1000+x+25]
            imgplot = plt.imshow(nucleus_patch)
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.grid(False)
            a.axis('off')
        if nuclei_out[k]:
            imgplot.set_clim(0.0, 0.7)
            fig.subplots_adjust(wspace=-.8)
        plt.suptitle('Codeword ' + str(k+1), fontsize=20)
        fig.savefig('paper/figures/nuclear_words/nuclear_words_' + str(k+1) + '.png', bbox_inches='tight', pad_inches=0.1)
        fig.clf()

    return nuclei_out, y_hat_out, bow_out, pids


def run_fake_network(batch_size=30):
    random.seed(33)

    train, test = bow_dataset.load_fake_data_sets()
    n_features = train.n_features
    print("Number of training patients = ("
          + str(train.n_samples_per_label[0]) + ", "
          + str(train.n_samples_per_label[1]) + ")")
    print("Number of testing patients = ("
          + str(test.n_samples_per_label[0]) + ", "
          + str(test.n_samples_per_label[1]) + ")")
    with tf.Graph().as_default():
        nuclei_placeholder = [tf.placeholder(tf.float32) for _ in range(batch_size)]
        labels_placeholder = tf.placeholder(tf.int32)

        codewords = dbow.codeword_representation_fake(nuclei_placeholder,
                                                      0,
                                                      n_features,
                                                      0.3, 0.6)
        bow = tf.expand_dims(dbow.bow(codewords), axis=0)
        for i in range(1, batch_size):
            codewords = dbow.codeword_representation_fake(nuclei_placeholder,
                                                          i,
                                                          n_features,
                                                          0.3, 0.6)
            bow = tf.concat([bow, tf.expand_dims(dbow.bow(codewords), axis=0)], axis=0)
        logits = dbow.bow_classifier_fake(bow)
        sess = tf.Session()

        for i in range(6):
            feed_dict = fill_feed_dict(train,
                                       nuclei_placeholder,
                                       labels_placeholder,
                                       batch_size)
            y = sess.run(logits, feed_dict=feed_dict)
            C = confusion_matrix(feed_dict[labels_placeholder], y)
            print(C)
