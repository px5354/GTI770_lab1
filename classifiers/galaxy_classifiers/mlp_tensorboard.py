#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # X - Lab's name

Students :
    Names — Permanent Code

Group :
    GTI770-H18-0X
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import mnist
import os


class MLPClassifierTensorBoard(object):
    def __init__(self, train_path, batch_size, image_size, learning_rate, dropout_probability,
                 number_of_steps, number_of_classes, number_of_channels, number_of_hidden_layer):

        """ Initialize the default parameters of a Multi-Layer Perceptron.

         Args:
            number_of_classes: The number of class the problem has.
            batch_size: The desired mini-batch size.
            image_size: The number of pixels in one dimension the image has (must be a square image).
            number_of_steps: The number of learning steps to run the training.
            learning_rate: The desired learning rate.
            train_path: The path in which the TensorBoard data will be saved.
        """

        self.number_of_classes = number_of_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.number_of_channels = number_of_channels
        self.number_of_steps = number_of_steps
        self.dropout_probability = dropout_probability
        self.learning_rate = learning_rate
        self.number_of_hidden_layer = number_of_hidden_layer
        self.train_path = train_path
        self.display_step = 1
        self.index = 0

    # def train_old(self, dataset):
    #     with tf.Session(graph=tf.Graph()) as sess:
    #
    #         # Do not apply softmax activation yet, see below.
    #         y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
    #
    #         with tf.name_scope("input"):
    #             X = tf.placeholder("float", [None, 74], name="X")
    #             y_ = tf.placeholder("float", [None, self.number_of_classes], name="y_ground_truth")
    #             keep_prob = tf.placeholder(tf.float32, name="dropout_keep_probability")
    #
    #         with tf.name_scope('cross_entropy'):
    #             with tf.name_scope('total'):
    #                 cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    #         tf.summary.scalar('cross_entropy', cross_entropy)
    #
    #         with tf.name_scope('train'):
    #             train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(
    #                 cross_entropy)
    #
    #         with tf.name_scope('accuracy'):
    #             with tf.name_scope('correct_prediction'):
    #                 correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #             with tf.name_scope('accuracy'):
    #                 accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #         tf.summary.scalar('accuracy', accuracy)
    #
    #         # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    #         merged = tf.summary.merge_all()
    #         train_writer = tf.summary.FileWriter(self.train_path + '/train',
    #                                              sess.graph)
    #         test_writer = tf.summary.FileWriter(self.train_path + '/test')
    #
    #         # Training loop
    #         for i in range(self.number_of_steps):
    #                 if i % 10 == 0:  # Record summaries and test-set accuracy
    #                     dict = self.feed_dict(X, y_, keep_prob, False)
    #                     merged = tf.summary.merge_all()
    #                     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #                     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    #                     # summary, train_accuracy, train_scores, train_prediction = sess.run(
    #                     #     [merged, accuracy, scores, predictions], feed_dict=dict)
    #
    #                     summary, train_accuracy = sess.run(
    #                         [merged, accuracy], feed_dict=dict)
    #
    #
    #
    #                     print('Accuracy at step %s: %s' % (i, train_accuracy))
    #
    #                 else:  # Record train set summaries, and train
    #                     if i % 100 == 99:  # Record execution stats
    #                         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #                         run_metadata = tf.RunMetadata()
    #                         summary, _ = sess.run([merged, train_step],
    #                                               feed_dict=self.feed_dict(X, y_, keep_prob, True),
    #                                               options=run_options,
    #                                               run_metadata=run_metadata)
    #                         train_writer.add_run_metadata(run_metadata, '-step%04d' % i)
    #                         train_writer.add_summary(summary, i)
    #                         print('Adding run metadata for', i)
    #
    #                     else:  # Record a summary
    #                         summary, _ = sess.run([merged, train_step], feed_dict=self.feed_dict(X, y_, keep_prob, True))
    #                         train_writer.add_summary(summary, i)
    #
    #         saver.save(sess, os.environ["VIRTUAL_ENV"] + "/data/models/exports/MLP/my_mlp/my_mlp_test")
    #
    #         # Build the signature_def_map.
    #         classification_inputs = tf.saved_model.utils.build_tensor_info(X)
    #         classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
    #         classification_outputs_scores = tf.saved_model.utils.build_tensor_info(scores)
    #         classification_signature = (
    #             tf.saved_model.signature_def_utils.build_signature_def(
    #                 inputs={
    #                     tf.saved_model.signature_constants.CLASSIFY_INPUTS:
    #                         classification_inputs
    #                 },
    #                 outputs={
    #                     tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
    #                         classification_outputs_classes,
    #                     tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
    #                         classification_outputs_scores
    #                 },
    #                 method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
    #         tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
    #         tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
    #
    #         prediction_signature = (
    #             tf.saved_model.signature_def_utils.build_signature_def(
    #                 inputs={'images': tensor_info_x},
    #                 outputs={'classes': tensor_info_y},
    #                 method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    #
    #         legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    #         builder.add_meta_graph_and_variables(
    #             sess, [tf.saved_model.tag_constants.TRAINING],
    #             signature_def_map={
    #                 'predict_images':
    #                     prediction_signature,
    #                 tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #                     classification_signature,
    #             },
    #             legacy_init_op=legacy_init_op)
    #
    #         # Export model.
    #         builder.save()
    #         print("Model saved and exported.")
    #
    #         train_writer.close()
    #         test_writer.close()
    #
    # def feed_dict(self, x, y_, keep_prob, train):
    #     """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    #     if train:
    #         xs, ys = mnist.train.next_batch(self.batch_size)
    #         k = self.dropout_probability
    #     else:
    #         xs, ys = mnist.test.images, mnist.test.labels
    #         k = 1.0
    #     return {x: xs, y_: ys, keep_prob: k}
    #
    # def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    #     """Reusable code for making a simple neural net layer.
    #
    #     It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    #     It also sets up name scoping so that the resultant graph is easy to read,
    #     and adds a number of summary ops.
    #     """
    #     # Adding a name scope ensures logical grouping of the layers in the graph.
    #     with tf.name_scope(layer_name):
    #         # This Variable will hold the state of the weights for the layer
    #         with tf.name_scope('weights'):
    #             weights = weight_variable([input_dim, output_dim])
    #             self.variable_summaries(weights)
    #         with tf.name_scope('biases'):
    #             biases = bias_variable([output_dim])
    #             variable_summaries(biases)
    #         with tf.name_scope('Wx_plus_b'):
    #             preactivate = tf.matmul(input_tensor, weights) + biases
    #             tf.summary.histogram('pre_activations', preactivate)
    #         activations = act(preactivate, name='activation')
    #         tf.summary.histogram('activations', activations)
    #         return activations
    #
    # def variable_summaries(var):
    #     """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    #     with tf.name_scope('summaries'):
    #         mean = tf.reduce_mean(var)
    #         tf.summary.scalar('mean', mean)
    #         with tf.name_scope('stddev'):
    #             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #         tf.summary.scalar('stddev', stddev)
    #         tf.summary.scalar('max', tf.reduce_max(var))
    #         tf.summary.scalar('min', tf.reduce_min(var))
    #         tf.summary.histogram('histogram', var)
    #
    # def weight_variable(shape):
    #     """Create a weight variable with appropriate initialization."""
    #     initial = tf.truncated_normal(shape, stddev=0.1)
    #     return tf.Variable(initial)
    #
    # def bias_variable(shape):
    #     """Create a bias variable with appropriate initialization."""
    #     initial = tf.constant(0.1, shape=shape)

    def train(self, dataset):

        with tf.Session(graph=tf.Graph()) as sess:

            with tf.name_scope("input"):
                X = tf.placeholder("float", [None, self.image_size], name="X")
                y_ = tf.placeholder("int64", [None], name="y_ground_truth")
                keep_prob = tf.placeholder(tf.float32, name="dropout_keep_probability")

            # We can't initialize these variables to 0 - the network will get stuck.
            def weight_variable(shape):
                """Create a weight variable with appropriate initialization."""
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial)

            def bias_variable(shape):
                """Create a bias variable with appropriate initialization."""
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)

            def variable_summaries(var):
                """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
                with tf.name_scope('summaries'):
                    mean = tf.reduce_mean(var)
                    tf.summary.scalar('mean', mean)
                    with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                    tf.summary.scalar('stddev', stddev)
                    tf.summary.scalar('max', tf.reduce_max(var))
                    tf.summary.scalar('min', tf.reduce_min(var))
                    tf.summary.histogram('histogram', var)

            def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
                """Reusable code for making a simple neural net layer.
                It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
                It also sets up name scoping so that the resultant graph is easy to read,
                and adds a number of summary ops.
                """
                # Adding a name scope ensures logical grouping of the layers in the graph.
                with tf.name_scope(layer_name):
                    # This Variable will hold the state of the weights for the layer
                    with tf.name_scope('weights'):
                        weights = weight_variable([input_dim, output_dim])
                        variable_summaries(weights)
                    with tf.name_scope('biases'):
                        biases = bias_variable([output_dim])
                        variable_summaries(biases)
                    with tf.name_scope('Wx_plus_b'):
                        preactivate = tf.matmul(input_tensor, weights) + biases
                        tf.summary.histogram('pre_activations', preactivate)
                    activations = act(preactivate, name='activation')
                    tf.summary.histogram('activations', activations)
                    return activations

            hidden1 = nn_layer(X, self.image_size, self.number_of_channels, 'layer1')
            dropped1 = tf.nn.dropout(hidden1, self.dropout_probability)
            hidden2 = nn_layer(dropped1, self.number_of_channels, self.number_of_channels, 'layer2')
            dropped2 = tf.nn.dropout(hidden2, self.dropout_probability)

            if self.number_of_hidden_layer == 3:
                hidden3 = nn_layer(dropped2, self.number_of_channels, self.number_of_channels, 'layer3')
                dropped3 = tf.nn.dropout(hidden3, self.dropout_probability)
            if self.number_of_hidden_layer == 4:
                hidden3 = nn_layer(dropped2, self.number_of_channels, self.number_of_channels, 'layer3')
                dropped3 = tf.nn.dropout(hidden3, self.dropout_probability)
                hidden4 = nn_layer(dropped3, self.number_of_channels, self.number_of_channels, 'layer4')
                dropped4 = tf.nn.dropout(hidden4, self.dropout_probability)
            if self.number_of_hidden_layer == 5:
                hidden3 = nn_layer(dropped2, self.number_of_channels, self.number_of_channels, 'layer3')
                dropped3 = tf.nn.dropout(hidden3, self.dropout_probability)
                hidden4 = nn_layer(dropped3, self.number_of_channels, self.number_of_channels, 'layer4')
                dropped4 = tf.nn.dropout(hidden4, self.dropout_probability)
                hidden5 = nn_layer(dropped4, self.number_of_channels, self.number_of_channels, 'layer5')
                dropped5 = tf.nn.dropout(hidden5, self.dropout_probability)


            # Do not apply softmax activation yet, see below.
            y = nn_layer(dropped2, self.number_of_channels, self.number_of_classes, 'output', act=tf.identity)
            if self.number_of_hidden_layer == 3:
                y = nn_layer(dropped3, self.number_of_channels, self.number_of_classes, 'output', act=tf.identity)
            if self.number_of_hidden_layer == 4:
                y = nn_layer(dropped4, self.number_of_channels, self.number_of_classes, 'output', act=tf.identity)
            if self.number_of_hidden_layer == 5:
                y = nn_layer(dropped5, self.number_of_channels, self.number_of_classes, 'output', act=tf.identity)

            scores, indices = tf.nn.top_k(y)
            indices = tf.cast(indices, tf.int64)

            with tf.name_scope('cross_entropy'):
                # The raw formulation of cross-entropy,
                #
                # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
                #                               reduction_indices=[1]))
                #
                # can be numerically unstable.
                #
                # So here we use tf.losses.sparse_softmax_cross_entropy on the
                # raw logit outputs of the nn_layer above, and then average across
                # the batch.
                with tf.name_scope('total'):
                    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                        labels=y_, logits=y)
            tf.summary.scalar('cross_entropy', cross_entropy)

            with tf.name_scope('train'):
                train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    cross_entropy)

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    predictions = tf.equal(tf.argmax(y, 1), y_)
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

            # Merge all the summaries and write them out to
            # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.train_path + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(self.train_path + '/test')
            tf.global_variables_initializer().run()

            # Train the model, and also write summaries.
            # Every 10th step, measure test-set accuracy, and write test summaries
            # All other steps, run train_step on training data, & add training summaries

            def feed_dict(x, y_, keep_prob, train):
                """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
                if train:
                    xs, ys = dataset.train.next_feature_batch(self.batch_size)
                    k = self.dropout_probability
                else:
                    xs, ys = dataset.valid.next_feature_batch(self.batch_size)
                    k = 1.0
                return {x: xs, y_: ys, keep_prob: k}

            # Training loop
            for i in range(self.number_of_steps):
                if i % 10 == 0:  # Record summaries and test-set accuracy
                    dict = feed_dict(X, y_, keep_prob, False)
                    summary, train_accuracy, train_scores, train_prediction = sess.run(
                        [merged, accuracy, scores, predictions], feed_dict=dict)
                    print('Accuracy at step %s: %s' % (i, train_accuracy))

                else:  # Record train set summaries, and train
                    if i % 100 == 99:  # Record execution stats
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, _ = sess.run([merged, train_step],
                                              feed_dict= feed_dict(X, y_, keep_prob, True),
                                              options=run_options,
                                              run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, '-step%04d' % i)
                        train_writer.add_summary(summary, i)
                        print('Adding run metadata for', i)

                    else:  # Record a summary
                        summary, _ = sess.run([merged, train_step], feed_dict= feed_dict(X, y_, keep_prob, True))
                        train_writer.add_summary(summary, i)

            saver = tf.train.Saver()
            saver.save(sess, os.environ["VIRTUAL_ENV"] + "/data/models/exports/MLP/my_mlp/my_mlp_test")

            mapping_string = tf.constant(["0", "1", "2"])
            prediction_classes = tf.contrib.lookup.index_to_string(indices, mapping=mapping_string)

            # Build the signature_def_map.
            classification_inputs = tf.saved_model.utils.build_tensor_info(X)
            classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
            classification_outputs_scores = tf.saved_model.utils.build_tensor_info(scores)
            classification_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                            classification_inputs
                    },
                    outputs={
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                            classification_outputs_classes,
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                            classification_outputs_scores
                    },
                    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
            tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_x},
                    outputs={'classes': tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            export_dir = self.train_path + "builder"
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.TRAINING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                legacy_init_op=legacy_init_op)

            # Export model.
            builder.save()
            print("Model saved and exported.")

            train_writer.close()
            test_writer.close()
