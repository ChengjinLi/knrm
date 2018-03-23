#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2018年03月23日

@author: lichengjin
"""
import tensorflow as tf
import numpy as np
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
    Unicode,
)
import sys
import time
import subprocess
import argparse
from traitlets.config.loader import PyFileConfigLoader
from deeplearning4ir.match_NN import ClickNN

reload(sys)
sys.setdefaultencoding('UTF8')


class SentCnnKnrm(ClickNN):
    lamb = Float(1.0 / 2, help="guassian sigma = lamb * bin_size").tag(config=True)
    neg_sample = Int(1, help='negative sample').tag(config=True)
    use_exact = Bool(True, help='include exact match bins').tag(config=True)
    emb_in = Unicode('None', help="embedding in").tag(config=True)
    n_filter = Int(128, help="number of bigram cnn filters").tag(config=True)
    h_max = Int(2, help="max ngrams").tag(config=True)
    use_term_weight = Bool(False, help="learn weight for each doc term").tag(config=True)
    term_weight_in = Unicode('None', help="Initial term weight").tag(config=True)
    window_size = Int(10, help="window size for AND , OR gate").tag(config=True)
    norm_doc_len = Bool(False).tag(config=True)
    cross = Bool(True).tag(config=True)
    drop_out = Float(0).tag(config=True)

    def __init__(self, **kwargs):
        super(SentCnnKnrm, self).__init__(**kwargs)

        print self.gate_type
        print self.use_exact
        self.mus = SentCnnKnrm.kernal_mus(self.n_bins, use_exact=self.use_exact)
        self.sigmas = SentCnnKnrm.kernel_sigmas(self.n_bins, self.lamb, self.use_exact)
        print self.sigmas

        print self.emb_in
        if self.emb_in != 'None':
            self.emb = self.load_word2vec(self.emb_in)
            self.embeddings = tf.Variable(
                tf.constant(self.emb, dtype='float32', shape=[self.vocabulary_size + 1, self.embedding_size]))
            print "Initialized embeddings with {0}".format(self.emb_in)
        else:
            self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size + 1, self.embedding_size], -1.0, 1.0))

        # Model parameters for feedfoward rank NN
        if self.cross:  # 不同粒度交叉匹配，所以是self.h_max * self.h_max
            self.total_bins = self.n_bins * self.h_max * self.h_max
        else:   # 相同粒度匹配，所以是self.h_max
            self.total_bins = self.n_bins * self.h_max
        self.W1 = SentCnnKnrm.weight_variable([self.total_bins, 1])
        self.b1 = tf.Variable(tf.zeros([1]))

        # Model parameters for convolutions
        self.Ws = []
        self.bs = []
        self.filter_shapes = []
        for h in range(1, self.h_max + 1):
            with tf.name_scope("conv-{0}-gram".format(h)):
                filter_shape = [h, self.embedding_size, 1, self.n_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv-W-{0}".format(h))
                b = tf.Variable(tf.constant(0.1, shape=[self.n_filter]), name="conv-b-{0}".format(h))
                self.filter_shapes.append(filter_shape)
                self.Ws.append(W)
                self.bs.append(b)

    def load_word2vec(self, emb_file_path):
        emb = np.random.uniform(low=-1, high=1, size=(self.vocabulary_size + 1, self.embedding_size))
        nlines = 0
        with open(emb_file_path) as f:
            for line in f:
                nlines += 1
                if nlines == 1:
                    continue
                items = line.split()
                tid = int(items[0])
                vec = np.array([float(t) for t in items[1:]])
                emb[tid, :] = vec
                if nlines % 20000 == 0:
                    print "load {0} vectors...".format(nlines)
        return emb

    def gen_doc_mask(self, Q, D):
        M = self.gen_mask(Q, D, use_exact=self.use_exact)
        if self.norm_doc_len:
            doc_len = np.sum(M[:, :, 50:], 2, keepdims=True) + 1
            title_len = np.sum(M[:, :, :50], 2, keepdims=True) + 1
            M[:, :, 50:] = M[:, :, 50:] / doc_len
            M[:, :, 0:50] = M[:, :, 0:50] / title_len
        return M

    def model(self, inputs_q, inputs_d, doc_mask, idf, mu, sigma, train=True):
        """
        The pointwise model graph
        :return: return the score predicted for each document in the batch
        """
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        q_embed = tf.nn.embedding_lookup(self.embeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(self.embeddings, inputs_d, name='demb')

        kdes = []
        title_kdes = []
        sims = []
        q_embed_expanded = tf.expand_dims(q_embed, -1)
        d_embed_expanded = tf.expand_dims(d_embed, -1)
        idf_expanded = idf
        q_cnn_emb_list = []
        d_cnn_emb_list = []
        idf_list = []

        for h in range(1, self.h_max + 1):
            with tf.name_scope("conv-{0}-gram".format(h)):
                q_conv = tf.nn.conv2d(q_embed_expanded, self.Ws[h - 1], strides=[1, 1, 1, 1], padding='VALID',
                                      name='conv')
                # [batch_size, len - h + 1, 1, n_filters]
                q_h = tf.nn.relu(tf.nn.bias_add(q_conv, self.bs[h - 1]), name='relu') + 0.000000001

                d_conv = tf.nn.conv2d(d_embed_expanded, self.Ws[h - 1], strides=[1, 1, 1, 1], padding='VALID',
                                      name='conv')
                # [batch_size, len - h + 1, 1, n_filters]
                d_h = tf.nn.relu(tf.nn.bias_add(d_conv, self.bs[h - 1])) + 0.000000001
                # q_h = q_embed
                # d_h = d_embed
                # [batch_size, len - h + 1, n_filters]
                q_h = tf.squeeze(q_h)
                d_h = tf.squeeze(d_h)
                # q_h = tf.reshape(q_h, [self.batch_size, self.max_q_len - h + 1, self.n_filter])
                # d_h = tf.reshape(d_h, [self.batch_size, self.max_d_len - h + 1, self.n_filter])

                normalized_q_embed = tf.nn.l2_normalize(q_h, 2)
                normalized_d_embed = tf.nn.l2_normalize(d_h, 2)

                q_cnn_emb_list.append(normalized_q_embed)
                d_cnn_emb_list.append(normalized_d_embed)

        for h1_dx, q_emb in enumerate(q_cnn_emb_list):
            for h2_dx, d_emb in enumerate(d_cnn_emb_list):
                if h1_dx != h2_dx and not self.cross:
                    continue

                # similarity matrix [batch_size, len - h1 + 1, len - h2 + 1]
                # tmp = tf.transpose(d_emb, perm=[0, 2, 1])
                # sim = tf.batch_matmul(q_emb, tmp, name='conv_similarity_matrix')
                sim = tf.matmul(q_emb, d_emb, transpose_b=True, name='conv_similarity_matrix')

                print sim.get_shape()
                sims.append(sim)

                h1 = h1_dx + 1
                h2 = h2_dx + 1
                # compute gaussian kernel
                # this is because CNN padding="VALID",
                # so h2-gram document length = doc_len - h2 + 1, h1-gram query length = q_len - h1 + 1
                # An alternative is to use 'SAME' padding, making length stay the same
                # rs_sim = tf.reshape(sim, [self.batch_size, self.max_q_len - h1 + 1, self.max_d_len - h2 + 1, 1])
                rs_sim = tf.expand_dims(sim, -1)

                # ignore exact match
                tmp = tf.exp(-tf.square(tf.subtract(rs_sim, mu)) / (tf.matmul(tf.square(sigma), 2)))
                tmp = tmp * (doc_mask[:, :self.max_q_len - (h1 - 1), :self.max_d_len - (h2 - 1), :])
                if self.drop_out > 0 and train:
                    tmp = tf.nn.dropout(tmp, 1 - self.drop_out)
                print tmp.get_shape()

                kde = tf.reduce_sum(tmp, [2])
                kde = tf.log(tf.maximum(kde, 1e-10)) * 0.01  # scale the data
                print kde.get_shape()
                aggregated_kde = tf.reduce_sum(kde * (idf[:, :self.max_q_len - (h1 - 1), :]), [1])
                print aggregated_kde.get_shape()
                kdes.append(aggregated_kde)  # [h^2, batch, n_bins]

        kde_tmp = tf.concat(kdes, 1)  # [batch, n_bins * h^2 ]
        print kde_tmp.get_shape()
        all_kde_tmp = kde_tmp
        kde_flat = tf.reshape(all_kde_tmp, [self.batch_size, self.total_bins])
        o = tf.tanh(tf.matmul(kde_flat, self.W1) + self.b1)
        print o.get_shape()

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print(total_parameters)
        return (sims, kde_flat), o

    def train(self, train_pair_file_path, val_pair_file_path, train_size, checkpoint_dir, load_model=False,
              test_point_file_path=None, test_size=0):

        # PLACEHOLDERS
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.

        # nodes to hold mu sigma
        input_mu = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_mu')
        input_sigma = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_sigma')

        # nodes to hold query and qterm idf. padding terms will have idf=0
        train_inputs_q = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len], name='train_inputs_q')
        train_input_idf = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len], name='idf')

        # nodes to hold training data, postive samples
        train_inputs_pos_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len],
                                            name='train_inputs_pos_d')

        # nodes to hold negative samples
        train_inputs_neg_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len])

        # mask padding terms
        # assume all docid >= 1
        # padding with 0
        # also mask out exact match
        input_train_mask_pos = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])
        input_train_mask_neg = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])

        # reshape place holders
        mu = tf.reshape(input_mu, shape=[1, 1, self.n_bins])
        sigma = tf.reshape(input_sigma, shape=[1, 1, self.n_bins])
        rs_train_doc_mask_pos = tf.reshape(input_train_mask_pos, [self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_train_doc_mask_neg = tf.reshape(input_train_mask_neg, [self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_idf = tf.reshape(train_input_idf, shape=[self.batch_size, self.max_q_len, 1])

        # training graph
        mid_res_pos, o_pos = self.model(train_inputs_q, train_inputs_pos_d, rs_train_doc_mask_pos, rs_idf, mu, sigma)
        mid_res_pos_test, o_pos_test = self.model(train_inputs_q, train_inputs_pos_d, rs_train_doc_mask_pos, rs_idf, mu,
                                                  sigma, False)
        _, o_neg = self.model(train_inputs_q, train_inputs_neg_d, rs_train_doc_mask_neg, rs_idf, mu, sigma)
        _, o_neg_test = self.model(train_inputs_q, train_inputs_neg_d, rs_train_doc_mask_neg, rs_idf, mu, sigma, False)
        loss = tf.reduce_mean(tf.maximum(0.0, 1 - o_pos + o_neg))
        loss_test = tf.reduce_mean(tf.maximum(0.0, 1 - o_pos_test + o_neg_test))

        # optimizer
        # lr = 0.001
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        lr = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=0.00001).minimize(loss)

        # Create a local session to run the training.

        with tf.Session() as sess:

            saver = tf.train.Saver()
            start_time = time.time()

            # Run all the initializers to prepare the trainable parameters.
            if not load_model:
                print "Initialize variables..."
                tf.initialize_all_variables().run()
                print('Initialized!')

            else:
                p = checkpoint_dir + 'model.ckpt'
                if True:
                    # saver.restore(sess, ckpt.model_checkpoint_path)
                    saver.restore(sess, p)
                    print "model loaded!"
                else:
                    print "no model found"
                    exit(-1)

            # Loop through training steps.
            step = 0
            for epoch in range(10000):
                pair_stream = open(train_pair_file_path)
                # first 10k lines are used as validation
                # for ldx in range(10000):
                #    pair_stream.readline()
                train_l = 0
                for BATCH in self.data_generator.pairwise_reader(pair_stream, self.batch_size, with_idf=True,
                                                                 or_window=-1):
                    step += 1
                    X, Y = BATCH
                    M_pos = self.gen_doc_mask(X[u'q'], X[u'd'])
                    M_neg = self.gen_doc_mask(X[u'q'], X[u'd_aux'])

                    if X[u'idf'].shape[0] != self.batch_size:
                        continue
                    train_feed_dict = {train_inputs_q: self.re_pad(X[u'q'], self.batch_size),
                                       train_inputs_pos_d: self.re_pad(X[u'd'], self.batch_size),
                                       train_inputs_neg_d: self.re_pad(X[u'd_aux'], self.batch_size),
                                       train_input_idf: self.re_pad(X[u'idf'], self.batch_size),
                                       input_mu: self.mus,
                                       input_sigma: self.sigmas,
                                       input_train_mask_pos: M_pos,
                                       input_train_mask_neg: M_neg}

                    # Run the graph and fetch some of the nodes.
                    _, l = sess.run([optimizer, loss], feed_dict=train_feed_dict)
                    # print step, l
                    train_l += l

                    if step % self.eval_frequency == 0:
                        print "training loss: %.3f" % (train_l / self.eval_frequency)
                        train_l = 0
                        val_l = 0
                        n_val_batch = 0
                        # w = sess.run(self.W1)
                        # print w
                        # p = sess.run(self.Proj)
                        # print p
                        val_pair_stream = open(val_pair_file_path)
                        for BATCH in self.val_data_generator.pairwise_reader(val_pair_stream, self.batch_size,
                                                                             with_idf=True, or_window=-1):
                            if BATCH is None:
                                break
                            X_val, Y_val = BATCH
                            M_pos = self.gen_mask_with_gate(X_val[u'q'], X_val[u'd'])
                            M_neg = self.gen_mask_with_gate(X_val[u'q'], X_val[u'd_aux'])
                            val_feed_dict = {train_inputs_q: self.re_pad(X_val[u'q'], self.batch_size),
                                             train_inputs_pos_d: self.re_pad(X_val[u'd'], self.batch_size),
                                             train_inputs_neg_d: self.re_pad(X_val[u'd_aux'], self.batch_size),
                                             train_input_idf: self.re_pad(X_val[u'idf'], self.batch_size),
                                             input_mu: self.mus,
                                             input_sigma: self.sigmas,
                                             input_train_mask_pos: M_pos,
                                             input_train_mask_neg: M_neg}
                            l = sess.run(loss_test, feed_dict=val_feed_dict)
                            val_l += l
                            n_val_batch += 1
                        val_pair_stream.close()
                        val_l /= n_val_batch
                        print('validation loss: %.3f' % (val_l))

                        # output evaluations
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        print('Step %d (epoch %.2f), %.1f ms per step' % (step,
                                                                          float(step) * self.batch_size / (
                                                                                      train_size * self.neg_sample),
                                                                          1000 * elapsed_time / self.eval_frequency))
                        sys.stdout.flush()

                    # save model
                    if (step + 1) % self.checkpoint_steps == 0:
                        saver.save(sess, checkpoint_dir + '/model.ckpt')
                        if float(step) * self.batch_size / (train_size * self.neg_sample) >= self.max_epochs:
                            break

                # END epoch
                pair_stream.close()
                # saver.save(sess, checkpoint_dir + '/model.ckpt'.format(epoch))

            # end training
            saver.save(sess, checkpoint_dir + '/model.ckpt')

    def test(self, test_point_file_path, test_size, output_file_path, checkpoint_dir=None, load_model=False):

        # PLACEHOLDERS
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.

        # nodes to hold mu sigma
        input_mu = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_mu')
        input_sigma = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_sigma')

        # nodes to hold query and qterm idf. padding terms will have idf=0
        test_inputs_q = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len], name='test_inputs_q')
        test_input_idf = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len], name='idf')

        # nodes to hold test data
        test_inputs_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len],
                                       name='test_inputs_pos_d')

        # mask padding terms
        # assume all docid >= 1
        # assume padded with 0
        test_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])

        # reshape place holders
        mu = tf.reshape(input_mu, shape=[1, 1, self.n_bins])
        sigma = tf.reshape(input_sigma, shape=[1, 1, self.n_bins])
        rs_test_mask = tf.reshape(test_mask, [self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_idf = tf.reshape(test_input_idf, shape=[self.batch_size, self.max_q_len, 1])

        # training graph
        inter_res, o = self.model(test_inputs_q, test_inputs_d, rs_test_mask, rs_idf, mu, sigma)

        # Create a local session to run the testing.

        with tf.Session() as sess:
            test_point_stream = open(test_point_file_path)
            outfile = open(output_file_path, 'w')
            saver = tf.train.Saver()

            if load_model:
                # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                # if ckpt and ckpt.model_checkpoint_path:
                p = checkpoint_dir + 'model.ckpt'
                if True:
                    # saver.restore(sess, ckpt.model_checkpoint_path)
                    saver.restore(sess, p)
                    print "model loaded!"
                else:
                    print "no model found"
                    exit(-1)
            else:
                tf.initialize_all_variables().run()
            weights = sess.run(self.W1)
            print weights

            # Loop through training steps.
            for b in range(int(np.ceil(float(test_size) / self.batch_size))):
                X, Y = next(
                    self.test_data_generator.pointwise_generate(test_point_stream, self.batch_size, with_idf=True,
                                                                with_label=False, or_window=-1))
                M = self.gen_mask_with_gate(X[u'q'], X[u'd'])
                test_feed_dict = {test_inputs_q: self.re_pad(X[u'q'], self.batch_size),
                                  test_inputs_d: self.re_pad(X[u'd'], self.batch_size),
                                  test_input_idf: self.re_pad(X[u'idf'], self.batch_size),
                                  input_mu: self.mus,
                                  input_sigma: self.sigmas,
                                  test_mask: M}

                # Run the graph and fetch some of the nodes.
                scores = sess.run(o, feed_dict=test_feed_dict)

                for score in scores:
                    outfile.write('{0}\n'.format(score[0]))

                # get the intermediate results
                test_sim, test_kde = sess.run(inter_res, feed_dict=test_feed_dict)
                for i in range(self.batch_size):
                    for j in range(self.total_bins):
                        print test_kde[i][j] * 100,
                    print ""

            outfile.close()
            test_point_stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path")

    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_file", '-f', help="train_pair_file_path")
    parser.add_argument("--validation_file", '-v', help="val_pair_file_path")
    parser.add_argument("--train_size", '-z', type=int, help="number of train samples")
    parser.add_argument("--load_model", '-l', action='store_true')

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file")
    parser.add_argument("--test_size", type=int, default=0)
    parser.add_argument("--output_score_file", '-o')
    parser.add_argument("--emb_file_path", '-e')
    parser.add_argument("--checkpoint_dir", '-s', help="store model from here")

    args = parser.parse_args()

    conf = PyFileConfigLoader(args.config_file_path).load_config()

    if args.train:
        nn = SentCnnKnrm(config=conf)
        nn.train(train_pair_file_path=args.train_file,
                 val_pair_file_path=args.validation_file,
                 train_size=args.train_size,
                 checkpoint_dir=args.checkpoint_dir,
                 load_model=args.load_model,
                 test_point_file_path=args.test_file,
                 test_size=args.test_size)
    else:
        nn = SentCnnKnrm(config=conf)
        nn.test(test_point_file_path=args.test_file,
                test_size=args.test_size,
                output_file_path=args.output_score_file,
                load_model=True,
                checkpoint_dir=args.checkpoint_dir)
