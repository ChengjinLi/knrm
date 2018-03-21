#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年06月21日

@author: lichengjin
训练数据通过队列+多线程方式传入model，测试数据通过一次性load到内存，placeholder+feed的方式传入model
"""
from __future__ import absolute_import
import os
import time
import tensorflow as tf
from data_helpers import get_config_args, load_all_data_at_once, load_batch_data_by_queue
from knrm import KNRMModel


class Train(object):

    def __init__(self, config):
        self.config = config
        if os.path.isdir(config.train_data_path):
            filename_list = [config.train_data_path + '/' + file_name for file_name in os.listdir(config.data_path)]
        else:
            filename_list = [config.train_data_path]
        query_term_ids, doc_term_ids = load_batch_data_by_queue(filename_list,
                                                                config.max_query_term_length,
                                                                config.max_doc_term_length,
                                                                config.num_epochs,
                                                                config.batch_size,
                                                                12)
        with tf.name_scope('train') as scope:
            with tf.variable_scope('model'):
                self.model = KNRMModel(query_term_ids, doc_term_ids, config, scope)

    def train_step(self, session, summary_writer=None):
        """train step"""
        start_time = time.time()
        fetches = [self.model.train_op,
                   self.model.global_step,
                   self.model.loss,
                   self.model.pos_scores,
                   self.model.neg_scores,
                   self.model.summary]
        _, global_step, loss_val, pos_scores_val, neg_scores_val, summary = session.run(fetches)
        print "pos_scores_val"
        print pos_scores_val
        print "neg_scores_val"
        print neg_scores_val
        if self.config.show_freq and (global_step <= 100 or global_step % self.config.show_freq == 0):
            step_time = time.time() - start_time
            examples_per_sec = self.config.batch_size / step_time
            print ("Train, step {}, loss {:g}, step-time {:g}, examples/sec {:g}"
                   .format(global_step, loss_val, step_time, examples_per_sec))
        if summary_writer:
            summary_writer.add_summary(summary, global_step)
        return global_step


class Test(object):

    def __init__(self, config):

        self.query_term_ids, self.doc_term_ids = load_all_data_at_once(config.test_data_path,
                                                                       config.max_query_term_length,
                                                                       config.max_doc_term_length,
                                                                       config.test_num)
        self.input_query_ph = tf.placeholder(tf.int32, [None, config.max_query_term_length], name='input_query')
        self.input_doc_ph = tf.placeholder(tf.int32, [None, 5, config.max_doc_term_length], name='input_doc')
        with tf.name_scope('test') as scope:
            with tf.variable_scope('model', reuse=True):
                self.model = KNRMModel(self.input_query_ph, self.input_doc_ph, config, scope)

    def test_step(self, session, summary_writer=None):
        """eval step"""
        start_time = time.time()
        feed_dict = dict()
        feed_dict[self.input_query_ph] = self.query_term_ids
        feed_dict[self.input_doc_ph] = self.doc_term_ids
        fetches = [self.model.global_step,
                   self.model.loss,
                   self.model.scores,
                   self.model.pos_scores,
                   self.model.neg_scores,
                   self.model.summary]
        global_step, loss_val, scores_val, pos_scores_val, neg_scores_val, summary = \
            session.run(fetches, feed_dict)
        score_difference = pos_scores_val - neg_scores_val
        bigger = 0
        lower = 0
        equal = 0
        for item in score_difference:
            if item > 0:
                bigger += 1
            elif item < 0:
                lower += 1
            else:
                equal += 1
        step_time = time.time() - start_time
        examples_per_sec = self.query_term_ids.shape[0] / step_time
        print ("Test, step {}, bigger {:g}, equal {:g}, lower {:g}, b/l, {:g}, step-time {:g}, examples/sec {:g}"
               .format(global_step, bigger, equal, lower, float(bigger)/lower, step_time, examples_per_sec))
        if summary_writer:
            summary_writer.add_summary(summary, global_step)


def main():
    config = get_config_args()
    out_dir = os.path.abspath(config.model_dir)
    print("Writing to {}\n".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    with tf.Graph().as_default(), tf.Session() as sess:
        train = Train(config)
        # test = Test(config)
        train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), sess.graph)
        # test_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "test"), sess.graph)
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        try:
            while not coord.should_stop():
                step = train.train_step(sess, train_summary_writer)
                # if step % config.test_freq == 0:
                #     test.test_step(sess, test_summary_writer)
                if step % config.save_freq == 0:
                    path = saver.save(sess, checkpoint_prefix, step)
                    print("Saved model checkpoint to {}\n".format(path))
        except tf.errors.OutOfRangeError:
            print "============Train finished========="
            path = saver.save(sess, checkpoint_prefix, step)
            print("Saved model checkpoint to {}\n".format(path))
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    main()
