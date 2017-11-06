"""Implements a model of behavior on a psychophys task as a (recurrent) DQN
with two actions: do nothing, or release the button to respond."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.framework import arg_scope

import matlab.engine  # Using to run experiment in psychtoolkit or whatever 
                      # Note that this import must occur LAST or things break
                      # Matlab sucks even in python

model_config = {
    'trial_length': 140,  # Number of frames in each trial
    'frame_subsample_rate': 5,  # Sample every kth frame, to make inputs shorter
    'vision_checkpoint_location': './inception_v3.ckpt',  # Obtained from http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    'LSTM_hidden_size': 20,
    'learning_rate': 0.001
}

model_config['LSTM_sequence_length'] = model_config['trial_length'] // model_config['frame_subsample_rate'] 


def _canonical_orientation(array):
    """Time as last -> first dimension"""
    return np.rollaxis(array, -1)

def _subsample(array, subsample_rate):
    """Should work with either numpy array or tensorflow tensor, assumes
       first axis is time."""
    return array[::subsample_rate]

class psychophys_model(object):
    """Class implementing the model. The construction of the model assumes
       batch_size = 1 (because it uses batch axis for time in some calls),
       changing this would require some minor edits, but shouldn't be necessary
       for the basic use."""

    def __init__(self, model_config, m_eng):
        example_trial = m_eng.samediff_step()
        example_input = np.array(example_trial['frames'])
        example_value = np.array(example_trial['value'])

        example_input = _canonical_orientation(example_input)
        example_value = _canonical_orientation(example_value)

        input_ph = tf.constant(example_input, dtype=tf.int32)  # TODO: switch to real placeholders
        reward_ph = tf.constant(example_value, dtype=tf.float32)  # TODO: switch to real placeholders
        epsilon_ph = tf.constant(0.5, dtype=tf.float32)  # TODO: switch to real placeholders
        
        # Subsample, resize, convert to float & scale to [-1, 1] for inception
        # Note: assumes input values are (ints) in [0, 255]
        possible_rewards = _subsample(reward_ph,
                                   model_config['frame_subsample_rate'])
        input_frames = _subsample(input_ph,
                                  model_config['frame_subsample_rate'])
        input_frames = 2. * (tf.cast(input_frames, tf.float32) / 255.) - 1. 
        self._input_frames = input_frames = tf.image.resize_nearest_neighbor(
            input_frames, [299, 299])

        
        # Do the recurrence: hoo boy
        # recurrence: init variables
        i = tf.constant(0, dtype=tf.int32)  # loop index

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            model_config['LSTM_hidden_size'],
            forget_bias=1.0,
            state_is_tuple=True)

        lstm_state = lstm_cell.zero_state(1, tf.float32)

        num_actions = 2  # Do nothing or release
        Q_vals = tf.zeros([1, num_actions], dtype=tf.float32)
        chose_to_release = tf.constant(False, dtype=tf.bool)

        explore_vals = tf.random_uniform([model_config['LSTM_sequence_length']],
                                          dtype=tf.float32)

        loss = tf.constant(0.)
        
        # recurrence: termination conditions
        def _recurrence_continues(i, lstm_state, Q_vals, chose_to_release, loss):
            trial_timeout = tf.equal(i, tf.constant(model_config['LSTM_sequence_length']))
            return tf.logical_not(tf.logical_or(trial_timeout, chose_to_release))

        # recurrence: loop body
        def _one_frame_forward(i, lstm_state, Q_vals, chose_to_release):
            """Runs one time-step forward from the input frame to the output Qs
               including epsilon-greedy action choice."""
            with tf.variable_scope('one_frame_forward', reuse=tf.AUTO_REUSE):
                # Get current frame
                curr_input_frame = input_frames[i:i+1]  # The :+1 keeps the dimension
                
                # Pass through inception_v3
                # Note: use_fused_batchnorm = False is to work around a bug that breaks
                # backprop through fused_batchnorm.
                with arg_scope(inception.inception_v3_arg_scope(use_fused_batchnorm=False)):
                    inception_features, _ = inception.inception_v3_base(curr_input_frame)
                    

                # Flatten and fully-connect to lstm inputs
                inception_features = slim.flatten(inception_features,
                                                  scope='inception_flattened')
                lstm_inputs = slim.fully_connected(inception_features,
                                                   model_config['LSTM_hidden_size'],
                                                   activation_fn=tf.nn.relu,
                                                   scope='lstm_inputs') 

                # LSTM!
                (lstm_outputs, lstm_state) = lstm_cell(lstm_inputs, lstm_state) 
                
                # Fully connect (linear) to Q estimates
                Q_vals = slim.fully_connected(lstm_outputs,
                                              num_actions,
                                              activation_fn=None,
                                              scope='Q_vals') 

                # is the greedy action to release?
                greedy_release = tf.less(Q_vals[0,0], Q_vals[0, 1])
                # is this an epsilon-exploring trial?
                explore = tf.less(explore_vals[i], epsilon_ph)

                # make the choice thereby
                chose_to_release = tf.logical_xor(greedy_release, explore)

                
            return (i, lstm_state, Q_vals, tf.squeeze(chose_to_release))

        def _loop_body(i, lstm_state, Q_vals, chose_to_release, loss):
            prev_Q_vals = Q_vals
            (i, lstm_state, Q_vals, chose_to_release) = _one_frame_forward(
                i, lstm_state, Q_vals, chose_to_release)
            # update loss for prev. step if i > 0
            loss = tf.cond(
                tf.greater(i, 0),
                lambda: loss + tf.square(tf.stop_gradient(tf.reduce_max(Q_vals)) - prev_Q_vals[0,0]),
                lambda: loss)
            i = i + 1
            return (i, lstm_state, Q_vals, chose_to_release, loss)
        
        # recurrence: actually running the loop
        (i, lstm_state, Q_vals, chose_to_release, loss) = tf.while_loop(
            _recurrence_continues,
            _loop_body,
            (i, lstm_state, Q_vals, chose_to_release, loss),
            swap_memory=True)


        # Get the reward for the trial -- either reward for releasing now, or
        # -1 if ran out of time
        trial_reward = tf.cond(chose_to_release,
                               lambda: possible_rewards[i],
                               lambda: -1.)

        # if released, update loss with reward difference from release Q, else
        # reward difference from do-nothing Q
        loss = tf.cond(
            chose_to_release,
            lambda: loss + tf.square(tf.stop_gradient(trial_reward) - Q_vals[0,1]),
            lambda: loss + tf.square(tf.stop_gradient(trial_reward) - Q_vals[0,0]))


        # training
        optimizer = tf.train.GradientDescentOptimizer(model_config['learning_rate'])
        train = optimizer.minimize(loss)

        
        # Initialize vision network from checkpoint
        tf.contrib.framework.init_from_checkpoint(
            model_config['vision_checkpoint_location'],
            {'InceptionV3/': 'InceptionV3/'})
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print(self.sess.run([i, Q_vals, chose_to_release, trial_reward]))
        print(self.sess.run([i, Q_vals, chose_to_release, trial_reward]))
        self.sess.run(train)
        print(self.sess.run([i, Q_vals, chose_to_release, trial_reward]))
        print(self.sess.run([i, Q_vals, chose_to_release, trial_reward]))
            
        
        
        
        
        
        


if __name__ == '__main__':
    m_eng = matlab.engine.start_matlab()
    m_eng.cd('..')  # Makes matlab work in the directory containing the
                    # samediff_step function, update as necessary
    psychophys_model(model_config, m_eng)
