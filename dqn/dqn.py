"""Implements a model of behavior on a psychophys task as a (recurrent) DQN
with two actions: do nothing, or release the button to respond."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.framework import arg_scope
#from tensorflow.python.client import timeline  # for profiling

import matlab.engine  # Using to run experiment in psychtoolkit or whatever 
                      # Note that this import must occur LAST or things break
                      # Matlab sucks even in python
# wait step
model_config = {
    'trial_length': 50,  # Number of frames in each trial
    'frame_subsample_rate': 5,  # Sample every kth frame, to make inputs shorter
    'vision_checkpoint_location': './inception_v3.ckpt',  # Obtained from http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    'LSTM_hidden_size': 50,
    'image_size': 32,  # width/height of input images (assumes square)
    'discount': 0.95,  # The temporal discount factor 
    'optimizer': 'RMSProp',  # One of 'Adam' or 'SGD' or 'RMSProp'
    'learning_rate': 5e-4,
    'learning_rate_decay': 0.95,  # multiplicative decay
    'learning_rate_decays_every': 1000,
    'max_grad_norm': 5,  # gradients will be clipped to this max global norm if it is not None
    'min_learning_rate': 1e-5,
    'num_trials': 15000, # How many trials to run
    'save_every': 1000,  # save model every n trials
    'save_path': '/home/andrew/data/att1/dqn/waitstep/checkpoint/model.ckpt',  # where to save/load model checkpoints
    'task_function_folder': '../miniexp/',  # where the task .m files are
    'task_function': 'wait_step',
    'reload': False,  # if true, start by reloading the model
    'init_epsilon': 0.2,  # exploration probability
    'epsilon_decay': 0.01,  # additive decay 
    'epsilon_decays_every': 500,  # number of trials between epsilon decays
    'min_epsilon': 0.0,
    'tune_vision_model': False  # whether to backprop through vision model.
                                # stopping backprop at the vision model output
                                # will significantly speed up training.
}

# 
#model_config = {
#    'trial_length': 100,  # Number of frames in each trial
#    'frame_subsample_rate': 5,  # Sample every kth frame, to make inputs shorter
#    'vision_checkpoint_location': './inception_v3.ckpt',  # Obtained from http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
#    'LSTM_hidden_size': 50,
#    'image_size': 32,  # width/height of input images (assumes square)
#    'discount': 0.95,  # The temporal discount factor 
#    'optimizer': 'RMSProp',  # One of 'Adam' or 'SGD' or 'RMSProp'
#    'learning_rate': 5e-4,
#    'learning_rate_decay': 0.95,  # multiplicative decay
#    'learning_rate_decays_every': 1000,
#    'max_grad_norm': 5,  # gradients will be clipped to this max global norm if it is not None
#    'min_learning_rate': 1e-5,
#    'num_trials': 50000, # How many trials to run
#    'save_every': 1000,  # save model every n trials
#    'save_path': '/home/andrew/data/att1/dqn/waitcolor/checkpoint/model.ckpt',  # where to save/load model checkpoints
#    'task_function_folder': '../miniexp/',  # where the task .m files are
#    'task_function': 'waitcolor_step',
#    'reload': True,  # if true, start by reloading the model
#    'init_epsilon': 0.2,  # exploration probability
#    'epsilon_decay': 0.01,  # additive decay 
#    'epsilon_decays_every': 500,  # number of trials between epsilon decays
#    'min_epsilon': 0.05,
#    'tune_vision_model': False  # whether to backprop through vision model.
#                                # stopping backprop at the vision model output
#                                # will significantly speed up training.
#}

np.random.seed(0)  # reproducibility
tf.set_random_seed(0) 


def _matlab_to_numpy(array, shape):
    """Converts results of matlab calls to numpy arrays efficiently"""
    return np.array(array._data).reshape(shape, order='F')


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
        model_config['LSTM_sequence_length'] = model_config['trial_length'] // model_config['frame_subsample_rate'] 
        self.m_eng = m_eng
        self.model_config = model_config
        trial_length = model_config['trial_length']
        im_size = model_config['image_size']
        self.input_ph = input_ph = tf.placeholder(
            tf.int32, [trial_length, im_size, im_size, 3]) 

        self.reward_ph = reward_ph = tf.placeholder(
            tf.float32, [trial_length, 1])

        self.epsilon_ph = epsilon_ph = tf.placeholder(tf.float32, [])
        
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
                if not model_config['tune_vision_model']:
                    inception_features = tf.stop_gradient(inception_features)
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
            discount = model_config['discount']
            loss = tf.cond(
                tf.greater(i, 0),
                lambda: loss + tf.square(discount * tf.stop_gradient(tf.reduce_max(Q_vals)) - prev_Q_vals[0,0]),
                lambda: loss)
            i = i + 1
            return (i, lstm_state, Q_vals, chose_to_release, loss)
        
        # recurrence: actually running the loop
        (i, lstm_state, Q_vals, chose_to_release, loss) = tf.while_loop(
            _recurrence_continues,
            _loop_body,
            (i, lstm_state, Q_vals, chose_to_release, loss),
            swap_memory=True)


        # make available for debugging/evaluating behavior
        self.step_trial_ended = i
        self.chose_to_release = chose_to_release
        self.final_Q_values = Q_vals


        # Get the reward for the trial -- either reward for releasing now, or
        # -1 if ran out of time
        self.trial_reward = trial_reward = tf.cond(
            tf.less(i, tf.constant(model_config['LSTM_sequence_length'])),
            lambda: possible_rewards[i],
            lambda: tf.constant([-1.]))

        # if released, update loss with reward difference from release Q, else
        # reward difference from do-nothing Q
        self.loss = loss = tf.cond(
            chose_to_release,
            lambda: loss + tf.square(tf.stop_gradient(trial_reward) - Q_vals[0,1]),
            lambda: loss + tf.square(tf.stop_gradient(trial_reward) - Q_vals[0,0]))


        # training
        if model_config['optimizer'] == 'Adam':
            optimizer = tf.train.AdamOptimizer(model_config['learning_rate'], epsilon=0.1)
        elif model_config['optimizer'] == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(model_config['learning_rate'])
        elif model_config['optimizer'] == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(model_config['learning_rate'])
        else:
            raise ValueError('Invalid optimizer')

        if model_config['max_grad_norm'] is None:
            self.train = optimizer.minimize(loss)
        else: # Gradient clipping
            grads_and_vars = optimizer.compute_gradients(loss) 
            gradients = [g for (g, _) in grads_and_vars]
            variables = [v for (_, v) in grads_and_vars]
            clipped_grads_and_vars = zip(tf.clip_by_global_norm(gradients, model_config['max_grad_norm'])[0], variables) 
            self.train = optimizer.apply_gradients(clipped_grads_and_vars)
        
        self.curr_learning_rate = model_config['learning_rate']
        self.min_learning_rate = model_config['min_learning_rate']
        self.learning_rate_decay = model_config['learning_rate_decay']
        self.learning_rate_decays_every = model_config['learning_rate_decays_every']
        
        # set to initialize vision network from checkpoint
        tf.contrib.framework.init_from_checkpoint(
            model_config['vision_checkpoint_location'],
            {'InceptionV3/': 'InceptionV3/'})

        # saving + restoring
        self.saver = tf.train.Saver()
        self.save_every = model_config['save_every']
        self.save_path = model_config['save_path']
        
        # create session and initialize
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # reload if desired
        if model_config['reload']:
            self.restore_parameters()

        # exploration parameters
        self.curr_epsilon = model_config['init_epsilon']
        self.min_epsilon = model_config['min_epsilon']
        self.epsilon_decay = model_config['epsilon_decay']
        self.epsilon_decays_every = model_config['epsilon_decays_every']

        # task function
        self.task_function = model_config['task_function']
        
    def run_trials(self, num_trials=None):
        if num_trials is None:
            num_trials = model_config['num_trials'] 
        log_length = 100 # how many trials to aggregate statistics over between reports
        this_trial = eval('self.m_eng.%s()' % self.task_function)
        rewards = np.zeros([log_length])
        losses = np.zeros([log_length])

        for trial_i in xrange(num_trials):
            trial_length = model_config['trial_length']
            im_size = model_config['image_size']
            this_input = _matlab_to_numpy(this_trial['frames'], shape=[im_size, im_size, 3, trial_length])
            this_value = _matlab_to_numpy(this_trial['value'], shape=[1, trial_length])

            this_input = _canonical_orientation(this_input)
            this_value = _canonical_orientation(this_value)

#            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # Profiling
#            run_metadata = tf.RunMetadata()
            
            this_reward, this_choice, this_step_ended, this_loss, this_Q_vals, _ = self.sess.run(
                [self.trial_reward, self.chose_to_release, self.step_trial_ended, self.loss, self.final_Q_values, self.train],
#                options=options, run_metadata=run_metadata,  # Profiling
                feed_dict={self.input_ph: this_input,
                           self.reward_ph: this_value,
                           self.epsilon_ph: self.curr_epsilon})

#            fetched_timeline = timeline.Timeline(run_metadata.step_stats)  # Profiling
#            chrome_trace = fetched_timeline.generate_chrome_trace_format()
#            with open('profiling/timeline_%i.json' % trial_i, 'w') as f:
#                f.write(chrome_trace)

#            print(trial_i, this_reward, this_choice, this_step_ended, this_loss, this_Q_vals)
            i_mod = trial_i % log_length
            rewards[i_mod] = this_reward
            losses[i_mod] = this_loss
            if i_mod == log_length-1:
                print(trial_i, np.mean(rewards), np.mean(losses))
                rewards = np.zeros([log_length])
                losses = np.zeros([log_length])


            # handle epsilon decay
            if trial_i % self.epsilon_decays_every == 0 and self.curr_epsilon > self.min_epsilon: 
                self.curr_epsilon -= self.epsilon_decay

            # handle learning_rate decay
            if trial_i % self.learning_rate_decays_every == 0 and self.curr_learning_rate > self.min_learning_rate: 
                self.curr_learning_rate *= self.learning_rate_decay

            # save progress
            if self.save_every is not None and trial_i % self.save_every == 0:
                model.save_parameters()

            # pass back result and get next trial
            was_correct = np.asscalar(this_reward) == 1.
            this_trial = eval('self.m_eng.%s(was_correct)' % self.task_function)

    def save_parameters(self):
        self.saver.save(self.sess, self.save_path)
        print('Checkpoint saved to ' + self.save_path)

    def restore_parameters(self):
        self.saver.restore(self.sess, self.save_path)
        print('Restored from ' + self.save_path)

if __name__ == '__main__':
    t = time.time()
    m_eng = matlab.engine.start_matlab()
    m_eng.cd(model_config['task_function_folder'])  # Makes matlab work in the directory containing the
                             # wait_step function, update as necessary
    model = psychophys_model(model_config, m_eng)
    print('init took %.2f seconds' % (time.time() - t))
    
    t = time.time()
    model.run_trials()
    print('running %i trials took %.2f seconds' % (model_config['num_trials'], time.time() - t))

    model.save_parameters()
