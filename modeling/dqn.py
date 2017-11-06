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
    'frame_subsample_rate': 5,  # Sample every kth frame, to make inputs shorter
    'vision_checkpoint_location': './inception_v3.ckpt',  # Obtained from http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    'LSTM_hidden_size': 20
}


def _canonical_orientation(array):
    """Time as last -> first dimension"""
    return np.rollaxis(array, -1)

def _subsample(array, subsample_rate):
    """Should work with either numpy array or tensorflow tensor, assumes
       first axis is time."""
    return array[::subsample_rate]

class psychophys_model(object):
    """Class implementing the model."""

    def __init__(self, model_config, m_eng):
        example_trial = m_eng.samediff_step()
        example_input = np.array(example_trial['frames'])
        example_value = np.array(example_trial['value'])

        example_input = _canonical_orientation(example_input)
        example_value = _canonical_orientation(example_value)

        input_ph = tf.constant(example_input, dtype=tf.int32)  # TODO: switch to real placeholders
        reward_ph = tf.constant(example_value, dtype=tf.float32)  # TODO: switch to real placeholders
        
        # Subsample, resize, convert to float & scale to [-1, 1] for inception
        # Note: assumes input values are (ints) in [0, 255]
        input_frames = _subsample(input_ph,
                                  model_config['frame_subsample_rate'])
        input_frames = 2. * (tf.cast(input_frames, tf.float32) / 255.) - 1. 
        input_frames = tf.image.resize_nearest_neighbor(input_frames, [299, 299])

        # Get current frame
        # TODO: put in loop

        
        
        # Pass through inception_v3
        # Note: use_fused_batchnorm = False is to work around a bug (from
        # summer 2017) that breaks backprop through fused_batchnorm.
        with arg_scope(inception.inception_v3_arg_scope(use_fused_batchnorm=False)):
            inception_features, _ = inception.inception_v3_base(input_frames)

        
        tf.contrib.framework.init_from_checkpoint(model_config['vision_checkpoint_location'],
                                {'InceptionV3/': 'InceptionV3/'})
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(inception_features)
            
        
        
        
        
        
        


if __name__ == '__main__':
    m_eng = matlab.engine.start_matlab()
    m_eng.cd('..')  # Makes matlab work in the directory containing the
                    # samediff_step function, update as necessary
    psychophys_model(model_config, m_eng)
