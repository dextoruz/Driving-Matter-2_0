
import random
from tensorflow.compat.v1.keras import backend as K
import tensorflow.compat.v1 as tf   
import numpy as np


## hyperparameters 
INPUT_IMAGE_SHAPE = ( 900, 1760, 1 )
NUM_ACTIONS = 6
GAMMA = 0.98
LEARNING_RATE = 0.01
EPSILON = 0.99


class QNetwork():
    def __init__(self, state_dim, action_size, LEARNING_RATE):
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        self.net = tf.layers.conv2d(self.state_in, 32 , (3,3), strides=(1, 1), activation=tf.nn.relu, name='Conv1')
        self.net = tf.layers.conv2d(self.net, 64 , (3,3), strides=(1, 1), activation=tf.nn.relu, name='Conv2')
        self.net = tf.layers.conv2d(self.net, 128 , (3,3), strides=(1, 1), activation=tf.nn.relu, name='Conv3')
        self.net = tf.layers.Flatten()(self.state_in)
        self.dense = tf.layers.dense(self.net, 128, activation=tf.nn.relu)
        self.q_state = tf.layers.dense(self.dense, action_size, activation=None, )
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)          
        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in)) 
        self.network = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=GAMMA, momentum=0.0, epsilon=EPSILON, name='RMSProp')

    def update_model(self, session, state, action, q_target):
        feed = {self.state_in:state, self.action_in:action, self.q_target_in:q_target}
        session.run(self.network, feed_dict=feed)

    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict={self.state_in:state})
        return q_state


class DQNAgent():
    def __init__(self):
        self.state_dim = INPUT_IMAGE_SHAPE
        self.action_size = NUM_ACTIONS
        self.q_network = QNetwork(self.state_dim, self.action_size, LEARNING_RATE)

        self.gamma = GAMMA
        self.eps = EPSILON
        self.learning_rate = LEARNING_RATE

        config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} ) 
        self.sess = tf.Session(config=config) 
        tf.keras.backend.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())


    def policy(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        if random.random() < self.eps:
            ## random action
            return np.random.randint(self.action_size)
        else: 
            ## greedy action
            return np.argmax(q_state[0][0])
         


    def train(self, state, action, next_state, reward, done):
        q_next_state = self.q_network.get_q_state(self.sess, [next_state])
        q_state = self.q_network.get_q_state(self.sess, [state])
        q_next_state = (1-done) * q_next_state
        q_target = reward + (self.gamma * np.amax(q_next_state) - np.amax(q_state))
        self.q_network.update_model(self.sess, [state], [action], [q_target])
        if done:
            self.eps = max(0.1, 0.99 * self.eps)

    
    def save_model(self):
        self.sess.save_weights("./models/weights.h5")

