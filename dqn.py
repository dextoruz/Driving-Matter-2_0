
from os import write
import random
# import tensorflow.compat.v1 as tf   
import numpy as np
import time
from car_game import Car
#############################
#if you want to use GPU to boost, use these code.  
# import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} ) 
# sess = tf.Session(config=config) 
# K.set_session(sess)
import pyautogui as gui


#############################

GAMMA=0.98
LEARNING_RATE=0.01
EPSILON=0.99


class QNetwork():
    def __init__(self, state_dim, action_size):
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        
        
        self.hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
        self.q_state = tf.layers.dense(self.hidden1, action_size, activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)          
        self.loss = tf.reduce_mean(tf.square(self.q_state_action- self.q_target_in)) 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)

    def update_model(self, session, state, action, q_target):
        feed = {self.state_in:state, self.action_in:action, self.q_target_in:q_target}
        session.run(self.optimizer, feed_dict=feed)

    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict={self.state_in:state})
        return q_state


class DQNAgent():
    def __init__(self, ):
        self.state_dim = (1,900,1760)
        self.action_size = 6
        self.q_network = QNetwork(self.state_dim, self.action_size)

        self.gamma = GAMMA
        self.eps = EPSILON
        self.learning_rate = LEARNING_RATE

        # self.sess = tf.Session()
        config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} ) 
        self.sess = tf.Session(config=config) 
        tf.keras.backend.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())


    def policy(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state[0][0])
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.eps else action_greedy
        return action


    def train(self, state, action, next_state, reward, done):
        q_next_state = self.q_network.get_q_state(self.sess, [next_state])
        q_state = self.q_network.get_q_state(self.sess, [state])
        q_next_state = (1-done) * q_next_state
        # print("Q next state: ", q_next_state[0][0])
        q_target = reward + (self.gamma * np.max(q_next_state) - np.max(q_state))
        self.q_network.update_model(self.sess, [state], [action], [q_target])

        if done: self.eps = max(0.1, 0.99 * self.eps)


def start():
    gui.hotkey('alt', 'tab')
    time.sleep(.2)
    tf.disable_v2_behavior()
    env = Car()
    agent = DQNAgent(env)

    episodes = 10
    actions = {
        0: "Up",
        1: "Down",
        2: "Left",
        3: "Right",
        4:"back left",
        5:"back right"
    }
    tf.train.list_variables('./saved_models')
    
    for ep in range(episodes):
        state = env.reset()
        tf.train.list_variables('./saved_models')

        if ep == 0:
            tf.train.load_checkpoint('./saved_models')

        totalReward = 0
        done = False
        print("\n\t\tEpisode: "+str(ep+1)+"\n")
        # print("| Action | State | Next State | Total Reward | Done |")
        while not done:
            action = agent.policy(state)
            if action > 5 or action < 0:
                # print("Not found")
                action = 2 

            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, next_state, reward, done)
            
            totalReward += reward
            state = next_state
            
            print("Reward & Action:", reward,  actions[action])
            with open("instant-rewards.txt", "a+") as f:
                f.write(str(reward)+'\n')
        if ep % 1 == 0:
            # print("Completed Training Cycle: " + str(epoch) + " out of " + str(self.num_of_epoch))
            # print("Current Loss: " + str(loss))

            saver = tf.train.Saver()
            saver.save(agent.sess, 'saved_models/testing')
            print("Model saved")
        with open("rewards.txt", "a+") as f:
            f.write(str(totalReward)+'\n')
        # print("\nEpisode: {}, Total Rewards: {1.2f}".format(ep, totalReward))
        # print("Episode: ", ep)
        print("\n\tTotal Reward: ", totalReward)


if __name__ == "__main__":
    start()