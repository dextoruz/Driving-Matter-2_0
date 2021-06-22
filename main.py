import pyautogui as gui
import tensorflow.compat.v1 as tf   
import time

from car_game import Car
from dqn import DQNAgent,


EPISODES = 10


def main():
    gui.hotkey('alt', 'tab')
    time.sleep(.2)
    tf.disable_v2_behavior()
    
    #### traning is started from here
    env = Car()
    agent = DQNAgent()
    actions = {
        0: "Up",
        1: "Down",
        2: "Left",
        3: "Right",
        4:"back left",
        5:"back right"
    }

    tf.train.list_variables('./model')
    for ep in range(EPISODES):
        state = env.reset()
        tf.train.list_variables('./model')

        if ep == 0:
            tf.train.load_checkpoint('./model')

        totalReward = 0
        done = False

        ## one episode duration
        while not done:
            action = agent.policy(state)

            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, next_state, reward, done)
            
            totalReward += reward
            state = next_state
            
            print("Reward & Action:", reward,  actions[action])
            with open("instant-rewards.txt", "a+") as f:
                f.write(str(reward)+'\n')

        ## saving model        
        saver = tf.train.Saver()
        saver.save(agent.sess, 'saved_models/testing')

        ## saving rewards
        with open("rewards.txt", "a+") as f:
            f.write(str(totalReward)+'\n')
        
        print("\n\tTotal Reward: ", totalReward)


if __name__ == "__main__":
    main()