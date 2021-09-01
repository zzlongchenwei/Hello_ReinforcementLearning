# -*- coding: utf-8 -*-
# @File    : cartpole_q_learning.py
# @Date    : 2021-09-01
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 
import sys 
sys.path.append('./')
import gym

from lib.RL import QLearning


def cartpole_update(env, rl):
    for episode in range(EPISODES):
        obs = env.reset()
        timestep = 0
        while True:
            env.render()
            action = rl.choose_action(str(obs))
            # apply action
            new_obs, reward, done, info = env.step(action)
            # update q table 
            rl.learn(str(obs), action, reward, str(new_obs))
            # update state
            obs = new_obs
            timestep += 1
            if done:
                env.render()
                print('Episode finished: {} timesteps.'.format(timestep))
                break
            

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    action_sapce = list(range(env.action_space.n))

    EPISODES = 1000
    LR = 0.1
    REWARD_DECAY = 0.9
    E_GREEDY = 0.9

    rl = QLearning(action_sapce, learning_rate=LR, reward_decay=REWARD_DECAY, e_greedy=E_GREEDY)
    cartpole_update(env, rl)
