# -*- coding: utf-8 -*-
# @File    : cartpole_q_learning.py
# @Date    : 2021-09-01
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 
import sys 
sys.path.append('./')
from pathlib import Path
import gym

from lib.RL import Sarsa
from lib.log import MyLog

mylog = MyLog(Path(__file__), filesave=True, consoleprint=True)
logger = mylog.logger


def cartpole_update(env, rl):
    for episode in range(EPISODES):
        obs = env.reset()
        timestep = 0
        action = rl.choose_action(str(obs))
        while True:
            env.render()
            # apply action
            new_obs, reward, done, info = env.step(action)
            new_action = rl.choose_action(str(new_obs))
            # update q table 
            rl.learn(str(obs), action, reward, str(new_obs), new_action)
            # update state
            obs = new_obs
            action = new_action
            timestep += 1
            if done:
                env.render()
                print('Episode finished: keep {} timesteps.'.format(timestep))
                break
    
    print('\033[36mgame over\033[0m')
    logger.info('LR:{}, E_GREEDY:{}, REWARD_DECAY:{},Q-table:\n{}'.format(LR,
                E_GREEDY, REWARD_DECAY, rl.q_table))
    mylog.pd_to_csv(rl.q_table)
            

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    action_space = list(range(env.action_space.n))

    EPISODES = 1000
    LR = 0.1
    REWARD_DECAY = 0.9
    E_GREEDY = 0.9

    rl = Sarsa(action_space, learning_rate=LR, reward_decay=REWARD_DECAY, e_greedy=E_GREEDY)
    cartpole_update(env, rl)
