""" 
-*- coding: utf-8 -*-
@File    : Sarsa.py
@Date    : 2021-08-29
@Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 
@From    : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/3_Sarsa_maze/RL_brain.py
"""
import numpy as np
import pandas as pd

np.random.seed(2)

class RL:
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9) -> None:
        self.actions = action_space
        self.gamma = reward_decay
        self.lr = learning_rate
        self.eposilon = e_greedy
        
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state row to q table
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)    
            )
    
    def choose_action(self, state):
        self.check_state_exist(state)

        if np.random.uniform() < self.eposilon:
            state_action = self.q_table.loc[state, :]
            # choice max score action (maybe have two with the same score)
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        
        return action
    
    def learn(self, *args):
        pass


# off-policy
class QLearning(RL):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9) -> None:
        super().__init__(action_space, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table[s_, :].max()
        else:
            q_target = r
        
        self.q_table[s, a] += self.lr * (q_target - q_predict)


# on-policy
class Sarsa(RL):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=.9, e_greedy=.9) -> None:
        super().__init__(action_space, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy=e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # continue choice to the next action
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
