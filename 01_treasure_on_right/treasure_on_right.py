""" 
-*- coding: utf-8 -*-
@File    : treasure_on_right.py
@Date    : 2021-08-27
@Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 
@From    : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/1_command_line_reinforcement_learning/treasure_on_right.py
"""
import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
log_path = os.path.join(os.getcwd(), 'Logs')

try:
    os.mkdir(log_path)
except FileExistsError:
    pass

log_file = os.path.join(log_path, time.strftime(
    '%Y%m%d%M', time.localtime(time.time())) + '.log')

fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# np.random.seed(2)

N_STATES = 6  # the length of the 1 dimensional space
ACTIONS = ['left', 'right']  # available actions
EPSILON = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 15  # maximum episodes
FRESH_TIME = 0.5  # fresh time for one move


def build_q_table(n_states, actions):
    """
    Initialize the Q table 
    Frame: row: n_states, column: actions
    """
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    logger.debug('Init Q-table:\n{}'.format(table))
    return table


def choose_action(state, q_table):
    """
    Choice the action from action space
    If a random figure bigger than epsilon, random choice a action.
    else choice the max score of the action.
    """
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax(axis='columns')
    return action_name


def get_env_feedback(S, A):
    """
    update state from current action, and get reward
    S: state
    A: action
    return: S_: new state, R: reward.
    """
    if A == "right":
        if S == N_STATES - 2:  # minus start and end position
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    """
    update print character
    """
    env_list = ['_'] * (N_STATES - 1) + ["🏆"]
    if S == "terminal":
        interaction = 'Episode %s: total_steps = %s' % (
            episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                             ', end='')

    else:
        env_list[S] = '🕺'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    """
    main RL loop
    """
    q_table = build_q_table(N_STATES, ACTIONS)

    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # according to current State and Action predict the next State and Reward
            # Reality, current State , current Action => A Score
            q_predict = q_table.loc[S, A]  # according to current State and Action get Q_score from Q_table 
            if S_ != 'terminal':
                # Estimate, next State,  max score of next Action of the next State => B Score
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # calculate the Q_score through max Action Q_score of next State
            else:
                q_target = R
                is_terminated = True

            # Foresight, update current
            # from current S&A foresee the future, e.g. 记吃记打
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r                           ')
    logger.info('Q-table:\n{}'.format(q_table))
