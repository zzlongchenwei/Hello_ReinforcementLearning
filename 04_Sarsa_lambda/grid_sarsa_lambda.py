# -*- coding: utf-8 -*-
# @File    : grid_sarsa_lambda.py
# @Date    : 2021-09-01
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/4_Sarsa_lambda_maze/run_this.py
import sys
sys.path.append('./')
from pathlib import Path

from lib.log import MyLog
from lib.RL import SarsaLambda
from lib.maze_env import Maze


def update_env(env, rl):
    for episode in range(EPISODES):
        state = env.reset()
        action = rl.choose_action(str(state))
        rl.eligibility_trace *= 0
        print(f'{episode + 1}: ', end='')

        while True:
            env.render()
            print(f'\033[34m{action_list[action]}\033[0m', end='')
            
            new_state, reward, done = env.step(action)
            new_action = rl.choose_action(str(new_state))
            rl.learn(str(state), action, reward, str(new_state), new_action)

            state = new_state
            action = new_action

            if done:
                env.render()
                print(f' {state}')
                break

    print('\033[36mgame over\033[0m')
    logger.info('LR:{}, E_GREEDY:{}, REWARD_DECAY:{},Q-table:\n{}'.format(LR,
                E_GREEDY, REWARD_DECAY, rl.q_table))
    mylog.pd_to_csv(rl.q_table)

    env.destroy()


if __name__ == '__main__':
    mylog = MyLog(Path(__file__))
    logger = mylog.logger

    EPISODES = 100
    LR = 0.01
    REWARD_DECAY = 0.9
    E_GREEDY = 0.9
    TRACE_DECAY = 0.9
    action_list = [" ⬆", " ⬇", " ➡", " ⬅"]

    env = Maze()
    rl = SarsaLambda(action_space=list(range(env.n_actions)),
                     learning_rate=LR,
                     reward_decay=REWARD_DECAY,
                     e_greedy=E_GREEDY,
                     trace_decay=TRACE_DECAY
                     )
    update_env(env, rl)
