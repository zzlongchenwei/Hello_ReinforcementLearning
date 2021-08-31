""" 
-*- coding: utf-8 -*-
@File    : run.py
@Date    : 2021-08-29
@Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 
@From    : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/2_Q_Learning_maze/run_this.py
"""
import sys
sys.path.append("./")
from pathlib import Path

from lib.log import MyLog
from lib.maze_env import Maze
from Sarsa import Sarsa


mylog = MyLog(Path(__file__).parent)
logger = mylog.logger
action_list = [" ⬆", " ⬇", " ➡", " ⬅"]


def update_state(env: Maze, rl: Sarsa) -> None:
    for episode in range(EPISODE):
        state = env.reset()
        print(f'{episode + 1}: ', end='')
        action = rl.choose_action(str(state))
        while True:
            env.render()
            # move_list += action_list[action]
            print(f'\033[34m{action_list[action]}\033[0m', end='')

            # execute action, choose new action
            new_state, reward, done = env.step(action)
            new_action = rl.choose_action(str(new_state))
            # update Q_table
            rl.learn(str(state), action, reward, str(new_state), new_action)

            # next step
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
    EPISODE = 200
    LR = 0.01
    REWARD_DECAY = 0.9
    E_GREEDY = 0.9

    env = Maze()
    rl = Sarsa(action_space=list(range(env.n_actions)),
               learning_rate=LR,
               reward_decay=REWARD_DECAY,
               e_greedy=E_GREEDY)
    env.after(100, update_state, env, rl)
    env.mainloop()
