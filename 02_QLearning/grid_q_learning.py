""" 
-*- coding: utf-8 -*-
@File    : run.py
@Date    : 2021-08-29
@Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 
@From    : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/2_Q_Learning_maze/run_this.py
"""
from pathlib import Path
import sys
sys.path.append("../")

from Q_Learning import MyQLearning
from lib.log import MyLog
from lib.maze_env import Maze


mylog = MyLog(Path(__file__))
logger = mylog.logger

action_list = [" ⬆", " ⬇", " ➡", " ⬅"]


def update_state(cartoon: Maze, rl: MyQLearning) -> None:
    for episode in range(EPISODE):
        state = cartoon.reset()
        print(f'{episode + 1}: ', end='')

        while True:
            cartoon.render()
            # choose action from current state (coordinate)
            action = rl.choose_action(str(state))

            print(f'\033[34m{action_list[action]}\033[0m', end='')

            # execute action
            new_state, reward, done = cartoon.step(action)
            # update Q_table
            rl.learn(str(state), action, reward, str(new_state))
            # next step
            state = new_state

            if done:
                cartoon.render()
                print(f' {state}')
                break

    print('\033[36mgame over\033[0m')
    logger.info('Q-table:\n{}'.format(rl.q_table))
    mylog.pd_to_csv(rl.q_table)

    cartoon.destroy()


if __name__ == '__main__':
    EPISODE = 200
    LR = 0.01
    REWARD_DECAY = 0.9
    E_GREEDY = 0.8

    env = Maze()
    myrl = MyQLearning(actions=list(range(env.n_actions)),
                     learning_rate=LR,
                     reward_decay=REWARD_DECAY,
                     e_greedy=E_GREEDY)

    env.after(100, update_state, env, myrl)
    env.mainloop()
