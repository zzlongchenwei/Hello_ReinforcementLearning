""" 
-*- coding: utf-8 -*-
@File    : maze_env.py
@Date    : 2021-08-29
@Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来- 
@From    : https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/2_Q_Learning_maze/maze_env.py
"""
import numpy as np
import time
import tkinter as tk


UNIT = 100  # pixels
MAZE_C = 7  # grid height
MAZE_R = 3  # grid width
SLEEP = .02
REC_SIZE = UNIT/2 - 6


class Maze(tk.Tk):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['up', 'down', 'right', 'left']
        self.n_actions = len(self.action_space)
        self.title("maze")
        self.geometry('{0}x{1}'.format(MAZE_C * UNIT, MAZE_R * UNIT))
        self.hell_list = []
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_R * UNIT, width=MAZE_C * UNIT)

        # create grids
        # draw row line
        # 行为x 列为y
        for r in range(0, MAZE_R * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_C * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        # draw column line
        for c in range(0, MAZE_C * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_R * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([UNIT/2, UNIT/2])

        # # hell1
        # hell1_center = origin + np.array([UNIT * 2, UNIT])
        # self.hell1 = self.canvas.create_rectangle(
        #     hell1_center[0] - 15, hell1_center[1] - 15,
        #     hell1_center[0] + 15, hell1_center[1] + 15,
        #     fill='black'
        # )
        # # hell2
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black'
        # )

        self.create_row_hell(origin, MAZE_C - 2)

        # create oval
        # oval_center = origin + UNIT * 2
        oval_center = origin + np.array([UNIT * (MAZE_C - 1), 0])
        self.oval = self.canvas.create_oval(
            oval_center[0] - REC_SIZE, oval_center[1] - REC_SIZE,
            oval_center[0] + REC_SIZE, oval_center[1] + REC_SIZE,
            fill='yellow'
        )

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - REC_SIZE, origin[1] - REC_SIZE,
            origin[0] + REC_SIZE, origin[1] + REC_SIZE,
            fill='red'
        )

        # pack
        self.canvas.pack()

    def create_row_hell(self, origin, number):
        for i in range(number):
            hell_center = origin + np.array([UNIT * (i + 1), 0])
            self.hell_list.append(self.canvas.create_rectangle(
                hell_center[0] - REC_SIZE, hell_center[1] - REC_SIZE,
                hell_center[0] + REC_SIZE, hell_center[1] + REC_SIZE,
                fill='black'))

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([UNIT/2, UNIT/2])
        self.rect = self.canvas.create_rectangle(
            origin[0] - REC_SIZE, origin[1] - REC_SIZE,
            origin[0] + REC_SIZE, origin[1] + REC_SIZE,
            fill='red'
        )
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT

        elif action == 1:  # down
            if s[1] < (MAZE_R - 1) * UNIT:
                base_action[1] += UNIT

        elif action == 2:  # right
            if s[0] < (MAZE_C - 1) * UNIT:
                base_action[0] += UNIT

        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(
            self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(hell) for hell in self.hell_list]:
            reward = -1
            done = True
            s_ = 'hell'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(SLEEP)
        self.update()

    def test(self):
        for t in range(10):
            s = self.reset()
            while True:
                self.render()
                a = 1
                s, r, done = self.step(a)
                if done:
                    break


if __name__ == '__main__':
    env = Maze()
    env.after(100, env.test)
    env.mainloop()
