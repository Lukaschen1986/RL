# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


unit = 40
step = 20
maze_h = 4  # grid height
maze_w = 4  # grid width
obj_reward = 1
hell_reward = -1
walk_reword = -0.1


class Maze(tk.Tk):
    def __init__(self):
        super(Maze, self).__init__()
        self._build_maze()
        self.action_space = ['u', 'd', 'r', 'l']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(maze_h * unit, maze_h * unit))

    def _build_maze(self):
        # 棋盘大小
        self.canvas = tk.Canvas(self, bg='white',
                           height=maze_h * unit,
                           width=maze_w * unit)

        # 创建网格
        for c in range(0, maze_w * unit, step):
            x0, y0, x1, y1 = c, 0, c, maze_h * unit
            self.canvas.create_line(x0, y0, x1, y1)
            
        for r in range(0, maze_h * unit, step):
            x0, y0, x1, y1 = 0, r, maze_h * unit, r
            self.canvas.create_line(x0, y0, x1, y1)
            
        # create origin
        origin = np.array([step/2, step/2])
        # create margin
        margin = int(origin[0] / 1.5)
        
        # 每个陷阱的随机位置
        r_min = 1
        r_max = (maze_w*unit - step) / step
        loc = np.arange(r_min, r_max+1)
        loc_choice = [np.random.choice(loc, 1)[0] for i in range(20)]
        loc1_1, loc1_2, loc2_1, loc2_2, loc3_1, loc3_2, loc4_1, loc4_2, loc5_1, loc5_2, loc6_1, loc6_2, loc7_1, loc7_2, loc8_1, loc8_2, loc9_1, loc9_2, loc10_1, loc10_2 = loc_choice[0], loc_choice[1], loc_choice[2], loc_choice[3], loc_choice[4], loc_choice[5], loc_choice[6], loc_choice[7], loc_choice[8], loc_choice[9], loc_choice[10], loc_choice[11], loc_choice[12], loc_choice[13], loc_choice[14], loc_choice[15], loc_choice[16], loc_choice[17], loc_choice[18], loc_choice[19]

        # 创建陷阱（10）
        hell1_center = origin + np.array([step * loc1_1, step * loc1_2]) # 陷阱中心坐标
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - margin, hell1_center[1] - margin,
            hell1_center[0] + margin, hell1_center[1] + margin,
            fill='black')
        
        hell2_center = origin + np.array([step * loc2_1, step * loc2_2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - margin, hell2_center[1] - margin,
            hell2_center[0] + margin, hell2_center[1] + margin,
            fill='black')
        
        hell3_center = origin + np.array([step * loc3_1, step * loc3_2])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - margin, hell3_center[1] - margin,
            hell3_center[0] + margin, hell3_center[1] + margin,
            fill='black')
        
        hell4_center = origin + np.array([step * loc4_1, step * loc4_2])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - margin, hell4_center[1] - margin,
            hell4_center[0] + margin, hell4_center[1] + margin,
            fill='black')
        
        hell5_center = origin + np.array([step * loc5_1, step * loc5_2])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - margin, hell5_center[1] - margin,
            hell5_center[0] + margin, hell5_center[1] + margin,
            fill='black')
        
        hell6_center = origin + np.array([step * loc6_1, step * loc6_2])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - margin, hell6_center[1] - margin,
            hell6_center[0] + margin, hell6_center[1] + margin,
            fill='black')
        
        hell7_center = origin + np.array([step * loc7_1, step * loc7_2])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - margin, hell7_center[1] - margin,
            hell7_center[0] + margin, hell7_center[1] + margin,
            fill='black')
        
        hell8_center = origin + np.array([step * loc8_1, step * loc8_2])
        self.hell8 = self.canvas.create_rectangle(
            hell8_center[0] - margin, hell8_center[1] - margin,
            hell8_center[0] + margin, hell8_center[1] + margin,
            fill='black')
        
        hell9_center = origin + np.array([step * loc9_1, step * loc9_2])
        self.hell9 = self.canvas.create_rectangle(
            hell9_center[0] - margin, hell9_center[1] - margin,
            hell9_center[0] + margin, hell9_center[1] + margin,
            fill='black')
        
        hell10_center = origin + np.array([step * loc10_1, step * loc10_2])
        self.hell10 = self.canvas.create_rectangle(
            hell10_center[0] - margin, hell10_center[1] - margin,
            hell10_center[0] + margin, hell10_center[1] + margin,
            fill='black')
        
        # 创建目标
        hell_loc = np.concatenate((hell1_center, hell2_center, hell3_center, hell4_center, hell5_center, hell6_center, hell7_center, hell8_center, hell9_center, hell10_center)).reshape(10,2)
        obj_center = hell_loc[0,:]
        # 如果目标的坐标和任意一个陷阱重合，则重新计算，直到不一致为止
        while 2 in np.sum(obj_center == hell_loc, axis=1):
            obj_choice = np.random.choice(loc, 1)[0]
            obj_center = origin + step * obj_choice
        
        self.obj = self.canvas.create_rectangle(
            obj_center[0] - margin, obj_center[1] - margin,
            obj_center[0] + margin, obj_center[1] + margin,
            fill='red')

        # 创建智能体
        self.agent = self.canvas.create_oval(
            origin[0] - margin, origin[1] - margin,
            origin[0] + margin, origin[1] + margin,
            fill='yellow')

        # pack all
        self.canvas.pack()

    def reset(self):
        # create origin
        origin = np.array([step/2, step/2])
        # create margin
        margin = int(origin[0] / 1.5)
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_oval(
            origin[0] - margin, origin[1] - margin,
            origin[0] + margin, origin[1] + margin,
            fill='yellow')
        # return observation
        return self.canvas.coords(self.agent)

    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > step:
                base_action[1] -= step
        elif action == 1:   # down
            #if s[1] < (maze_h - 1) * unit:
            if s[1] < maze_h * unit - step:
                base_action[1] += step
        elif action == 2:   # right
            #if s[0] < (maze_w - 1) * unit:
            if s[0] < maze_w * unit - step:
                base_action[0] += step
        elif action == 3:   # left
            if s[0] > step:
                base_action[0] -= step                
                
        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent
        
        s_ = self.canvas.coords(self.agent)  # next state

        # reward function
        if s_ == self.canvas.coords(self.obj):
            reward = obj_reward
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3), self.canvas.coords(self.hell4), self.canvas.coords(self.hell5), self.canvas.coords(self.hell6), self.canvas.coords(self.hell7), self.canvas.coords(self.hell8), self.canvas.coords(self.hell9), self.canvas.coords(self.hell10)]:
            reward = hell_reward
            done = True
            s_ = 'terminal'
        else:
            reward = walk_reword
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


#def update():
#    for t in range(10):
#        s = env.reset()
#        while True:
#            env.render()
#            a = 1
#            s, r, done = env.step(a)
#            if done:
#                break

#if __name__ == '__main__':
#    env = Maze()
#    env.after(100, update)
#    env.mainloop()
