import numpy as np
import time
import sys
import random

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['0','1','2','3']
        self.title('Maze')
        self.MAZE_Limit = np.array([6, 8]) # (Width,Height)
        self.geometry('{0}x{1}'.format(self.MAZE_Limit[0] * UNIT, self.MAZE_Limit[1] * UNIT))
        self.origin = np.array([0, 0]) # original loc of agent
        self.cur_state = np.array([0, 0]) # agent loc
        self.oval_state = np.array([4, 5]) # reward!
        self.block_state = np.array([[1, 3], [2, 4]]) # you shall no pass!
        self.hell_state =  np.array([4, 6]) # reward!
        self._build_maze()
    
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.MAZE_Limit[1] * UNIT,
                                width=self.MAZE_Limit[0] * UNIT)
        # create grids
        for c in range(0, self.MAZE_Limit[0] * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.MAZE_Limit[1] * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.MAZE_Limit[1] * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.MAZE_Limit[0] * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # set origin and oval (goal)
        self.cur_state = self.origin.copy()
        
        # draw environment
        origin_center = self.origin.copy() + int(.5*UNIT)
        
        #print(origin_center)
        cur_center = origin_center
        oval_center = origin_center + self.oval_state.copy()*UNIT
        block_center = origin_center + self.block_state.copy()*UNIT
        hell_center = origin_center + self.hell_state.copy()*UNIT

        # create goal
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        
        # create hell
        self.hell = self.canvas.create_oval(
            hell_center[0] - 15, hell_center[1] - 15,
            hell_center[0] + 15, hell_center[1] + 15,
            fill='red')

        # create agent
        self.rect = self.canvas.create_rectangle(
            cur_center[0] - 15, cur_center[1] - 15,
            cur_center[0] + 15, cur_center[1] + 15,
            fill='red')
        
        for i in range(len(self.block_state)):
            self.cross = self.canvas.create_line(block_center[i,0]-15, block_center[i,1]-15, block_center[i,0]+15, block_center[i,1]+15)
            self.cross = self.canvas.create_line(block_center[i,0]+15, block_center[i,1]-15, block_center[i,0]-15, block_center[i,1]+15)

        # pack all
        self.canvas.pack()

    def reset(self):
        # print('----reset env')
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect)
        self.canvas.delete(self.oval)
        self.canvas.delete(self.hell)
        self.cur_state = self.origin.copy()
        
        origin_center = self.origin.copy() + int(.5*UNIT)
        cur_center = origin_center
        oval_center = origin_center + self.oval_state.copy()*UNIT
        hell_center = origin_center + self.hell_state.copy()*UNIT
        
        # create oval
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        
        # create hell
        self.hell = self.canvas.create_oval(
            hell_center[0] - 15, hell_center[1] - 15,
            hell_center[0] + 15, hell_center[1] + 15,
            fill='red')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            cur_center[0] - 15, cur_center[1] - 15,
            cur_center[0] + 15, cur_center[1] + 15,
            fill='red')
        
        return True

    def take_action(self, action):
        next_state = self.cur_state.copy()
        invalid_move = False
        
        if action == 0: # up
            next_state[1] = self.cur_state[1] - 1
            if (next_state[1] < 0):# boundary check
                invalid_move  = True
        
        elif action == 1: # right
            next_state[0] = self.cur_state[0] + 1
            if (next_state[0] > self.MAZE_Limit[0]-1):# boundary check
                invalid_move  = True
                
        elif action == 2: # down
            next_state[1] = self.cur_state[1] + 1   
            if (next_state[1] > self.MAZE_Limit[1]-1):# boundary check
                invalid_move  = True
                
        elif action == 3: # left
            next_state[0] = self.cur_state[0] - 1
            if (next_state[0] < 0):# boundary check
                invalid_move  = True
       
        if (np.any(np.all(next_state == self.block_state, axis=1))): # block check
                invalid_move  = True
    
        if (invalid_move): # don't move
            next_state = self.cur_state.copy()
            
        # draw animation
        next_center = (next_state-self.cur_state)*UNIT
                
        self.canvas.move(self.rect, next_center[0], next_center[1])  # move agent
        self.cur_state = next_state.copy()
       
        if ((self.cur_state ==self.oval_state).all()):
            print ("you got it !!!")
            self.reset()
            
        if ((self.cur_state ==self.hell_state).all()):
            print ("you die !!!")
            self.reset()
                
        return True

    def render(self):
        time.sleep(0.001)
        self.update()
