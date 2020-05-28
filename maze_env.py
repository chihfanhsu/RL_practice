import numpy as np
import time
import sys
import random

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

class Maze(tk.Tk, object):
    def __init__(self, UNIT = 40):
        super(Maze, self).__init__()
        self.action_space = ['0','1','2','3']
        self.title('Maze')
        self.MAZE_Limit = np.array([4, 4]) # (Width,Height)
        self.UNIT = UNIT
        self.geometry('{0}x{1}'.format(self.MAZE_Limit[0]*self.UNIT, self.MAZE_Limit[1]*self.UNIT))
        self.tot_states = self.MAZE_Limit[0]*self.MAZE_Limit[1]
        
        self.origin = np.array([0, 0]) # original loc of agent
        self.origin_center = self.origin.copy() + int(.5*self.UNIT)
        self.start_state =  np.array([1, 2]) # agent loc
        self.cur_state = self.start_state.copy() # agent loc
        self.oval_state = np.array([[0, 0],[3, 3]]) # reward!
        self.block_state = np.array([[2, 1]]) # you shall not pass!
        self.hell_state =  np.array([[4, 4]]) # reward!
        
        self.reward = np.ones((len(self.action_space),self.tot_states))*-1
        # set reward to oval state = 0
        for a in range(len(self.action_space)):
            for t in range(len(self.oval_state)):
                if (self.position2state(self.oval_state[t])<self.tot_states):
                    self.reward[a,self.position2state(self.oval_state[t])] = 0
        
        # set reward to block state = 0
        for a in range(len(self.action_space)):
            for t in range(len(self.block_state)):
                if (self.position2state(self.block_state[t])<self.tot_states):
                    self.reward[a,self.position2state(self.block_state[t])] = 0

        self._build_maze()
    
    def fresh_figure(self):
        # calcualte the coordinates for the UI
        cur_center = self.origin_center + self.start_state.copy()*self.UNIT
        oval_center = self.origin_center + self.oval_state.copy()*self.UNIT
        block_center = self.origin_center + self.block_state.copy()*self.UNIT
        hell_center = self.origin_center + self.hell_state.copy()*self.UNIT
        # draw oval figure
        for i in range(len(self.oval_state)):
            self.oval = self.canvas.create_oval(
                oval_center[i,0] - 15, oval_center[i,1] - 15,
                oval_center[i,0] + 15, oval_center[i,1] + 15,
                fill='yellow')
        
        # draw hell figure
        for i in range(len(self.hell_state)):
            self.hell = self.canvas.create_oval(
                hell_center[i,0] - 15, hell_center[i,1] - 15,
                hell_center[i,0] + 15, hell_center[i,1] + 15,
                fill='red')
        
        # draw agent figure
        self.rect = self.canvas.create_rectangle(
            cur_center[0] - 15, cur_center[1] - 15,
            cur_center[0] + 15, cur_center[1] + 15,
            fill='red')
    
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.MAZE_Limit[1]*self.UNIT,
                                width=self.MAZE_Limit[0]*self.UNIT)
        # create grids
        for c in range(0, self.MAZE_Limit[0]*self.UNIT, self.UNIT):
            x0, y0, x1, y1 = c, 0, c, self.MAZE_Limit[1] * self.UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.MAZE_Limit[1]*self.UNIT, self.UNIT):
            x0, y0, x1, y1 = 0, r, self.MAZE_Limit[0]*self.UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        
        block_center = self.origin_center + self.block_state.copy()*self.UNIT
        # draw block state
        for i in range(len(self.block_state)):
            self.cross = self.canvas.create_line(block_center[i,0] - 15,
                                                 block_center[i,1] - 15,
                                                 block_center[i,0] + 15,
                                                 block_center[i,1] + 15)
            self.cross = self.canvas.create_line(block_center[i,0] + 15,
                                                 block_center[i,1] - 15,
                                                 block_center[i,0] - 15,
                                                 block_center[i,1] + 15)
        
        self.fresh_figure()
        # pack all
        self.canvas.pack()
    
    def render(self):
        self.update()
        time.sleep(0.01)
    
    def reset(self):
        self.render()
        self.canvas.delete(self.rect)
        self.canvas.delete(self.oval)
        self.canvas.delete(self.hell)
        self.cur_state = self.start_state.copy()
        self.fresh_figure()
        return True
    
    def take_action(self, action, show_animate = False):
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
            
        # move agent figure
        if show_animate:
            next_center = (next_state - self.cur_state)*self.UNIT
            self.canvas.move(self.rect, next_center[0], next_center[1])  # move agent
        
        reward = self.reward[action,self.position2state(self.cur_state)].copy()
        self.cur_state = next_state.copy()
        # termination check
        if np.any(np.all(self.cur_state == self.oval_state, axis=1)):
            next_state = self.cur_state.copy()
            return True, reward
        
        if np.any(np.all(self.cur_state == self.hell_state, axis=1)):
            next_state = self.cur_state.copy()
            return True, reward

        return False, reward
    
    def position2state(self, pos, inv = False):
        if inv == False:
            in_pos = np.array(pos)
            if in_pos.shape == (2,):
                return in_pos[1]*self.MAZE_Limit[0] + in_pos[0]
            else:
                print("Error corrdinate!")
                return False
        else:
            in_pos = np.array(pos)
            if in_pos.shape == ():
                return np.array([np.int(np.floor(in_pos/self.MAZE_Limit[0])), np.int(in_pos%self.MAZE_Limit[0])])
            else:
                print("Error state number")
                return False
    
    # get environment model
    def get_env_model(self):
        p = np.zeros((len(self.action_space), self.MAZE_Limit[0]*self.MAZE_Limit[1], self.MAZE_Limit[0]*self.MAZE_Limit[1]))
        for i in range(self.MAZE_Limit[0]):
            for j in range(self.MAZE_Limit[1]):
                for a in range(len(self.action_space)):
                    self.cur_state = np.array([i,j]).copy()
                    self.take_action(a, show_animate=False)
                    p[a, self.position2state([i,j]), self.position2state(self.cur_state)] = 1
        
        # set oval state
        for a in range(len(self.action_space)):
            for t in range(len(self.oval_state)):
                if (self.position2state(self.oval_state[t])<self.tot_states):
                    p[a,self.position2state(self.oval_state[t])] = 0
        
        # set block state
        for a in range(len(self.action_space)):
            for t in range(len(self.block_state)):
                if (self.position2state(self.block_state[t])<self.tot_states):
                    p[a,self.position2state(self.block_state[t])] = 0
        
        return p