import cv2
import gym
import numpy as np
import copy
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT 

def create_env(env_id):
    if env_id == 'gridworld':
        env = ExplorationGame(40)
    elif env_id == 'gridworldwall':
        env = ExplorationGameWall(40)
    elif env_id == 'SuperMarioBros-v3':
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
    else:
        env = gym.make(env_id)
    return env

def _process_frame(frame, env_name):
    if 'gridworld' in env_name:
        return frame

    elif 'Pong' in env_name:
        frame = frame[35:195]
        frame = frame[::4,::4,0]
        frame[frame == 144] = 0
        frame[frame == 109] = 0
        frame[frame != 0] = 1
        return frame.astype(np.float32)

    elif 'Montezuma' in env_name:
        frame = frame[35:195]
        frame = frame[::4,::4,0]
        frame = frame/255
        return frame.astype(np.float32)

    elif 'Breakout' in env_name:
        frame = frame[34:34 + 160, :160]
        frame = cv2.resize(frame, (40, 40))
        frame = frame[:,:,0]
        frame = frame.astype(np.float32)
        frame[frame==142] = 0.5
        frame[frame>1] = 1
        return frame

    elif 'Mario' in env_name:
        frame = cv2.resize(frame, (40, 40))
        frame = frame[:,:,0]
        frame = frame/255
        return frame.astype(np.float32)

class ExplorationGame(object):
    def __init__(self, size=40):
        self.size = size
        self.action_space = 4
    
    def reset(self):
        self.state = np.zeros((self.size, self.size))
        self.steps = 0
        self.state[0,0] = 1
        self.position = [0,0]
        self.path = [copy.copy(self.position)]
        return self.state.astype(np.float32)
    
    def step(self, action):
        self.steps += 1
        
        if action == 0: # up
            self.position[0] = max(self.position[0]-1,0)
        elif action == 1: # down
            self.position[0] = min(self.position[0]+1,self.size-1)
        elif action == 2: # left
            self.position[1] = max(self.position[1]-1,0)
        elif action == 3: # right
            self.position[1] = min(self.position[1]+1,self.size-1)
        self.state[self.position[0],self.position[1]] = 1
        self.path.append(copy.copy(self.position))
        
        done = False
        reward = 0
        info = 0
        if self.steps == 400:
            #if self.state[self.size-1, self.size-1] == 1:
            #    reward = 20
            done = True
            info = len(np.unique(self.path,axis=0))
        return self.state.astype(np.float32), reward, done, info


class ExplorationGameWall(object):
    def __init__(self, size=40):
        self.size = size
        self.action_space = 4

    def reset(self):
        self.state = np.zeros((self.size, self.size))
        self.state[1:,1:] = 2
        self.steps = 0
        self.state[0,0] = 1
        self.position = [0,0]
        self.path = [copy.copy(self.position)]
        return self.state.astype(np.float32)

    def step(self, action):
        self.steps += 1
        current_position = copy.copy(self.position)

        if action == 0: # up
            self.position[0] = max(self.position[0]-1,0)
        elif action == 1: # down
            self.position[0] = min(self.position[0]+1,self.size-1)
        elif action == 2: # left
            self.position[1] = max(self.position[1]-1,0)
        elif action == 3: # right
            self.position[1] = min(self.position[1]+1,self.size-1)

        if self.state[self.position[0],self.position[1]] == 2:  # hit the wall
             self.position = current_position

        if self.position[0] != 0:
            self.state[self.position[0],self.position[1]] = 1
        self.path.append(copy.copy(self.position))

        done = False
        reward = 0
        info = 0
        if self.position == [0, self.size-1]:
            reward = 40
            done = True
            info = len(np.unique(self.path,axis=0))
        elif self.steps == 400:
            done = True
            info = len(np.unique(self.path,axis=0))

        return self.state.astype(np.float32), reward, done, info
