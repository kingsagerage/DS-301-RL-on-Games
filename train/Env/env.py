import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from perlin_noise import PerlinNoise3

class CustomEnv3D(gym.Env):
    def __init__(self):
        super(CustomEnv3D, self).__init__()
        # Define action and observation spaces
        self.grid_size = 1024
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(13) 

        self.laction1 = -1
        self.laction2 = -1
        # Initialize agent positions and orientations
        self.agent1_pos = [512, 512, 512]  # Initial position of agent 1
        self.agent2_pos = [256, 256, 256]  # Initial position of agent 2
        self.agent1_facing = [0, 0, 1]  # Initial facing direction of agent 1 (towards positive z-axis)
        self.agent2_facing = [0, 0, -1]  # Initial facing direction of agent 2 (towards negative z-axis)

        self.agent_size = 20

        self.reward1 = 0
        self.reward2 = 0

    def get_direction(self, agentAttacking, agentDefending):
        # Calculate direction from agentAttacking to agentDefending
        diff = np.array(agentDefending) - np.array(agentAttacking)
        return np.degrees(np.arctan2(diff[1], diff[0]))

    def step(self, action1, action2):
        self.reward1 -= 1
        self.reward2 -= 1
        self.laction1 = action1
        self.laction2 = action2

        #Adding premove positions to account for player 1 moving first (so that if P2 shoots they will end up hitting the other agent even if P1 moved first)
        self.agent1_prepos = self.agent1_pos

        # Agent 1 actions
        self._take_action(self.agent1_pos, action1, self.agent1_facing, self.agent2_pos, 1)

        # Agent 2 actions
        self._take_action(self.agent2_pos, action2, self.agent2_facing, self.agent1_prepos, 2)

        # Clip agent positions to stay within the grid boundaries
        self.agent1_pos = np.clip(self.agent1_pos, [0, 0, 0], [1023, 1023, 1023])
        self.agent2_pos = np.clip(self.agent2_pos, [0, 0, 0], [1023, 1023, 1023])

        observation1 = self._get_observation(self.agent1_pos, self.agent2_pos)
        observation2 = self._get_observation(self.agent2_pos, self.agent1_pos)
        done = self.reward1 >= 1000 or self.reward2 >= 1000

        return observation1, observation2, self.reward1, self.reward2, done, {}

    def _take_action(self, agent_pos, action, agent_facing, enemy_pos, agent_number):
        #Readjusting movement speed and size for the 3D map so agents have an easier time getting to one another
        if action == 0:  # Move up
            agent_pos[2] -= 7
        elif action == 1:  # Move down
            agent_pos[2] += 7
        elif action == 2:  # Move left
            agent_pos[1] -= 7
        elif action == 3:  # Move right
            agent_pos[1] += 7
        elif action == 4:  # Move forward
            agent_pos[0] += 7
        elif action == 5:  # Move backward
            agent_pos[0] -= 7
        elif action == 6:  # Shoot
            distancel = self.distance()

            if distancel < 200:  # If distance is less than 100, hit the target
                if agent_number == 1:
                    self.reward1 += 1000  # Reward for agent 1 being hit
                else:
                    self.reward2 += 1000  # Reward for agent 1 being hit
            else:
                if agent_number == 1:
                    self.reward1 -= 1  # Penalty for missing the target
                else:
                    self.reward2 -= 1
        elif action == 7:  # Rotate clockwise along Z
            self.agent2_facing[0] += 10
        elif action == 8:  # Rotate counter-clockwise along Z
            self.agent2_facing[0] -= 10
        elif action == 9:  # Rotate clockwise along X
            self.agent2_facing[1] += 10
        elif action == 10:  # Rotate counter-clockwise along X
            self.agent2_facing[1] -= 10
        elif action == 11:  # Rotate clockwise along Y
            self.agent2_facing[2] += 10
        elif action == 12:  # Rotate counter-clockwise along Y
            self.agent2_facing[2] -= 10
    def distance(self):
        z_diff = self.agent1_pos[0] - self.agent2_pos[0]
        x_diff = self.agent1_pos[1] - self.agent2_pos[1]
        y_diff = self.agent1_pos[2] - self.agent2_pos[2]
        dist = math.sqrt(z_diff**2 + y_diff**2 + x_diff**2)
        return dist
    def _get_observation(self, agent_pos, enemy_pos):
        direction = self.get_direction(agent_pos, enemy_pos)
        distance = np.linalg.norm(np.array(enemy_pos) - np.array(agent_pos))
        return direction, distance

    def reset(self):
        self.agent1_pos = [768, 512, 512] 
        self.agent2_pos = [256, 512, 512]
        self.agent1_facing = [0, 0, 1]
        self.agent2_facing = [0, 0, -1]
        self.reward1 = 0
        self.reward2 = 0
        return self._get_observation(self.agent1_pos, self.agent2_pos), self._get_observation(self.agent2_pos, self.agent1_pos)


class CustomEnv3DWithraytracing(gym.Env):
    def __init__(self):
        super(CustomEnv3D, self).__init__()
        # Define action and observation spaces
        self.grid_size = 1024
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(13) 

        self.laction1 = -1
        self.laction2 = -1
        # Initialize agent positions and orientations
        self.agent1_pos = [512, 512, 512]  # Initial position of agent 1
        self.agent2_pos = [256, 256, 256]  # Initial position of agent 2
        self.agent1_facing = [0, 0, 1]  # Initial facing direction of agent 1 (towards positive z-axis)
        self.agent2_facing = [0, 0, -1]  # Initial facing direction of agent 2 (towards negative z-axis)

        self.agent_size = 20

        self.reward1 = 0
        self.reward2 = 0

        self.noise = PerlinNoise3(100)

    def get_direction(self, agentAttacking, agentDefending):
        # Calculate direction from agentAttacking to agentDefending
        diff = np.array(agentDefending) - np.array(agentAttacking)
        return np.degrees(np.arctan2(diff[1], diff[0]))

    def step(self, action1, action2):
        self.reward1 -= 1
        self.reward2 -= 1
        self.laction1 = action1
        self.laction2 = action2

        #Adding premove positions to account for player 1 moving first (so that if P2 shoots they will end up hitting the other agent even if P1 moved first)
        self.agent1_prepos = self.agent1_pos

        # Agent 1 actions
        self._take_action(self.agent1_pos, action1, self.agent1_facing, self.agent2_pos, 1)

        # Agent 2 actions
        self._take_action(self.agent2_pos, action2, self.agent2_facing, self.agent1_prepos, 2)

        # Clip agent positions to stay within the grid boundaries
        self.agent1_pos = np.clip(self.agent1_pos, [0, 0, 0], [1023, 1023, 1023])
        self.agent2_pos = np.clip(self.agent2_pos, [0, 0, 0], [1023, 1023, 1023])

        observation1 = self._get_observation(self.agent1_pos, self.agent2_pos)
        observation2 = self._get_observation(self.agent2_pos, self.agent1_pos)
        done = self.reward1 >= 1000 or self.reward2 >= 1000

        return observation1, observation2, self.reward1, self.reward2, done, {}

    def _take_action(self, agent_pos, action, agent_facing, enemy_pos, agent_number):
        #Readjusting movement speed and size for the 3D map so agents have an easier time getting to one another
        if action == 0:  # Move up
            agent_pos[2] -= 7
        elif action == 1:  # Move down
            agent_pos[2] += 7
        elif action == 2:  # Move left
            agent_pos[1] -= 7
        elif action == 3:  # Move right
            agent_pos[1] += 7
        elif action == 4:  # Move forward
            agent_pos[0] += 7
        elif action == 5:  # Move backward
            agent_pos[0] -= 7
        elif action == 6:  # Shoot
            distancel = self.distance()

            if distancel < 200:  # If distance is less than 100, hit the target
                if agent_number == 1:
                    self.reward1 += 1000  # Reward for agent 1 being hit
                else:
                    self.reward2 += 1000  # Reward for agent 1 being hit
            else:
                if agent_number == 1:
                    self.reward1 -= 1  # Penalty for missing the target
                else:
                    self.reward2 -= 1
        elif action == 7:  # Rotate clockwise along Z
            self.agent2_facing[0] += 10
        elif action == 8:  # Rotate counter-clockwise along Z
            self.agent2_facing[0] -= 10
        elif action == 9:  # Rotate clockwise along X
            self.agent2_facing[1] += 10
        elif action == 10:  # Rotate counter-clockwise along X
            self.agent2_facing[1] -= 10
        elif action == 11:  # Rotate clockwise along Y
            self.agent2_facing[2] += 10
        elif action == 12:  # Rotate counter-clockwise along Y
            self.agent2_facing[2] -= 10
    def distance(self):
        z_diff = self.agent1_pos[0] - self.agent2_pos[0]
        x_diff = self.agent1_pos[1] - self.agent2_pos[1]
        y_diff = self.agent1_pos[2] - self.agent2_pos[2]
        dist = math.sqrt(z_diff**2 + y_diff**2 + x_diff**2)
        return dist
    def _get_observation(self, agent_pos, enemy_pos):
        direction = self.get_direction(agent_pos, enemy_pos)
        distance = np.linalg.norm(np.array(enemy_pos) - np.array(agent_pos))
        return direction, distance

    def reset(self):
        self.agent1_pos = [768, 512, 512] 
        self.agent2_pos = [256, 512, 512]
        self.agent1_facing = [0, 0, 1]
        self.agent2_facing = [0, 0, -1]
        self.reward1 = 0
        self.reward2 = 0
        return self._get_observation(self.agent1_pos, self.agent2_pos), self._get_observation(self.agent2_pos, self.agent1_pos)



actions = ['up','down','left','right','Rotate clockwise','Rotate counter-clockwise','shoot']
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation spaces
        self.grid_size = 1024
        self.grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(7) 

        self.laction1 = -1
        self.laction2 = -1
        # Initialize agent positions and orientations
        self.agent1_pos = [512, 512]  # Initial position of agent 1
        self.agent2_pos = [256, 256]  # Initial position of agent 2
        self.agent1_facing = 90 
        self.agent2_facing = 270

        self.agent_size = 20

        self.reward1 = 0
        self.reward2 = 0

        #self.fig, self.ax = plt.subplots()
        #self.im = self.ax.imshow(self.grid)

    def get_direction(self,agentAttacking,agentDefending):
        x_diff = agentDefending[0] - agentAttacking[0]
        y_diff = agentDefending[1] - agentAttacking[1]
        return math.degrees(math.atan2(y_diff, x_diff))

    def step(self, action1, action2):
        self.reward1 -= 1
        self.reward2 -= 1
        self.laction1 = action1
        self.laction2 = action2
        if action1 == 0:  # Move up
            self.agent1_pos[1] -= 1
        elif action1 == 1:  # Move down
            self.agent1_pos[1] += 1
        elif action1 == 2:  # Move left
            self.agent1_pos[0] -= 1
        elif action1 == 3:  # Move right
            self.agent1_pos[0] += 1
        elif action1 == 4:  # Rotate clockwise
            self.agent1_facing += 10
        elif action1 == 5:  # Rotate counter-clockwise
            self.agent1_facing -= 10
        elif action1 == 6:  # Shoot
            diff = self.agent1_facing - self.get_direction(self.agent1_pos,self.agent2_pos)
            if diff!= 0:
                self.reward1 += 360/abs(diff)
            else:
                self.reward1 += 400

        if action2 == 0:  # Move up
            self.agent2_pos[1] -= 1
        elif action2 == 1:  # Move down
            self.agent2_pos[1] += 1
        elif action2 == 2:  # Move left
            self.agent2_pos[0] -= 1
        elif action2 == 3:  # Move right
            self.agent2_pos[0] += 1
        elif action2 == 4:  # Rotate clockwise
            self.agent2_facing += 10
        elif action2 == 5:  # Rotate counter-clockwise
            self.agent2_facing -= 10
        elif action2 == 6:  # Shoot
            diff = self.agent2_facing - self.get_direction(self.agent2_pos,self.agent1_pos)
            if diff!= 0:
                self.reward1 += 360/abs(diff)
            else:
                self.reward1 += 400

        self.agent1_pos[0] = max(0, min(self.agent1_pos[0], 1023))
        self.agent1_pos[1] = max(0, min(self.agent1_pos[1], 1023))
        self.agent2_pos[0] = max(0, min(self.agent2_pos[0], 1023))
        self.agent2_pos[1] = max(0, min(self.agent2_pos[1], 1023))

        observation1 = self._get_observation1()
        observation2 = self._get_observation2()
        done = self.reward1 >= 1000 or self.reward2 >= 1000

        return observation1 , observation2, self.reward1, self.reward2, done, {}
    def _get_grid(self):
        self.grid.fill(0)

        # Draw agents on the grid
        self.grid[self.agent1_pos[1] - self.agent_size // 2:self.agent1_pos[1] + self.agent_size // 2,
                  self.agent1_pos[0] - self.agent_size // 2:self.agent1_pos[0] + self.agent_size // 2] = [255, 0, 0]  # Agent 1 (red)
        self.grid[self.agent2_pos[1] - self.agent_size // 2:self.agent2_pos[1] + self.agent_size // 2,
                  self.agent2_pos[0] - self.agent_size // 2:self.agent2_pos[0] + self.agent_size // 2] = [0, 0, 255]  # Agent 2 (blue)

        return self.grid
    def distance(self, pos1, pos2):
        # Calculate Euclidean distance between pos1 and pos2
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
    def _get_observation1(self):
        # Calculate direction and distance between agents
        direction = int(self.get_direction(self.agent1_pos, self.agent2_pos))
        distance = int(self.distance(self.agent1_pos, self.agent2_pos))
        return direction, distance
    def _get_observation2(self):
        # Calculate direction and distance between agents
        direction = int(self.get_direction(self.agent2_pos, self.agent1_pos))
        distance = int(self.distance(self.agent2_pos, self.agent1_pos))
        return direction, distance
    def reset(self):
        self.agent1_pos = [768, 512] 
        self.agent2_pos = [256, 512]  
        self.agent1_facing = 0
        self.agent2_facing = 180 
        self.reward1 = 0
        self.reward2 = 0
        return self._get_observation1(), self._get_observation2()
    def render(self, mode='human'):
        # Visualize the environment
        self.im.set_data(self._get_grid())
        self.ax.set_title(f'Action 1: {actions[self.laction1]}, Action 2: {actions[self.laction2]}')
        plt.pause(0.1)

def createEnv(env):
    if(env == "3D"): return CustomEnv3D()
    else: return CustomEnv()