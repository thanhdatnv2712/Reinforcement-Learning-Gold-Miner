import sys
from DQNModel import * # A class of creating a deep q-learning model
from BFS_explore import BFS_energy
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
# from Memory import Memory # A class of creating a batch in order to store experiences for the training process
# from Memory import *
import datetime 
import numpy as np
from collections import deque

import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

# Parameters for training a DQN model
N_EPISODE = 1000000 #The number of episodes for training
MAX_STEP = 1000   #The number of steps for each episode
BATCH_SIZE = 128   #The number of experiences for each replay 
MEMORY_SIZE = 100000 #The size of the batch for storing experiences
SAVE_NETWORK = 100  # After this number of episodes, the DQN model is saved for testing later. 
INITIAL_REPLAY_SIZE = 1024 #The number of experiences are stored in the memory batch before starting replaying
# INPUTNUM = 198 #The number of input values for the DQN model
INPUTNUM = 192
ACTIONNUM = 6  #The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map

# Initialize environment
minerEnv = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game

train = False #The variable is used to indicate that the replay starts, and the epsilon starts decrease.
#Training Process
#the main part of the deep-q learning agorithm 

GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
BFS_EPS = 1152

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent(state_size=INPUTNUM, action_size=ACTIONNUM, seed=0)
BFS_agent = BFS_energy()

writer = SummaryWriter()

steps_done = 0
total_step = 0
eps = EPS_START

scores = []
scores_window = deque(maxlen=100)

for episode_i in range(0, N_EPISODE):
    # Choosing a map in the list
    mapID = np.random.randint(1, 6) #Choosing a map ID from 5 maps in Maps folder randomly
    # mapID = 2
    posID_x = np.random.randint(MAP_MAX_X) #Choosing a initial position of the DQN agent on X-axes randomly
    posID_y = np.random.randint(MAP_MAX_Y) #Choosing a initial position of the DQN agent on Y-axes randomly
    #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
    request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100") 
    #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
    minerEnv.send_map_info(request)
    # Getting the initial state
    minerEnv.reset() #Initialize the game environment
    s = minerEnv.get_state()#Get the state after reseting. 
                            #This function (get_state()) is an example of creating a state for the DQN model 
    total_reward = 0 #The amount of rewards for the entire episode
    action_game = []
    local_step = 0
    terminate = False #The variable indicates that the episode ends
    maxStep = minerEnv.state.mapInfo.maxStep #Get the maximum number of steps for each episode in training
    #Start an episde for training
    for step in range(0, maxStep):
        norm = np.linalg.norm(s)
        s_norm = s/norm
        action = 4
        if episode_i <= BFS_EPS:
            action = BFS_agent.act(s)
            action = np.int64(action) 
        else:
            action = agent.act(s_norm, eps)  # Getting an action from the DQN model from the state (s)
        action_game.append(action)
        minerEnv.step(str(action))  # Performing the action in order to obtain the new state
        s_next = minerEnv.get_state()  # Getting a new state
        norm = np.linalg.norm(s_next)
        s_norm_next = s_next/norm
        # print (s_next.shape, action)                
        reward = minerEnv.get_reward()  # Getting a reward
        terminate = minerEnv.check_terminate()  # Checking the end status of the episode
        # Add this transition to the memory batch
        agent.step(s_norm, action, reward, s_norm_next, terminate)
        total_reward = total_reward + reward #Plus the reward to the total rewad of the episode
        s = s_next #Assign the next state for the next step.
        # Saving data to file

        if terminate == True:
            #If the episode ends, then go to the next episode
            print ("Episode: {} Step: {}".format(episode_i, step), end='')
            local_step = step
            total_step += step
            break
    scores_window.append(total_reward)
    scores.append(total_reward)
    eps = max(EPS_END, EPS_DECAY*eps)
    writer.add_scalar('Reward/train', total_reward, episode_i)
    writer.add_scalar('Score/train', minerEnv.state.score, episode_i)
    writer.add_scalar('Step per Episode/train', local_step, episode_i)
    print ("\t| Total step: ", total_step, "\t| Total Reward: ", total_reward, "\t| Score: ", minerEnv.state.score)
    print (action_game)
    if episode_i % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_i, np.mean(scores_window)))
    if  (np.mean(scores_window)>350 and episode_i > 5) or episode_i %50 == 0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_i-100, np.mean(scores_window)))
        now = datetime.datetime.now()
        torch.save(agent.qnetwork_local.state_dict(), "TrainedModels/DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1) + ".pth")        
    if np.mean(scores_window)>=1500 and episode_i > BFS_EPS:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_i-100, np.mean(scores_window)))
        now = datetime.datetime.now()
        torch.save(agent.qnetwork_local.state_dict(), "TrainedModels/DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1) + ".pth")
        break         
print('Complete')
