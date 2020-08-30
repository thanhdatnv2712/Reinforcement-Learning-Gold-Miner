import sys
from DQNModel import * # A class of creating a deep q-learning model
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
# from Memory import Memory # A class of creating a batch in order to store experiences for the training process
# from Memory import *
import pandas as pd
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

# Create header for saving DQN learning file
now = datetime.datetime.now() #Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", "Action", "Epsilon", "Done", "Termination_Code"] #Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv" 
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

#file_reward = open('sta_reward.txt', 'a')
#file_loss = open('sta_loss', 'a')

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
EPS_DECAY = 0.999
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent(state_size=INPUTNUM, action_size=ACTIONNUM, seed=0)

writer = SummaryWriter()

steps_done = 0

eps = EPS_START

scores = []
scores_window = deque(maxlen=100)
total_step = 0

for episode_i in range(0, N_EPISODE):
    # Choosing a map in the list
    # mapID = np.random.randint(1, 6) #Choosing a map ID from 5 maps in Maps folder randomly
    mapID = 2
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
    terminate = False #The variable indicates that the episode ends
    maxStep = minerEnv.state.mapInfo.maxStep #Get the maximum number of steps for each episode in training
    
    #Start an episde for training
    for step in range(0, maxStep):
        action = agent.act(s, eps)  # Getting an action from the DQN model from the state (s)
        minerEnv.step(str(action.item()))  # Performing the action in order to obtain the new state
        s_next = minerEnv.get_state()  # Getting a new state
        # print (s_next.shape, action)                
        reward = minerEnv.get_reward()  # Getting a reward
        terminate = minerEnv.check_terminate()  # Checking the end status of the episode
        # Add this transition to the memory batch
        agent.step(s, action, reward, s_next, terminate)
        total_reward = total_reward + reward #Plus the reward to the total rewad of the episode
        s = s_next #Assign the next state for the next step.
        # Saving data to file

        if terminate == True:
            #If the episode ends, then go to the next episode
            print ("Step: ", step, end= ' ')
            total_step += step
            break
    scores_window.append(total_reward)
    scores.append(total_reward)
    eps = max(EPS_END, EPS_DECAY*eps)
    writer.add_scalar('Reward/train', total_reward, episode_i)
    print ("\t| Total step: ", total_step, "\t| Total Reward: ", total_reward, "\t| Score: ", minerEnv.state.score)
    # print (total_step, eps)
    if episode_i % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_i, np.mean(scores_window)))
    if  np.mean(scores_window)>0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_i-100, np.mean(scores_window)))
        now = datetime.datetime.now()
        torch.save(agent.qnetwork_local.state_dict(), "TrainedModels/DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1) + ".pth")        
    if np.mean(scores_window)>=50:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_i-100, np.mean(scores_window)))
        now = datetime.datetime.now()
        torch.save(agent.qnetwork_local.state_dict(), "TrainedModels/DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1) + ".pth")
        break         
print('Complete')
