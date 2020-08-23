import sys
from DQNModel import DQN # A class of creating a deep q-learning model
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from Memory import Memory # A class of creating a batch in order to store experiences for the training process
import heapq
import pandas as pd
import datetime 
import numpy as np
from queue import Queue 
from modelBFS_energy import BFS_energy
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

# Parameters for training a DQN model
N_EPISODE = 10000 #The number of episodes for training
MAX_STEP = 100   #The number of steps for each episode
BATCH_SIZE = 32   #The number of experiences for each replay 
MEMORY_SIZE = 100000 #The size of the batch for storing experiences
SAVE_NETWORK = 100  # After this number of episodes, the DQN model is saved for testing later. 
INITIAL_REPLAY_SIZE = 1000 #The number of experiences are stored in the memory batch before starting replaying
# INPUTNUM = 1MAP_MAX_Y8 #The number of input values for the DQN model
INPUTNUM = (21,9, 20)
ACTIONNUM = 6  #The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map

# Initialize a DQN model and a memory batch for storing experiences
DQNAgent = DQN(INPUTNUM, ACTIONNUM)
memory = Memory(MEMORY_SIZE)

# Initialize environment
minerEnv = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game

train = False #The variable is used to indicate that the replay starts, and the epsilon starts decrease.
# num_step = np.zeros(shape = (MAP_MAX_X,MAP_MAX_Y))
# value = np.zeros(shape = (MAP_MAX_X,MAP_MAX_Y),dtype=int)
# visited = np.zeros(shape = (MAP_MAX_X,MAP_MAX_Y))
# all_lay = [] # luu vi tri dam lay
# all_gold = [] #luu vi tri vang
# next_time = [-5,-20,-40,-100,-120]  # ton hao energy khi di vao dam lay
# trace = [["" for x in range(MAP_MAX_Y)] for y in range(MAP_MAX_X)] # truy vet cach di
# inf = 10000000000000000






modelBFS = BFS_energy()
all_map = [[],[],[],[],[],[]]
print(all_map[1])
for e in range(3):
    for idmap in range(1,6):
        x = np.random.randint(MAP_MAX_X) 
        y = np.random.randint(MAP_MAX_Y)
        print('map',idmap)
        request = ("map" + str(idmap) + "," + str(x) + "," + str(y) + ",50,100")
        minerEnv.send_map_info(request)
        minerEnv.reset()
        s_next = minerEnv.get_state()
        total_reward = 0

        for step in range(0,MAX_STEP):
            action = modelBFS.act(s_next)
            minerEnv.step(action)
            reward = minerEnv.get_reward()
            s_next = minerEnv.get_state()
            energy = s_next[...,MAP_MAX_X*MAP_MAX_Y+2:MAP_MAX_X*MAP_MAX_Y+3]
            print('energy: ',energy)
            print('reward : ',reward,end= ' ')
            total_reward += reward
            print('total reward : ' ,total_reward)
            print('------------------------------------------------------------------')
            terminate = minerEnv.check_terminate()
            print('num lay :',modelBFS.cnt)
            if terminate == True:
                break
        all_map[idmap].append(total_reward)
        # minerEnv.step(str(tr))
# 
#

for i in range(1,6):
    print('map',i,'=',all_map[i])





    
