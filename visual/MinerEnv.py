from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
import numpy as np
import random
from GAME_SOCKET import GameSocket #in testing version, please use GameSocket instead of GameSocketDummy
from MINER_STATE import State

TreeID = 1
TrapID = 2
SwampID = 3


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        
        self.score_pre = self.state.score#Storing the last score for designing the reward function
        
    def start(self): #connect to server
        self.socket.connect()

    def end(self): #disconnect server
        self.socket.close()

    def send_map_info(self, request):#tell server which map to run
        self.socket.send(request)

    def reset(self): #start new game
        try:
            message = self.socket.receive() #receive game info from server
            print(message)
            self.state.init_state(message) #init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            #print("New state: ", message)
            self.state.update_state(message) #update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        # Building the map
        mapOrigin = np.zeros([self.state.mapInfo.max_y + 1, self.state.mapInfo.max_x + 1], dtype=np.int32)
        mapID = [0, 0, 0, 0, 0]
        mapID[random.randint(1, 5)-1] = 1
        # mapID[1] = 1
        view = np.zeros([self.state.mapInfo.max_y + 1, self.state.mapInfo.max_x + 1])
        #print (view.shape)
        view[self.state.y - 1, self.state.x - 1] = 1
        gold = np.zeros([1, 24])
        idx = 0
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    mapOrigin[j, i] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    mapOrigin[j, i] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID: # Swamp
                    mapOrigin[j, i] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    #print (j, i)
                    mapOrigin[j, i] = self.state.mapInfo.gold_amount(i, j)
                    gold[0, idx] = self.state.mapInfo.gold_amount(i, j)
                    idx += 1
        players_info = [["player", self.state.score, self.state.energy, "ON"]]
        envs_dict = {
            "map" : mapOrigin.tolist(),
            "agent": [self.state.x, self.state.y],
            "bot": [],
            "players_info": players_info
        }
        norm = np.linalg.norm(gold)
        gold = gold/norm
        gold[0, 23] = self.state.energy / 100
        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
        DQNState = mapID + DQNState + gold.flatten().tolist()
        # ## BFS 
        mapOrigin = np.transpose(mapOrigin)
        BFS_state = mapOrigin.flatten().tolist()
        BFS_state.append(self.state.x)
        BFS_state.append(self.state.y)
        BFS_state.append(self.state.energy)
        # DQNState = np.array(DQNState)
        ## -> origin state
        # return DQNState, envs_dict
        # ### -> BFS state
        return np.array(BFS_state), envs_dict

    def check_terminate(self):
        return self.state.status != State.STATUS_PLAYING
