import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket #in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State


TreeID = 1
TrapID = 2
SwampID = 3
class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        self.last_x = self.state.x
        self.last_y = self.state.y
        self.last_energy = self.state.energy
        self.last_action = self.state.lastAction
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
            self.state.init_state(message) #init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action): #step process
        self.last_x = self.state.x
        self.last_y = self.state.y
        self.last_energy = self.state.energy

        self.socket.send(action) #send action to server
        try:
            message = self.socket.receive() #receive new state from server
            self.state.update_state(message) #update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        # Building the map
        #mapID = [0, 0, 0, 0, 0]
        #mapID[int(self.socket.mapId[-1])-1] = 1
        view = np.zeros([self.state.mapInfo.max_y + 1, self.state.mapInfo.max_x + 1])
        #print (view.shape)
        view[self.state.y - 1, self.state.x - 1] = 1
        gold = np.zeros([1, 24])
        idx = 0
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    #print (j, i)
                    gold[0, idx] = self.state.mapInfo.gold_amount(i, j)
                    idx += 1
        norm = np.linalg.norm(gold)
        gold = gold/norm
        gold[0, 23] = self.state.energy / 100
        DQNState = view.flatten().tolist() #Flattening the map matrix to a vector
        DQNState = DQNState + gold.flatten().tolist()
        DQNState = np.array(DQNState)
        return DQNState

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        #print (self.state.x, self.state.y, self.last_x, self.last_y, self.state.energy, self.last_energy)
        self.score_pre = self.state.score
        if score_action > 0:
            #If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action / 5
            
        #If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        # if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
        #     reward -= 0.1
        # if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
        #     reward -= 0.2
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            reward -= 0.4

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -1
            
        #Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -0.3

        if self.state.status == State.STATUS_PLAYING:
            reward += 0.03

        if self.state.energy >= 80 and self.state.lastAction == 4:
            reward += -1

        if self.state.energy >= 45 and self.state.lastAction == 4:
            reward += -0.5

        if self.state.energy < 20:
            reward += -0.5

        if self.state.mapInfo.gold_amount(self.state.x,self.state.y) >= 50:
            reward += 0.5

        if self.state.status == State.STATUS_STOP_END_STEP:
            reward += 0.5

        # if there is no gold, but the agent still crafts golds, it will be punished
        if self.state.mapInfo.get_obstacle(self.last_x, self.last_y) < 0 and int(self.state.lastAction) == 5:
            reward += -0.2

        if (self.state.mapInfo.gold_amount(self.last_x, self.last_y) > 0 and self.last_energy > 5) and (int(self.state.lastAction)!= 5):
            reward += -0.2

        if (self.state.mapInfo.gold_amount(self.last_x, self.last_y) > 0 and self.last_energy > 5) and (int(self.state.lastAction)== 5):
            reward += 0.5

        if (self.state.mapInfo.gold_amount(self.last_x, self.last_y) >= 50 and self.last_energy > 5) and (int(self.state.lastAction)== 5):
            reward += 1

        if (self.state.mapInfo.gold_amount(self.last_x, self.last_y) == 0) and (int(self.state.lastAction)!= 5):
            reward += -0.3
 
        return reward

    def check_terminate(self):
        #Checking the status of the game
        #it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING   
