from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
import torch
from MinerEnv import MinerEnv
import numpy as np
from DQNModel import Agent 
from envsGame import envsGame
from BFS_explore import BFS_energy

ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5

HOST = "localhost"
PORT = 1111

if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

# load json and create model
agent = Agent(218, 6, -1)
# bfs_agent = BFS_energy()
envsGame = envsGame()

print("Loaded model from disk")
status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
                  3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}
try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    minerEnv.reset()
    s, vs = minerEnv.get_state()  ##Getting an initial state
    step = 0
    while not minerEnv.check_terminate():
        try:
            envsGame.render(vs["map"], vs["agent"], vs["bot"], vs["players_info"], step)
            action = agent.action(s) # Getting an action from the trained model
            # action = bfs_agent.act(s)
            print("next action = ", action)
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            s_next, vs = minerEnv.get_state()  # Getting a new state
            s = s_next
            step += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Finished.")
            break
    print(status_map[minerEnv.state.status])
except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")
