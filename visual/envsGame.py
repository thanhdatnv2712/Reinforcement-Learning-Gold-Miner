import random
import cv2
import numpy as np
from prettytable import PrettyTable
from PIL import ImageFont, ImageDraw, Image

class envsGame(object):
    '''
    Visual Gold Miner Game
    '''
    def __init__(self):
        self.font = ImageFont.truetype("Roboto/Roboto-Black.ttf", 15)
        self.bot = cv2.imread("obj/blue.png")
        self.agent = cv2.imread("obj/agent.png")
        self.land = cv2.imread("obj/land.png")
        self.swamp = cv2.imread("obj/swamp.png")
        self.trap = cv2.imread("obj/trap.png")
        self.tree = cv2.imread("obj/tree.png")
        self.gold = cv2.imread("obj/gold.png")
        self.MAX_X = 21
        self.MAX_Y = 9
        self.SIZE = 60        
        self.envs = np.zeros([self.MAX_Y*self.SIZE, self.MAX_X*self.SIZE + self.SIZE * 4, 3], dtype=np.uint8)
        self.tblScore = np.zeros([self.MAX_Y*self.SIZE, self.SIZE*4, 3], dtype=np.uint8)
        self.tbl = PrettyTable(['ID', 'Score','Energy','status'])

    def draw_text(self, mat, text):
        cv2_im_rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        draw.text((10, 10),text, font= self.font)
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        return cv2_im_processed

    def maketblStatus(self, players_info, step):
        for info in players_info:
            self.tbl.add_row(info)
        content = "Step: {}\n{}\n".format(step, self.tbl.get_string())
        self.tblScore = self.draw_text(self.tblScore, content)
        self.envs[:, self.MAX_X*self.SIZE : , :] = self.tblScore

    def render(self, Map, agent, bots, players_info, step):
        '''
            Args:
                Map: [[], [], [],..,[], []] -> np.array
                agent: []
                bots: [[], [], []] 
                players_info: [[], [], [], []]
                step: int
            Return
                show Gold miner game
        '''
        self.maketblStatus(players_info, step)
        for i in range(self.MAX_Y):
            for j in range(self.MAX_X):
                if [j, i] == agent:
                    self.envs[i*self.SIZE : (i+1)*self.SIZE, j*self.SIZE : (j+1)*self.SIZE, :] = self.agent
                elif [j, i] in bots:
                    self.envs[i*self.SIZE : (i+1)*self.SIZE, j*self.SIZE : (j+1)*self.SIZE, :] = self.bot
                elif Map[i][j] > 0:
                    self.envs[i*self.SIZE : (i+1)*self.SIZE, j*self.SIZE : (j+1)*self.SIZE, :] = self.gold
                elif Map[i][j] == -1:
                    self.envs[i*self.SIZE : (i+1)*self.SIZE, j*self.SIZE : (j+1)*self.SIZE, :] = self.tree
                elif Map[i][j] == -2:
                    self.envs[i*self.SIZE : (i+1)*self.SIZE, j*self.SIZE : (j+1)*self.SIZE, :] = self.trap      
                elif Map[i][j] == -3:
                    self.envs[i*self.SIZE : (i+1)*self.SIZE, j*self.SIZE : (j+1)*self.SIZE, :] = self.swamp
                else:
                    self.envs[i*self.SIZE : (i+1)*self.SIZE, j*self.SIZE : (j+1)*self.SIZE, :] = self.land

        # self.envs = np.concatenate((self.envs, self.tblScore), 1)
        cv2.imshow("visual game", self.envs)
        cv2.waitKey(200)

# players_info = [['player', 500, 75, "ON"], ['bot1', 50, 25, "ON"], ['bot2', 100, 35, "ON"], ['bot3', 70, 45, "DIE"]]
# Map = [[0,0,-2,100,0,0,-1,-1,-3,0,0,0,-1,-1,0,0,-3,0,-1,-1,0],[-1,-1,-2,0,0,0,-3,-1,0,-2,0,0,0,-1,0,-1,0,-2,-1,0,0],[0,0,-1,0,0,0,0,-1,-1,-1,0,0,100,0,0,0,0,50,-2,0,0],[0,0,0,0,-2,0,0,0,0,0,0,0,-1,50,-2,0,0,-1,-1,0,0],[-2,0,200,-2,-2,300,0,0,-2,-2,0,0,-3,0,-1,0,0,-3,-1,0,0],[0,-1,0,0,0,0,0,-3,0,0,-1,-1,0,0,0,0,0,0,-2,0,0],[0,-1,-1,0,0,-1,-1,0,0,700,-1,0,0,0,-2,-1,-1,0,0,0,100],[0,0,0,500,0,0,-1,0,-2,-2,-1,-1,0,0,-2,0,-3,0,0,-1,0],[-1,-1,0,-2,0,-1,-2,0,400,-2,-1,-1,500,0,-2,0,-3,100,0,0,0]]
# Map = np.array(Map)
# bots_pos = [[5, 3], [2, 5], [17, 7]]
# agent_pos = [11, 5]

# game = envsGame()
# game.render(Map, agent_pos, bots_pos, players_info, 77)