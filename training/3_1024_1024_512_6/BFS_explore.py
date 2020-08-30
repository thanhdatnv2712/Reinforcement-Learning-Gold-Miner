import numpy as np
from queue import Queue 

class BFS_energy:
    def __init__(self,cnt = 0
                ,MAP_MAX_X = 21
                ,MAP_MAX_Y = 9
                ,next_time = [-5,-20,-40,-100,-120]):
        self.cnt = cnt # dem so lan buoc vo dam lay
        self.MAP_MAX_X = MAP_MAX_X
        self.MAP_MAX_Y = MAP_MAX_Y
        self.next_time = next_time
    
    #/*
    #   next_time : so nang luong tinh theo lan buoc vao dam lay
    #   all_lay : tat cac coordinate cua dam lay 
    #   num_step : luu lai so buoc di den tung vi tri (matrix 21*9)
    #   all_gold : tat ca coordinate cua mo vang
    #   value : nang luong tieu hao de den duoc tung vi tri (matrix 21*9)
    # */

    def BFS(self,matrix,value,num_step,visited,trace,all_gold,all_lay ,posx,posy,energy):
        q=Queue()
        q.put((posx,posy))
        visited[posx][posy] = 1
        while not q.empty():
            row, col = q.get()
            
            if (row,col) in all_lay:
                self.cnt+=1
            if self.cnt != 0:
                for i in range(len(all_lay)):
                    r = all_lay[i][0]
                    l = all_lay[i][1]
                    if visited[r][l] == 0:
                        matrix[r][l] = self.next_time[min(self.cnt,4)]

            if col+1 < self.MAP_MAX_Y and row >=0 and visited[row][col+1] == 0 and energy + matrix[row][col+1] > 0:
                q.put((row, col+1))
                trace[row][col+1] = trace[row][col] + "3"
                visited[row][col+1] = 1
                num_step[row][col+1] = num_step[row][col] +1
                if matrix[row][col+1] >0:
                    value[row][col+1] = value[row][col] + 4
                else:
                    value[row][col+1] = value[row][col] - matrix[row][col+1]



            if row+1 < self.MAP_MAX_X and col >=0 and visited[row+1][col] == 0 and energy + matrix[row+1][col] > 0:
                q.put((row+1, col))
                trace[row+1][col] = trace[row][col] + "1"
                visited[row+1][col] = 1
                num_step[row+1][col] = num_step[row][col] +1
                if matrix[row+1][col] >0:
                    value[row+1][col] = value[row][col] + 4
                else:
                    value[row+1][col] = value[row][col] - matrix[row+1][col]


            if 0 <= col-1 and row < self.MAP_MAX_X and visited[row][col-1] == 0 and energy + matrix[row][col-1] > 0:
                q.put((row, col-1))
                trace[row][col-1] = trace[row][col] + "2"
                visited[row][col-1] = 1
                num_step[row][col-1] = num_step[row][col] +1
                if matrix[row][col-1] >0:
                    value[row][col-1] = value[row][col] +4
            
                else:
                    value[row][col-1] = value[row][col] - matrix[row][col-1]


            if 0 <= row-1 and col < self.MAP_MAX_Y and visited[row-1][col] == 0 and energy + matrix[row-1][col] > 0:
                q.put((row-1, col))
                trace[row-1][col] = trace[row][col] + "0"
                visited[row-1][col] = 1
                num_step[row-1][col] = num_step[row][col] +1
                if matrix[row-1][col] >0:
                    value[row-1][col] = value[row][col] + 4
                else:
                    value[row-1][col] = value[row][col] - matrix[row-1][col]

        return num_step, value, trace, all_gold  


    def act(self,s):
            # Lay map, pos ,energy tu state
            maps = np.array(s[...,0:self.MAP_MAX_X*self.MAP_MAX_Y]).reshape(self.MAP_MAX_X,self.MAP_MAX_Y)
            pos = np.array(s[...,self.MAP_MAX_X*self.MAP_MAX_Y : self.MAP_MAX_X*self.MAP_MAX_Y+2])
            energy = s[...,self.MAP_MAX_X*self.MAP_MAX_Y+2:self.MAP_MAX_X*self.MAP_MAX_Y+3]
            energy = energy[0]
            inf = 10000000000000000
            x,y = pos
            #khoi tao acton
            action = '4'
            def reset():
                num_step = np.zeros(shape = (self.MAP_MAX_X,self.MAP_MAX_Y))
                value = np.zeros(shape = (self.MAP_MAX_X,self.MAP_MAX_Y),dtype=int)
                visited = np.zeros(shape = (self.MAP_MAX_X,self.MAP_MAX_Y))
                all_gold = []
                all_lay = []
                trace = [["" for x in range(self.MAP_MAX_Y)] for y in range(self.MAP_MAX_X)]
                return num_step , value , visited , all_gold , all_lay ,trace
            
            def get_dir(matrix , s , x,y):
                if s == '2' and y-1 >=0: # turn up
                    return np.array([x,y-1])
                if s == '3' and  y+1<self.MAP_MAX_Y: # turn down
                    return np.array([x,y+1])
                if s == '0' and x-1 >=0: # turn left
                    return np.array([x-1,y])
                if s == '1' and x+1 < self.MAP_MAX_X: # turn right
                    return np.array([x+1,y])
                return np.array([x,y])

            
            num_step, value, visited, all_gold, all_lay,tr =  reset()
            next_time = [-5,-20,-40,-10000,-1200000]

            # Set lai map 

            for i in range(self.MAP_MAX_X):
                for j in range(self.MAP_MAX_Y):
                    if(maps[i][j] == 0): # gap dat
                        maps[i][j] = -1
                    elif(maps[i][j] == -1) : #gap rung
                        maps[i][j] = -20
                    elif maps[i][j] == -2: #gap bay
                        maps[i][j] = -10
                    elif maps[i][j] == -3: # gap dam lay
                        maps[i][j] = next_time[min(self.cnt,4)]
                        all_lay.append((i,j))
                    else :
                        all_gold.append((i,j))
            
            nstep, value, trace , coord_gold= self.BFS(maps,num_step,value,visited,tr,all_gold,all_lay,x,y,energy)
            
            #mining:
            if maps[x][y] > 0 :
                if energy > 5: 
                    action = '5'
                else : # rest
                    action = '4'


            else:

                # /*
                # tim duong di den bai vang ton it energy nhat  
                # 
                # */      
                Min = inf
                ix , iy = x,y        
                for idx,idy in coord_gold:
                    if Min > value[idx][idy] and value[idx][idy]!=0:
                        Min = value[idx][idy]
                        ix = idx
                        iy = idy
                trace[ix][iy] += '4'

                act = trace[ix][iy][0]
                if energy <=10: # check khi nang luong con lai ko the di den duoc mo vang ==> rest
                    act = '4'
                    action = '4'
                    next_move_value = -inf
                    di = np.array([x,y])
                elif Min == inf :
                    Min_step_value = -inf
                    step = ''
                    move = [0,1,2,3]
                    for i in move:
                        next_pos = get_dir(maps,str(i),x,y)
                        if x == next_pos[0] and y == next_pos[1]:
                            continue
                        if Min_step_value <  maps[next_pos[0]][next_pos[1]]:
                            Min_step_value = maps[next_pos[0]][next_pos[1]]
                            step = str(i)
                    next_move_value = Min_step_value
                else:
                    di = get_dir(maps,act,x,y)
                    next_move_value = maps[di[0]][di[1]]
                
                
                if energy + next_move_value > 0 : #moving
                        action = act
                else : # resting
                    action = '4'

            return action
