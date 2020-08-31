import numpy as np
from queue import Queue 
inf = 10000000000000000
class BFS_energy:
    def __init__(self,cnt = 0
                ,MAP_MAX_X = 21
                ,MAP_MAX_Y = 9
                ,damlay = -5
                ,step = 100
                ,next_time = {0:-5,-5:-20,-20:-40,-40:-100,-100:-100}):
        self.cnt = cnt # dem so lan buoc vo dam lay
        self.MAP_MAX_X = MAP_MAX_X
        self.MAP_MAX_Y = MAP_MAX_Y
        self.next_time = next_time
        self.step = step # so luot choi con lai
        self.damlay = damlay
        self.stack = []
    #/*
    #   next_time : so nang luong tinh theo lan buoc vao dam lay
    #   all_lay : tat cac coordinate cua dam lay 
    #   num_step : luu lai so buoc di den tung vi tri (matrix 21*9)
    #   all_gold : tat ca coordinate cua mo vang
    #   value : nang luong tieu hao de den duoc tung vi tri (matrix 21*9)
    # */

    # def normal_BFS(self,matrix,all_gold,all_lay):
    #     q = Queue()
    #     q.push

    def BFS(self,matrix,value,num_step,visited,trace,all_gold,all_lay ,posx,posy,energy,count):
        q=Queue()
        q.put((posx,posy))
        visited[posx][posy] = 1
        current = self.damlay
        count = 0
        while not q.empty():
            row, col = q.get()
            
            if (row,col) in all_lay:
                count+=1
            if count != 0:
                for i in range(len(all_lay)):
                    r = all_lay[i][0]
                    l = all_lay[i][1]
                    if visited[r][l] == 0:
                        matrix[r][l] = self.next_time[current]
                current = self.next_time[current]
            if current >= 40:
                for p in all_lay:
                    visited[p[0]][p[1]] = 1
                    value[p[0]][p[1]] =inf
                    step[p[0]][p[1]] = -inf

            if col+1 < self.MAP_MAX_Y and row >=0:
                if visited[row][col+1] == 0 :
                    q.put((row, col+1))
                    trace[row][col+1] = trace[row][col] + "3"
                    visited[row][col+1] = 1
                    num_step[row][col+1] = num_step[row][col] +1
                    if matrix[row][col+1] >0:
                        value[row][col+1] = value[+row][col] + 4
                    else:
                        if (row,col+1) in all_lay:
                            value[row][col+1] = value[row][col] - 10*matrix[row][col+1]
                        else:
                            value[row][col+1] = value[row][col] - matrix[row][col+1]
                else:
                    tmp = value[row][col+1]
                    
                    # visited[row][col+1] =
                    
                    if matrix[row][col+1] >0:
                        tmp = value[row][col] + 4
                    else:
                        if (row,col+1) in all_lay:
                            tmp = value[row][col] - 10*matrix[row][col+1]
                        else:
                            tmp = value[row][col] - matrix[row][col+1]
                    if tmp < value[row][col+1] :
                        value[row][col+1] = tmp
                        num_step[row][col+1] = num_step[row][col] +1
                        trace[row][col+1] = trace[row][col] + "3"
                    



            if row+1 < self.MAP_MAX_X and col >=0:
                
                if visited[row+1][col] == 0 :
                    q.put((row+1, col))
                    trace[row+1][col] = trace[row][col] + "1"
                    visited[row+1][col] = 1
                    num_step[row+1][col] = num_step[row][col] +1
                    if matrix[row+1][col] >0:
                        value[row+1][col] = value[row][col] + 4
                    else:
                        if (row+1,col) in all_lay:
                            value[row+1][col] = value[row][col] - 10*matrix[row+1][col]
                        else :
                            value[row+1][col] = value[row][col] - matrix[row+1][col]

                    # print(value[row+1][col], matrix[row+1][col], row, col)
                else:
                    tmp = value[row+1][col]
                    # visited[row][col+1] =
                    
                    if matrix[row+1][col] >0:
                        tmp = value[row][col] + 4
                    else:
                        if (row+1,col) in all_lay:
                            tmp = value[row][col] - 10*matrix[row+1][col]
                        else:
                            tmp = value[row][col] - matrix[row+1][col]
                    if tmp < value[row+1][col] :
                        value[row+1][col] = tmp
                        num_step[row+1][col] = num_step[row][col] +1
                        trace[row+1][col] = trace[row][col] + "1"
                    


            if 0 <= col-1 and row < self.MAP_MAX_X :
                if visited[row][col-1] == 0 :
                    q.put((row, col-1))
                    trace[row][col-1] = trace[row][col] + "2"
                    visited[row][col-1] = 1
                    num_step[row][col-1] = num_step[row][col] + 1
                    if matrix[row][col-1] >0:
                        value[row][col-1] = value[row][col] +4
                
                    else:
                        if (row,col-1) in all_lay:
                            value[row][col-1] = value[row][col] - 10*matrix[row][col-1]
                        else:
                            value[row][col-1] = value[row][col] - matrix[row][col-1]
                else :
                    tmp = value[row][col-1]
                    
                    # visited[row][col+1] =
                    
                    if matrix[row][col-1] >0:
                        tmp = value[row][col] + 4
                    else:
                        if (row,col-1) in all_lay:
                            tmp = value[row][col] - 10*matrix[row][col-1]
                        else:
                            tmp = value[row][col] - matrix[row][col-1]
                    if tmp < value[row][col-1] :
                        value[row][col-1] = tmp
                        num_step[row][col-1] = num_step[row][col] +1
                        trace[row][col-1] = trace[row][col] + "2"
                    


            if 0 <= row-1 and col < self.MAP_MAX_Y :
                if visited[row-1][col] == 0 :
                    q.put((row-1, col))
                    trace[row-1][col] = trace[row][col] + "0"
                    visited[row-1][col] = 1
                    num_step[row-1][col] = num_step[row][col] +1
                    if matrix[row-1][col] >0:
                        value[row-1][col] = value[row][col] + 4
                    else:
                        if (row-1,col) in all_lay:
                            value[row-1][col] = value[row][col] - 10*matrix[row-1][col]
                        else :
                            value[row-1][col] = value[row][col] - matrix[row-1][col]
                
                else :
                    tmp = value[row-1][col]
                    
                    # visited[row][col+1] =
                    
                    if matrix[row-1][col] >0:
                        tmp = value[row][col] + 4
                    else:
                        if (row-1,col) in all_lay:
                            tmp = value[row][col] - 10*matrix[row-1][col]
                        else:
                            tmp = value[row][col] - matrix[row-1][col]
                    if tmp < value[row-1][col] :
                        value[row-1][col] = tmp
                        num_step[row-1][col] = num_step[row][col] +1
                        trace[row-1][col] = trace[row][col] + "0"

        return num_step, value, trace, all_gold  


    def act(self,s):
            # Lay map, pos ,energy tu state
            maps = np.array(s[...,0:self.MAP_MAX_X*self.MAP_MAX_Y]).reshape(self.MAP_MAX_X,self.MAP_MAX_Y)
            pos = np.array(s[...,self.MAP_MAX_X*self.MAP_MAX_Y : self.MAP_MAX_X*self.MAP_MAX_Y+2])
            energy = s[...,self.MAP_MAX_X*self.MAP_MAX_Y+2:self.MAP_MAX_X*self.MAP_MAX_Y+3]
            bot1 = s[...,self.MAP_MAX_X*self.MAP_MAX_Y+3:self.MAP_MAX_X*self.MAP_MAX_Y+5]
            bot2 = s[...,self.MAP_MAX_X*self.MAP_MAX_Y+5:self.MAP_MAX_X*self.MAP_MAX_Y+7]
            bot3 = s[...,self.MAP_MAX_X*self.MAP_MAX_Y+7:self.MAP_MAX_X*self.MAP_MAX_Y+9]
            bot = []
            bot.append(bot1)
            bot.append(bot2)
            bot.append(bot3)
            energy = energy[0]
            count = 0
            
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
            next_time = self.next_time
            # Set lai map 

            for i in range(self.MAP_MAX_X):
                for j in range(self.MAP_MAX_Y):
                    if(maps[i][j] == 0): # gap dat
                        maps[i][j] = -1
                    elif(maps[i][j] == -1) : #gap rung
                        maps[i][j] = -4
                    elif maps[i][j] == -2: #gap bay
                        maps[i][j] = -2
                    elif maps[i][j] == -3: # gap dam lay
                        maps[i][j] = self.damlay
                        if self.damlay + 50 <= 0:
                            visited[i][j] = 1
                        all_lay.append((i,j))
                    else :
                        # if visited[i][j] == 0:
                            all_gold.append((i,j))
            
            
            
            # # #mining:
            if maps[x][y] > 0 :
                if energy > 5: 
                    # print("gold : ",maps[x][y], (x,y))
                    action = '5'
                else : # rest
                    action = '4'


            else:

                # /*
                # tim duong di den bai vang ton it energy nhat  
                # 
                # */      
                nstep, value, trace , coord_gold= self.BFS(maps,num_step,value,visited,tr,all_gold,all_lay,x,y,energy,count)
                Min = inf
                # print(nstep)
                # print(trace)
                # print(value)

                ix , iy = x,y        
                point = {}
                # print('---------------------------------------------------')
                gold_nearby = 0
                
                    

                for idx,idy in coord_gold:
                    step_val = self.step - nstep[idx][idy]
                    
                    num_craft = 0
                    val = 0
                    if step_val <= 0:
                        val = -100000000
                    else :
                        num_craft = maps[idx][idy]/50
                           
                        if step_val > num_craft:
                            val = (num_craft)*50
                            # print('val: ', (idx,idy),step_val)
                        else:
                            val = (num_craft - self.step)*50
                    add_point = 0
                    if 1000 > maps[idx][idy] >= 500:
                        val+=50
                    if maps[idx][idy] >= 1000:
                        val+=100
                
                    point[(idx,idy)] =  -(value[idx][idy]) + val/(num_step[idx][idy]+4+num_craft)
                    print('point :',(idx,idy),maps[idx][idy], point[(idx,idy)],val,nstep[idx][idy],value[idx][idy] )
                d = [[0,1],[1,0],[0,-1],[-1,0],[1,1],[-1,1],[1,-1],[-1,-1]]
                # if len(self.stack)>1:


                for id in range(len(d)):
                    xx = x + d[id][0]
                    yy = y + d[id][1]
                    if xx < self.MAP_MAX_X and xx >= 0 and yy < self.MAP_MAX_Y and yy >= 0:
                        if maps[xx][yy] > 0:
                            lay = 0
                            for id1 in range(len(d)):
                                x1 = xx + d[id1][0]
                                y1 = yy + d[id1][0]
                                if xx < self.MAP_MAX_X and xx >= 0 and yy < self.MAP_MAX_Y and yy >= 0 and (x1,y1) in all_lay:
                                    lay+=1
                            if lay <=3 :

                                # self.stack.append((xx,yy))
                                if point[(xx,yy)] > 0 :
                                    point[(xx,yy)] *= 10
                                else :
                                    point[(xx,yy)] *=10

                Max = -inf
                for idx,idy in coord_gold:
                    if Max < point[(idx,idy)]:
                        Max = point[(idx,idy)]
                        ix = idx
                        iy = idy
                trace[ix][iy] += '4'
                # if
                act = trace[ix][iy][0]
                if energy <23: # check khi nang luong con lai ko the di den duoc mo vang ==> rest
                    act = '4'
                    action = '4'
                    next_move_value = -inf
                    di = np.array([x,y])
                elif Max == -inf :
                    Min_step_value = -inf
                    step = ''
                    far = -inf
                    p = [x,y]
                    for i in range(21):
                        for j in range(9):
                            if far < nstep[i][j] and trace[i][j]!='':
                                far = nstep[i][j]
                                # print(trace[i][j])
                                action = trace[i][j][0]
                                p = (i,j)
                            elif far == nstep[i][j]:
                                if value[i][j] < value[p[0]][p[1]]:
                                    p = (i,j)
                                    action = trace[i][j][0]
                    pos = get_dir(maps,action,x,y)
                    Min_step_value = maps[pos[0]][pos[1]]
                    next_move_value = Min_step_value
                else:
                    di = get_dir(maps,act,x,y)
                    next_move_value = maps[di[0]][di[1]]
                
                
                if energy + next_move_value > 0 : #moving
                        action = act
                else : # resting
                    action = '4'
                
            next_pos = get_dir(maps,action,x,y)
            if x!= next_pos[0] or y!= next_pos[1]:
                if (next_pos[0],next_pos[1]) in all_lay:
                    # print('into lay')
                    self.damlay = next_time[self.damlay]
            # print('lay : ' , self.cnt,energy,(x,y),'->',(next_pos[0],next_pos[1]))
            self.step -=1
            return action