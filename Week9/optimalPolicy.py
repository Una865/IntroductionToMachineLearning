import numpy as np

grid = np.zeros((5,5))
path = [[]for i in range(5)]

episodes = 20
ghama = 0.9

for ep in range(episodes):
    for i in range(5):
        for j in range(5):
            up = grid[i-1, j] if i > 0 else 0
            down = grid[i+1, j] if  i < 4 else 0
            right = grid[i, j+1] if j < 4 else 0
            left = grid[i, j-1] if j > 0 else 0

            dirs = [up, down, right, left]

            prob_dirs = [1,1,1,1]
            v_state = 0
            max_state = -100
            if i == 0 and j == 1:
                v_state = 10 + ghama*grid[4,1]
                max_state = v_state
            elif i == 0 and j == 3:
                v_state = 5 + ghama * grid[2, 3]
                max_state = v_state
            else:
                 for d in range(len(dirs)):
                       if dirs[d] == 0:
                            reward = -1
                            v_state_next = grid[i,j]
                       else:
                            reward = 0
                            v_state_next = dirs[d]

                       v_state = prob_dirs[d]*(reward + ghama*v_state_next)
                       if max_state<v_state:
                           max_state = v_state
            if ep!=0:
               grid[i,j] = max(max_state,grid[i,j])
            else:
                grid[i,j] = max_state



grid = np.round(grid,1)
print()
print("Optimal function of state values:")
print()
print(grid)
print()
print("Optimal policy:")
print()
for i in range(5):
    for j in range(5):
        up = grid[i - 1, j] if i > 0 else -1000
        down = grid[i + 1, j] if i < 4 else -1000
        right = grid[i, j + 1] if j < 4 else -1000
        left = grid[i, j - 1] if j > 0 else -1000

        dirs = [up, down, right, left]
        dict = {0:'up',1:'down',2:'right',3:'left'}
        field = []
        maxv = -1000
        for d in range(4):
            if dirs[d]>maxv:
                maxv= dirs[d]
        for d in range(4):
            if dirs[d] == maxv:
                field.append(dict[d])
        path[i].append(field)
for i in range(5):
    print(path[i], end = ',')
    print()
