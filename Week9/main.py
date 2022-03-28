import numpy as np

grid = np.zeros((5,5))

episodes = 20
ghama = 0.9

for ep in range(episodes):
    for i in range(5):
        for j in range(5):
            up_grid = grid[i-1, j] if i > 0 else 0
            down_grid = grid[i+1, j] if  i < 4 else 0
            right_grid = grid[i, j+1] if j < 4 else 0
            left_grid = grid[i, j-1] if j > 0 else 0

            dirs = [up_grid, down_grid, right_grid, left_grid]

            prob_dirs = [0.25,0.25,0.25,0.25]
            v_state = 0
            if i == 0 and j == 1:
                v_state = 10 + ghama*grid[4,1]

            elif i == 0 and j == 3:
                v_state = 5 + ghama * grid[2, 3]
            else:
                 for d in range(len(dirs)):
                       if dirs[d] == 0:
                            reward = -1
                            v_state_next = grid[i,j]
                       else:
                            reward = 0
                            v_state_next = dirs[d]

                       v_state += prob_dirs[d]*(reward + ghama*v_state_next)

            grid[i,j] = v_state

print(np.round(grid,1))




