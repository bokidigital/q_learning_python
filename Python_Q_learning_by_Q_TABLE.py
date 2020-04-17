import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

HOLE = 1
GOAL = 2
ROAD = 0

map = [[ROAD,HOLE,ROAD,ROAD,ROAD],
		[ROAD,HOLE,ROAD,ROAD,ROAD],
		[ROAD,HOLE,ROAD,ROAD,ROAD],
		[ROAD,HOLE,ROAD,ROAD,ROAD],
		[ROAD,ROAD,ROAD,ROAD,GOAL]]

# define width and length of MAP, it will be used in this way:
# map[y_of_map][x_of_map]
y_of_map = 5
x_of_map = 5

# define total actions
gTotal_Actions = 4 #up, down, left, right

# discount factor
Gamma = 0.5

# Learning rate
alpha = 0.001

# Training loop
Trainning = 100000

# Declare and Init Q table
gQ_table = [[0] * gTotal_Actions for _ in range(x_of_map * y_of_map)]

init_location = 0 # start from upper left
gState = init_location
gAction = 0
gEpisode = Trainning + 150 # after trainning, test X times.

# How many step used.  reference only.
used_step = 0

# test purpose, if want to test specific steps
future_actions = [3,3,3,1,1,3,3]

def show_qTable(gQ_table):
    for state in range(0,x_of_map * y_of_map,1):
        for act in range(0,gTotal_Actions,1):
            print("gState[{:d}][{:d}]={:f}".format(state,act,gQ_table[state][act]))

def evn_render(gState):
    global map
    y_of_where_am_I = gState // x_of_map
    x_of_where_am_I = gState % y_of_map

    for y in range(0,y_of_map,1):
        for x in range(0,x_of_map,1):
            if y_of_where_am_I == y and x_of_where_am_I == x:
                print("T",end="")
            elif map[y][x] == ROAD:
                print("_",end="")
            elif map[y][x] == HOLE:
                print("O",end="")
            elif map[y][x] == GOAL:
                print("#",end="")
            else:
                print("PANIC, unknow element of MAP")
                exit()
        print("") # chnage to new line
	
def map_render(gState, episode_n, step_n, action):

    action_dict = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
    action_name = action_dict[action]

    map_reshaped = np.array(map).reshape(1,-1)[0]
    map_reshaped[gState] = 3
    map_reshaped = map_reshaped.reshape(5,5)

    fig = plt.figure()
    sns.heatmap(map_reshaped, linewidths=5, center=1)
    plt.title(f'Episode: {episode_n}, N steps: {step_n}, action: {action_name}')
    plt.savefig(f'debug/episode_{episode_n}_{step_n}_{action_name}.png')
    plt.close()
	
def choose_action(gState):
    if random.randint(0,10) > 5: # do random selection
        action = random.randint(0,gTotal_Actions - 1)
    else:
        action = get_action_from_MAX_Q(gState)
    return action

def get_action_from_MAX_Q(gState):
    global gQ_table
    maxQ = gQ_table[gState][0] # init a maxQ as first action, it will be updted later.
    ret_act = 0

    for act in range(1,gTotal_Actions,1):
        if maxQ < gQ_table[gState][act]:
            maxQ = gQ_table[gState][act]
            ret_act = act 

    return ret_act

def get_MAX_Q_from_state(gState):
    global gQ_table
    maxQ = gQ_table[gState][0] # init a maxQ as first action, it will be updted later.
    for act in range(1,gTotal_Actions,1):
        if maxQ < gQ_table[gState][act]:
            maxQ = gQ_table[gState][act]
    return maxQ

def get_reward_of_state(gState):
    global map
    if map[gState // x_of_map][gState % y_of_map] == HOLE:
        reward = -100
    elif map[gState // x_of_map][gState % y_of_map] == ROAD:
        reward = 0
    elif map[gState // x_of_map][gState % y_of_map] == GOAL:
        reward = 100
    else:
        print("PANIC")
        exit()
    return reward

def env_step(action,gState):
    done = False

    if (action == 0) and (gState - x_of_map > 0): # UP
        gState = gState - x_of_map
    elif (action == 1) and (gState + y_of_map < y_of_map * x_of_map): # DOWN
        gState = gState + x_of_map
    elif (action == 2) and (gState - 1 > 0): # LEFT
        gState = gState - 1
    elif (action == 3) and (gState + 1 < y_of_map * x_of_map): # RIGHT
        gState = gState + 1

    reward = get_reward_of_state(gState)
    if gState == y_of_map * x_of_map - 1 or map[gState // x_of_map][gState % y_of_map] == HOLE:
        done = True

    return gState, reward, done
    

# choose an action, from random or gQ_table
for i in range(0, gEpisode, 1):
    gAction = choose_action(gState)
    if i > Trainning: # After training, all action taken from Q table
        #show_qTable(gQ_table)
        gAction = get_action_from_MAX_Q(gState)
        print("gAction=",gAction)

    #gAction = future_actions[i]

# update state
    new_state, reward, done = env_step(gAction,gState)
    used_step = used_step + 1
    if i > Trainning:
        print("Episode={:d}, New gState={:d}, reward={:d}, done={:b} used_step={:d}".format(i, new_state,reward,done,used_step))
        evn_render(new_state)
	map_render(new_state, i, used_step, gAction)

    # update reward, q-value
    gQ_table[gState][gAction] = (1 - alpha) * gQ_table[gState][gAction] + alpha * (reward + Gamma * get_MAX_Q_from_state(new_state))
    #show_qTable(gQ_table)

# check game over yet?
    if (done): # The most bottom right is GOAL.
        if i > Trainning:
            if(reward == 100):
                print("Game Over, reward={:d}, used_step={:d}".format(reward, used_step))
            #show_qTable(gQ_table)
        gState = 0
        new_state = 0
        used_step = 0

    gState = new_state
