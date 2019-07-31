import random
import numpy as np
import copy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 50

HOLE = 1
GOAL = 2
ROAD = 0
PLAYER = 3

MYMAP = [[ROAD,HOLE,ROAD,ROAD,ROAD],
		[ROAD,HOLE,ROAD,ROAD,ROAD],
		[ROAD,HOLE,ROAD,ROAD,ROAD],
		[ROAD,HOLE,ROAD,ROAD,ROAD],
		[ROAD,ROAD,ROAD,ROAD,GOAL]]

def merge_player_and_MYMAP(Player):
    global MYMAP
    pMYMAP = copy.deepcopy(MYMAP)
    y_of_Player = Player // Y_OF_MYMAP
    x_of_Player = Player % X_OF_MYMAP
    pMYMAP[y_of_Player][x_of_Player] = PLAYER
    return pMYMAP

# define width and length of MYMAP, it will be used in this way:
# MYMAP[Y_OF_MYMAP][X_OF_MYMAP]
Y_OF_MYMAP = 5
X_OF_MYMAP = 5

# define total actions
gTotal_Actions = 4 #up, down, left, right

def evn_render(pMYMAP):

    for y in range(0,Y_OF_MYMAP,1):
        for x in range(0,X_OF_MYMAP,1):
            if pMYMAP[y][x] == PLAYER:
                print("T",end="")
            elif pMYMAP[y][x] == ROAD:
                print("_",end="")
            elif pMYMAP[y][x] == HOLE:
                print("O",end="")
            elif pMYMAP[y][x] == GOAL:
                print("#",end="")
            else:
                print("PANIC, unknow element of MYMAP")
                exit()
        print("") # chnage to new line


def get_reward_of_state(gState):
    global MYMAP
    if MYMAP[gState // X_OF_MYMAP][gState % Y_OF_MYMAP] == HOLE:
        reward = -100
    elif MYMAP[gState // X_OF_MYMAP][gState % Y_OF_MYMAP] == ROAD:
        reward = 0
    elif MYMAP[gState // X_OF_MYMAP][gState % Y_OF_MYMAP] == GOAL:
        reward = 100
    else:
        print("PANIC")
        exit()
    return reward

def env_step(action,state):
    done = False

    if (action == 0) and (state - X_OF_MYMAP > 0): # UP
        state = state - X_OF_MYMAP
    elif (action == 1) and (state + Y_OF_MYMAP < Y_OF_MYMAP * X_OF_MYMAP): # DOWN
        state = state + X_OF_MYMAP
    elif (action == 2) and (state - 1 > 0): # LEFT
        state = state - 1
    elif (action == 3) and (state + 1 < Y_OF_MYMAP * X_OF_MYMAP): # RIGHT
        state = state + 1

    reward = get_reward_of_state(state)
    if state == Y_OF_MYMAP * X_OF_MYMAP - 1 or MYMAP[state // X_OF_MYMAP][state % Y_OF_MYMAP] == HOLE:
        done = True

    return state, reward, done, 0

##############################

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size =  state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    state_size = X_OF_MYMAP * Y_OF_MYMAP
    action_size = gTotal_Actions #env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES+100):
        #state = env.reset()
        player = 0  # where is playler,
        MYMAP_merged_state = merge_player_and_MYMAP(player)  # merge Player location into MYMAP array.
        NN_state = np.reshape(MYMAP_merged_state, [1, state_size]) # reshape MYMAP_merged_state to NN input form
        #evn_render(MYMAP_merged_state) # env.render()

        for time in range(5000):
            action = agent.act(NN_state)
            print ("action = ",action)
            next_state, reward, done, _ = env_step(action,player) #env.step(action)
            player = next_state

            MYMAP_merged_state = merge_player_and_MYMAP(next_state) # merge Player location into MYMAP array.

            #if e>EPISODES:
            #    evn_render(MYMAP_merged_state) # env.render()

            NN_next_state = np.reshape(MYMAP_merged_state, [1, state_size]) # reshape MYMAP_state to NN input form
            agent.remember(NN_state, action, reward, NN_next_state, done)
            NN_state = NN_next_state # update current state

            if done:
                print("reward: {} episode: {}/{}, used_step: {}, epsilon: {:.2}"
                      .format(reward, e, EPISODES, time, agent.epsilon))
                evn_render(MYMAP_merged_state) # env.render()
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
