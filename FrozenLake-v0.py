# https://github.com/Lukaschen1986/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Q%20Learning%20with%20FrozenLake.ipynb

# -*- coding: utf-8 -*-
import numpy as np
import gym
import random
import matplotlib.pyplot as plt

# Step 1: Create the environment
env = gym.make("FrozenLake-v0")

# Step 2: Create the Q-table and initialize it
action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros((state_size, action_size))

# Step 3: Create the hyperparameters
total_epochs = 20000          # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 100               # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
eps = 1.0                     # Exploration rate
max_eps = 1.0                 # Exploration probability at start
min_eps = 0.01                # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# Step 4: The Q learning algorithm
# List of rewards
global_rewards = []
#global_steps = []

# 2 For life or until learning is stopped
for epoch in range(1, total_epochs+1):
    # epoch = 1
    # Reset the environment
    state = env.reset()
    #step = 0
    done = False
    local_rewards = 0
    
    for stp in range(1,max_steps+1):
        # stp = 0
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        trade_off = random.uniform(0, 1)
        
        ## If this number greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if trade_off > eps:
            action = np.argmax(q_table[state,:])
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
        
#        if action == 0:
#            print("action is left")
#        elif action == 1:
#            print("action is down")
#        elif action == 2:
#            print("action is right")
#        else:
#            print("action is up")
        # Take the action(a) and observe the outcome state(s') and reward(r)
        next_state, immediate_reward, done, info = env.step(action)
        
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # q_table[new_state,:] : all the actions we can take from new state
        delta_q = immediate_reward + gamma * np.max(q_table[next_state,:]) - q_table[state,action]
        q_table[state,action] += learning_rate*(delta_q)
        local_rewards += immediate_reward
        
        # Our new state is state
        state = next_state
        
        # If done (if we're dead) : finish episode
        if done == True: 
            break
    #print("steps: " + str(stp))
    #global_steps.append(stp)
    
    # Reduce epsilon (because we need less and less exploration)
    eps = min_eps + (max_eps-min_eps)*np.exp(-decay_rate*epoch) 
    #eps = eps*(1-decay_rate)**epoch
    global_rewards.append(local_rewards)

print ("Score over time: " +  str(sum(global_rewards)/total_epochs))
print(q_table)

# Step 5: Use our Q-table to play FrozenLake
env.reset()

for epoch in range(1,6):
    state = env.reset()
    done = False
    print("****************************************************")
    print("EPOCH ", epoch)

    for stp in range(1,max_steps+1):
        env.render()
        # Take the action(index) that have the maximum expected future reward given that state
        action = np.argmax(q_table[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            break
    state = new_state
env.close()
