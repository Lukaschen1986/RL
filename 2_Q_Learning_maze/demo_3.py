# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#from maze_env_2 import Maze
from maze_env_3 import Maze

#action_space = ['u', 'd', 'r', 'l']

#env = Maze(unit=40, step=40, maze_h=4, maze_w=4)
env = Maze()
actions = list(range(env.n_actions)) # a list

Q_table_agent = pd.DataFrame(columns=actions, dtype=np.float64)
Q_table_guard = pd.DataFrame(columns=actions, dtype=np.float64)

alpha_agent=0.9 # learning_rate
gamma_agent=0.5 # discount
eps_agent=0.5 # greedy
decay_rate_agent=0.01

alpha_guard=0.9 # learning_rate
gamma_guard=0.5 # discount
eps_guard=0.5 # greedy
decay_rate_guard=0.01


def check_state_exist(state, Q_table, actions):
    if state not in Q_table.index:
        # append new state to q table
        Q_table = Q_table.append(pd.Series([0]*len(actions), index=Q_table.columns, name=state))
    else:
        Q_table = Q_table
    return Q_table

def choose_action(state, actions, eps, decay_rate, epoch, Q_table):
    Q_table = check_state_exist(state, Q_table, actions)
    # action selection
    trade_off = np.random.uniform() # np.random.uniform(low=0, high=1, size=1)[0]
    if trade_off < eps:
        # exploration: choose random action
        action = np.random.choice(actions)
    else:
        # exploitation: choose best action
        state_action = Q_table.loc[state, :] # this is a Series
        state_action = state_action.reindex(np.random.permutation(state_action.index)) # permutation if some actions have same value
        action = state_action.idxmax()
    # decay_rate on eps
    eps = eps*(1-decay_rate)**epoch
    return action, Q_table, eps

def execute_action(state, action, reward, state_next, gamma, alpha, Q_table):
    Q_table = check_state_exist(state_next, Q_table, actions)
    Q = Q_table.loc[state, action]
    if state_next != "terminal":
        Q_pred = reward + gamma * Q_table.loc[state_next, :].max()  # next state is not terminal
    else:
        Q_pred = reward  # next state is terminal
    Q = Q + alpha * (Q_pred - Q)  # update Q
    Q_table.loc[state, action] = Q # rewrite Q to q_table
    return Q_table


for epoch in range(1,101):
    state = env.reset()
    state_agent, state_guard = state[0], state[1]
    state_agent = str(state_agent)
    state_guard = str(state_guard)
    
    while True:
        env.render()
        
        action_agent, Q_table_agent, eps_agent = choose_action(state_agent, actions, eps_agent, decay_rate_agent, epoch, Q_table_agent)
        action_guard, Q_table_guard, eps_guard = choose_action(state_guard, actions, eps_guard, decay_rate_guard, epoch, Q_table_guard)
        
        s_agent_next, r_agent, d_agent, s_guard_next, r_guard, d_guard = env.step(action_agent, action_guard)
        

        s_agent_next = str(s_agent_next)
        s_guard_next = str(s_guard_next)
        
        Q_table_agent = execute_action(state_agent, action_agent, r_agent, s_agent_next, gamma_agent, alpha_agent, Q_table_agent)
        Q_table_guard = execute_action(state_guard, action_guard, r_guard, s_guard_next, gamma_guard, alpha_guard, Q_table_guard)
        
        state_agent = s_agent_next
        state_guard = s_guard_next
        
        if d_agent or d_guard:
            break
    
    print("epoch: " + str(epoch))

#print ("Score over time: " +  str(sum(global_rewards)/epoch))

# end of game
print("game over")
env.destroy()
