# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from maze_env import Maze

#action_space = ['u', 'd', 'r', 'l']

env = Maze()
actions = list(range(env.n_actions)) # a list
Q_table = pd.DataFrame(columns=actions, dtype=np.float64)
alpha=0.5 # learning_rate
gamma=0.9 # discount
eps=0.5 # greedy
decay_rate=0.01
epoch = 1

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
    state = str(state)
        
    while True:
        env.render()
        action, Q_table, eps = choose_action(state, actions, eps, decay_rate, epoch, Q_table)
        state_next, reward, done = env.step(action)
        state_next = str(state_next)
        Q_table = execute_action(state, action, reward, state_next, gamma, alpha, Q_table)
        state = state_next
        
        if done:
            break
