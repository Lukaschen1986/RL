# -*- coding: utf-8 -*-
"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, alpha=0.01, gamma=0.9, eps=0.5, decay_rate=0.01):
        self.actions = actions  # a list
        self.alpha = alpha # learning_rate
        self.gamma = gamma # discount
        self.eps = eps # greedy
        self.decay_rate = decay_rate
        self.epoch = 1
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
        
    def choose_action(self, state):
        self.check_state_exist(state)
        # action selection
        trade_off = np.random.uniform() # np.random.uniform(low=0, high=1, size=1)[0]
        if trade_off < self.epsilon:
            # exploration: choose random action
            action = np.random.choice(self.actions)
        else:
            # exploitation: choose best action
            state_action = self.q_table.loc[state, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index)) # permutation if some actions have same value
            action = state_action.idxmax()
        # decay_rate on eps
        self.eps = self.epsilon*(1-self.decay_rate)**self.epoch
        self.epoch += 1
        return action
    
    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
            )
