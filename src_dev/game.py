import numpy as np
import pandas as pd
import os
import math
from copy import copy
from tqdm import tqdm
class TicTacToe:
    def __init__(self):
        
        self.state = np.zeros(9)
        
    def get_valid_moves(self,state):
        valid_actions = np.zeros(9)
        valid_actions[np.where(state==0)]=1
        return valid_actions
    def check_if_action_is_valid(self,state,action):
        valid_actions = self.get_valid_moves(state)
        action_index = np.where(action==1)[0]
        if len(action_index)!=1:
            return False
        action_index= action_index[0]
        if valid_actions[action_index]!=1:
            return False
        return True
    def get_next_state_from_next_player_prespective(self,state,action,player):
        next_state = state.copy()
        next_state[np.argmax(action)]=1
        return next_state *-1
    def win_or_draw(self,state):
        state = state.reshape(3,3)
        if (np.any(state.sum(0)==3)) or (np.any(state.sum(1)==3))\
            or (state[np.diag_indices(3)].sum()==3) or \
            (np.fliplr(state)[np.diag_indices(3)].sum()==3):
            return 1
        if (np.any(state.sum(0)==-3)) or (np.any(state.sum(1)==-3))\
            or (state[np.diag_indices(3)].sum()==-3) or \
            (np.fliplr(state)[np.diag_indices(3)].sum()==-3):
            return -1
        if len(np.where(state==0)[0])==0:
            return 0
        return None
    def get_reward_for_next_player(self,state,player):
        winner = self.win_or_draw(state)
        if winner:
            if winner in [-1,1]:
#                 print(f"player {-1*player} won")
                return -1*player
#             print("Draw")
            return 0
        return winner
    def play(self,board_state,player,action_index):
        board_state[action_index] = player
        return board_state,self.win_or_draw(board_state),-1*player
        
win_indices = [[0,1],\
                [1,2],\
                [2,3]]
class Connect2:
    def __init__(self):
        pass
    def get_valid_moves(self,state):
        valid_actions = np.zeros(4)
        valid_actions[np.where(state==0)]=1
        return valid_actions
    def check_if_action_is_valid(self,state,action):
        valid_actions = self.get_valid_moves(state)
        action_index = np.where(action==1)[0]
        if len(action_index)!=1:
            return False
        action_index= action_index[0]
        if valid_actions[action_index]!=1:
            return False
        return True
    def get_next_state_from_next_player_prespective(self,state,action,player):
        next_state = state.copy()
        next_state[np.argmax(action)]=1
        return next_state *-1
    def win_or_draw(self,state):
        if np.any(np.array([np.sum(state[w]) for w in win_indices])==2):
            return 1
        if np.any(np.array([np.sum(state[w]) for w in win_indices])==-2):
            return -1
        if len(np.where(state==0)[0])==0:
            return 0
        return None
    def get_reward_for_next_player(self,state,player):
        winner = self.win_or_draw(state)
        if winner:
            if winner in [-1,1]:
#                 print(f"player {-1*player} won")
                return -1*player
#             print("Draw")
            return 0
        return winner
    def play(self,board_state,player,action_index):
        board_state[action_index] = player
        return board_state,self.win_or_draw(board_state),-1*player
                