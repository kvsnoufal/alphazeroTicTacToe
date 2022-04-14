import numpy as np
import os
from config import Config as cfg
from game import TicTacToe,Connect2
from mcts import MonteCarloTreeSearch
from dataset import TrainingDataset
from tqdm import tqdm
from value_policy_function import ValuePolicyNetwork
from copy import copy

os.makedirs(cfg.SAVE_PICKLES,exist_ok=True)
save_path = os.path.join(cfg.SAVE_PICKLES,cfg.DATASET_PATH)


game = TicTacToe()
# game = Connect2()
vpn = ValuePolicyNetwork()
policy_value_network = vpn.get_vp
mtcs = MonteCarloTreeSearch(game,policy_value_network)
root_node = mtcs.init_root_node()
num_games = cfg.SELFPLAY_GAMES


training_dataset = TrainingDataset()
for game_number in tqdm(range(num_games),total=num_games):
    player = 1
    node = root_node
    dataset = []
    while game.win_or_draw(node.state)==None:
#         print("{}".format(node.state.reshape(3,3)))
#         print("player: {}".format(player))
        parent_state = copy(node.state)
        node = mtcs.run_simulation(root_node=node,num_simulations=1600,player=player)
        action,node,action_probs = mtcs.select_move(node=node,mode="explore",temperature=1)
        dataset.append([parent_state,action_probs,player])
        player = -1*player

#     print("{}".format(node.state.reshape(3,3)))
#     print("player: {}".format(player))
    winner = game.get_reward_for_next_player(node.state,player)
#     print("winner : {}".format(winner))
    training_dataset.add_game_to_training_dataset(dataset,winner)
    if game_number%500 == 0:
        training_dataset.save(save_path) 
        print("saving....",game_number)

    
training_dataset.save(save_path)    