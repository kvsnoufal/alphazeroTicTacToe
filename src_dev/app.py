from flask import Flask,jsonify,session,request
from flask_cors import CORS, cross_origin
import numpy as np
import pickle
import os
import sys
from model import NeuralNetwork
from dataset import TrainingDataset
from config import Config as cfg
from game import TicTacToe,Connect2
from value_policy_function import ValuePolicyNetwork
from mcts import MonteCarloTreeSearch
from copy import copy
from config import Config as cfg



app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key="session"

def init():
    game = TicTacToe()
    modelpath = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(0))
    vpn = ValuePolicyNetwork(modelpath)
    policy_value_network = vpn.get_vp
    mtcs = MonteCarloTreeSearch(game,policy_value_network)
    return game,mtcs


game,mtcs = init()
def build_tree(node):
    tree = {}
    if type(node.state) ==np.ndarray:
        tree["name"]= str(list([int(t) for t in node.state]))
        tree["attributes"]={}
        tree["attributes"]["nodeBoardState"] = list([int(t) for t in node.state])
    tree["attributes"]["pl"]= int(node.player)
    tree["attributes"]["visits"]= int(node.total_visits_N)
    tree["attributes"]["Avg Value"]= node.mean_action_value_of_next_state_Q 
    tree["children"]=[]
    for k,v in node.children.items():
        if type(v.state) ==np.ndarray:
        
            tree["children"].append(build_tree(v))
    return tree
@app.route("/", methods=["GET"])
@cross_origin()
def hello_word():
    return "Hello World"
@app.route("/play", methods=["POST"])
@cross_origin()
def play():
    bot_player = int(request.form["botPlayer"])
    records_storage = eval(request.form["records_storage"])
    records_storage = {int(k):v for k,v in records_storage.items()}
    # print(records_storage)
    player=1
    root_node = mtcs.init_root_node()
    
    node = root_node
    state = copy(node.state)
    counter = 0
    for counter,record in records_storage.items():
        state = np.array(copy(record["state"]))
        player = int(record["player"])
        # print("Player: {} state:\n{}".format(player,state.reshape(3,3)))
        if bot_player == player:
            node = mtcs.run_simulation(root_node=node,num_simulations=1600,player=player)
            action,node,action_probs = mtcs.select_move(node=node,mode="exploit",temperature=1)
            action = np.argmax(action)
            # print("Checking.....Bot: ",action,"Record:",record["action"])
        else:
            action = record["action"]
            if counter>0:
                node = mtcs.select_subtree_for_manual_action(node,action) 
            else:
                print("No node")
        state,won,player = game.play(state,player,action)
        if counter==0 and bot_player==-1:
            node = mtcs.init_root_node()
            node.state = state * -1
            node.player = -1
    node = mtcs.run_simulation(root_node=node,num_simulations=1600,player=player)
    tree = node.build_tree(copy(node))
    # print("before",node.state)
    # print(node.print_tree())
    # print(tree)
    action,node,action_probs = mtcs.select_move(node=node,mode="exploit",temperature=1)
    # print("after",node.state)
    action = np.argmax(action)
    # print("next_action",action)
    records_storage[counter+1] = {"state":list([int(t) for t in state]),"player":int(player),"action":int(action)}
    records_storage = {str(k):v for k,v in records_storage.items()}
    # print(records_storage)
    json_response = jsonify({"next_action":int(action),\
                    "records":records_storage,\
                    "tree":tree})
    return json_response

if __name__=="__main__":
    print("starting************")
    # app.run(debug=True,host="0.0.0.0")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)