import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from model import NeuralNetwork
from dataset import TrainingDataset
from config import Config as cfg
from game import TicTacToe,Connect2
from glob import glob
import pandas as pd
from value_policy_function import ValuePolicyNetwork
from mcts import MonteCarloTreeSearch
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Trainer:
    def __init__(self,modelpath=None):
        os.makedirs(cfg.SAVE_MODEL_PATH,exist_ok=True)
        os.makedirs(cfg.LOGDIR,exist_ok=True)
        self.model = NeuralNetwork().to(device)
        self.modelpath=modelpath
        self.latest_file_number = -1
        if modelpath:
            self.model.load_state_dict(torch.load(modelpath))
        else:
            all_models = glob(cfg.SAVE_MODEL_PATH + "/*.pt")
            if len(all_models)>0:
                files = [int(os.path.basename(f).split("_")[0]) for f in all_models]
                self.latest_file_number = max(files)
                latest_file = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number))
                print("latest_model ...{}".format(latest_file))                
                self.model.load_state_dict(torch.load(latest_file))
            else:
                savepath = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number))
                torch.save(self.model.state_dict(), savepath)
                print("init.....Saving Model.....BL",savepath)
            
        self.train_data,self.eval_data = self.load_data()
        
        

        
    def load_data(self):
        ds = TrainingDataset()
        save_path = os.path.join(cfg.SAVE_PICKLES,cfg.DATASET_PATH)
        ds.load(save_path)
        return ds.retreive_test_train_data()
    def train(self):
        train_dataloader = DataLoader(self.train_data,\
                        batch_size=cfg.BATCH_SIZE,\
                        shuffle=True)
        eval_dataloader = DataLoader(self.eval_data,\
                                batch_size=cfg.BATCH_SIZE,\
                                shuffle=False)
        value_criterion = nn.MSELoss().to(device)
        policy_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(self.model.parameters())
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',\
                factor=0.5, patience=5, threshold=0.0001, threshold_mode='rel',\
                     cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        best_loss = 1000
        history = []
        for epoch in range(cfg.EPOCHS):
            self.model.train()
            train_loss = 0
            for i, (X,v,p) in enumerate(train_dataloader):
                X = X.to(device)
                v = v.to(device)
                p = p.to(device)

                yv,yp = self.model(X)
                vloss = value_criterion(yv,v)
                aloss = policy_criterion(yp,p)

                loss = vloss + aloss
                train_loss+=loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = train_loss/len(train_dataloader)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (X,v,p) in enumerate(eval_dataloader):
                    X = X.to(device)
                    v = v.to(device)
                    p = p.to(device)

                    yv,yp = self.model(X)
                    vloss = value_criterion(yv,v)
                    aloss = policy_criterion(yp,p)

                    loss = vloss + aloss
                    val_loss+=loss.item()
                val_loss = val_loss/len(eval_dataloader)
                lr_scheduler.step(val_loss)
                if val_loss<best_loss:
                        best_loss = val_loss
                        
                        savepath = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number+1))
                        torch.save(self.model.state_dict(), savepath)
                        print("Saving Model.....BL",savepath)
            print(f"Epoch {epoch}:: Train Loss: {train_loss}; Eval Loss: {val_loss};")
            history.append([epoch,train_loss,val_loss])
        
        history = pd.DataFrame(history,columns=["Epoch","Tr_Loss","Eval_Loss"])
        logpath = os.path.join(cfg.LOGDIR,"{}_history.csv".format(self.latest_file_number+1))
        history.to_csv(logpath,index=None)
        print(history)
    def evaluate(self):
        game = TicTacToe()
#         game = Connect2()
        model_path_old = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number))
        vpn_old = ValuePolicyNetwork(model_path_old)
        policy_value_network_old = vpn_old.get_vp
        mtcs_old = MonteCarloTreeSearch(game,policy_value_network_old)
        
        model_path = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number+1))
        vpn = ValuePolicyNetwork(model_path)
        policy_value_network = vpn.get_vp
        mtcs = MonteCarloTreeSearch(game,policy_value_network)
        
        root_node = mtcs.init_root_node()
        num_games = cfg.EVAL_GAMES
        results = []
        for game_number in tqdm(range(num_games),total=num_games):
            player=1
            node = root_node
            while game.win_or_draw(node.state)==None:
        #         print("{}".format(node.state.reshape(3,3)))
        #         print("player: {}".format(player))
                if player ==1:
                    node = mtcs.run_simulation(root_node=node,num_simulations=1600,player=player)
                    action,node,action_probs = mtcs.select_move(node=node,mode="exploit",temperature=1)
                if player ==-1:
                    node = mtcs_old.run_simulation(root_node=node,num_simulations=1600,player=player)
                    action,node,action_probs = mtcs_old.select_move(node=node,mode="exploit",temperature=1)
                player = -1*player

        #     print("{}".format(node.state.reshape(3,3)))
        #     print("player: {}".format(player))
            winner = game.get_reward_for_next_player(node.state,player)
        #     print("winner : {}".format(winner))
            if winner==1:
                results.append([game_number,winner,"model"])
            if winner==0:
                results.append([game_number,winner,"draw"])
            if winner==-1:
                results.append([game_number,winner,"old_model"])

        mtcs_old = MonteCarloTreeSearch(game,policy_value_network_old)
        mtcs = MonteCarloTreeSearch(game,policy_value_network)

        root_node = mtcs.init_root_node()

        for game_number in tqdm(range(num_games),total=num_games):
            player=1
            node = root_node
            while game.win_or_draw(node.state)==None:
        #         print("{}".format(node.state.reshape(3,3)))
        #         print("player: {}".format(player))
                if player ==1:
                    node = mtcs_old.run_simulation(root_node=node,num_simulations=1600,player=player)
                    action,node,action_probs = mtcs_old.select_move(node=node,mode="exploit",temperature=1)
                if player ==-1:
                    node = mtcs.run_simulation(root_node=node,num_simulations=1600,player=player)
                    action,node,action_probs = mtcs.select_move(node=node,mode="exploit",temperature=1)
                player = -1*player
        #     print("{}".format(node.state.reshape(3,3)))
        #     print("player: {}".format(player))
            winner = game.get_reward_for_next_player(node.state,player)
        #     print("winner : {}".format(winner))
            if winner==1:
                results.append([game_number,winner,"old_model"])
            if winner==0:
                results.append([game_number,winner,"draw"])
            if winner==-1:
                results.append([game_number,winner,"model"])
        dfresults = pd.DataFrame(results,columns=["game_number","winner","model"])
        print(dfresults.groupby("model")["winner"].value_counts())
        logpath = os.path.join(cfg.LOGDIR,"{}_result.csv".format(self.latest_file_number+1))
        dfresults.to_csv(logpath,index=None)
if __name__=="__main__":
    trainer = Trainer()
    trainer.train()
    
    trainer.evaluate()