import argparse

from loader import BioDataset
from dataloader import DataLoaderFinetune
from splitters import random_split, species_split, random_split_abs_value

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
import graph_prompt as Prompt
import pandas as pd

import os
import pickle
import wandb

criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, loader, optimizer, prompt):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch, prompt)
        y = batch.go_target_downstream.view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader, prompt):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch, prompt)

        y_true.append(batch.go_target_downstream.view(pred.shape).detach().cpu())
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_scores = torch.cat(y_scores, dim = 0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            roc_list.append(roc_auc_score(y_true[:,i], y_scores[:,i]))
        else:
            roc_list.append(np.nan)

    return np.array(roc_list) #y_true.shape[1]

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='Number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='Embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='How the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--model_file', type=str, default = '', help='File path to read the model (if there is any)')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--tuning_type', type=str, default="gpf", help='\'gpf\' for GPF and \'gpf-plus\' for GPF-plus in the paper')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default = 0, help='Number of workers for dataset loading')
    parser.add_argument('--eval_train', type=int, default = 1, help='Evaluating training or not')
    parser.add_argument('--split', type=str, default = "species", help='The way of dataset split(e.g., \'species\' for bio data)')
    parser.add_argument('--num_layers', type=int, default = 1, help='A range of [1,2,3]-layer MLPs with equal width')
    parser.add_argument('--pnum', type=int, default = 5, help='The number of independent basis for GPF-plus')
    parser.add_argument('--shot_number', type=int, default = 50, help='Number of shots')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)


    root_supervised = 'dataset/supervised'

    dataset = BioDataset(root_supervised, data_type='supervised')

    print(dataset)

    if args.split == "random":
        print("random splitting")
        train_dataset, valid_dataset, test_dataset = random_split(dataset, seed = args.seed) 
    elif args.split == "species":
        trainval_dataset, test_dataset = species_split(dataset)
        train_dataset, valid_dataset, _ = random_split_abs_value(trainval_dataset, seed = args.seed,
                                                                 number_train=args.shot_number,
                                                                 frac_valid=0.15, frac_test=0)
        test_dataset_broad, test_dataset_none, _ = random_split(test_dataset, seed = args.seed, frac_train=0.5,
                                                                frac_valid=0.5, frac_test=0)
        print("species splitting")
    else:
        raise ValueError("Unknown split name.")

    train_loader = DataLoaderFinetune(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoaderFinetune(valid_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    if args.split == "random":
        test_loader = DataLoaderFinetune(test_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
    else:
        ### for species splitting
        test_easy_loader = DataLoaderFinetune(test_dataset_broad, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_hard_loader = DataLoaderFinetune(test_dataset_none, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)

    num_tasks = len(dataset[0].go_target_downstream)

    print(train_dataset[0])

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, layer_number = args.num_layers)

    if not args.model_file == "":
        model.from_pretrained(args.model_file)
    
    model.to(device)

    if args.tuning_type == 'gpf':
        prompt = Prompt.SimplePrompt(args.emb_dim).to(device)
    elif args.tuning_type == 'gpf-plus':
        prompt = Prompt.GPFplusAtt(args.emb_dim, args.pnum).to(device)

    #set up optimizer
    model_param_group = []
    model_param_group.append({"params": prompt.parameters(), "lr": args.lr})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    
    ### for random splitting
    test_acc_list = []
    
    ### for species splitting
    test_acc_easy_list = []
    test_acc_hard_list = []



    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer, prompt)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader, prompt)
        else:
            train_acc = 0
            print("ommitting training evaluation")
        val_acc = eval(args, model, device, val_loader, prompt)

        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)

        if args.split == "random":
            test_acc = eval(args, model, device, test_loader, prompt)
            test_acc_list.append(test_acc)
        else:
            test_acc_easy = eval(args, model, device, test_easy_loader, prompt)
            test_acc_hard = eval(args, model, device, test_hard_loader, prompt)
            test_acc_easy_list.append(test_acc_easy)
            test_acc_hard_list.append(test_acc_hard)
            print(test_acc_easy)
            print(test_acc_hard)

        print("")
    
    with open('result.log', 'a+') as f:
        f.write(str(args.runseed) + ' ' + str(np.array(test_acc_easy_list)[np.array(val_acc_list).argmax()]) + ' ' + str(np.array(test_acc_hard_list)[np.array(val_acc_list).argmax()]))
        f.write('\n')



if __name__ == "__main__":
    main()