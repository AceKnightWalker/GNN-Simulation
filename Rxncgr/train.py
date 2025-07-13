import torch, math
from rxn_cgr1 import construct_loader, ChemDataset
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

from model import GIN, GINE
from GrapTransformations.cell_encoding import CellularRingEncoding
import GrapTransformations.subgraph_bag_encoding as SBE
from torch_geometric.transforms import LineGraph
import time
import torch_geometric as tg

class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, rev=False):
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std

def train_epoch(model, loader, optimizer, loss, stdzer):
    model.train()
    loss_all = 0

    for data in loader:
        optimizer.zero_grad()

        out = model(data)
        result = loss(out, stdzer(data.y))
        result.backward()

        optimizer.step()
        loss_all += loss(stdzer(out, rev=True), data.y)

    return math.sqrt(loss_all / len(loader.dataset))

def pred(model, loader, loss, stdzer):
    model.eval()

    preds, ys = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = stdzer(out, rev=True)
            preds.extend(pred.cpu().detach().tolist())

    return preds

# INDIVIDUAL TESTS

def train(folder, mode='rxn', transform_type=None):
    torch.manual_seed(0)

    # Dynamically select the transformation
    if transform_type == "CellularRing":
        transform = CellularRingEncoding(
            max_ring_size=3, 
            aggr_edge_atr=False, 
            aggr_vertex_feat=True, 
            explicit_pattern_enc=True, 
            edge_attr_in_vertices=False
        )
        print("CellularRingEncoding")
        
    elif transform_type == "Subgraph":
        print("SubGraph")
        policy = SBE.policy2transform("ego_nets", 3)
        transform = SBE.SubgraphBagEncoding(policy)
        #transform = SBE.SubgraphBagEncodingNoSeparator(policy) DS/DSS Not specified


    elif transform_type is None:
        transform = None 
        print("No Transformation applied")
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")


    train_loader = construct_loader(folder + "/train.csv", True, mode=mode, transform=transform, extra=True)
    val_loader = construct_loader(folder + "/val.csv", False, mode=mode, transform=transform, extra=True)
    test_loader = construct_loader(folder + "/test.csv", False, mode=mode, transform=transform, extra=True)

    mean = np.mean(train_loader.dataset.labels)
    std = np.std(train_loader.dataset.labels)
    stdzer = Standardizer(mean, std)

    
    hidden_units = 300
    depth = 3
    heads = 5


    model = GINE(train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features, hidden_units=hidden_units, depth=depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.MSELoss(reduction='sum')
    print(model)

    total_train_time = 0

    for epoch in range(0, 100):
        
        epoch_start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, loss, stdzer)
        preds = pred(model, val_loader, loss, stdzer)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_train_time += epoch_duration


        print(f"Epoch {epoch}: Train RMSE = {train_loss:.4f}, Val RMSE = {root_mean_squared_error(preds, val_loader.dataset.labels):.4f}, Time = {epoch_duration:.2f}s")

    preds = pred(model, test_loader, loss, stdzer)

    test_rmse = root_mean_squared_error(preds, test_loader.dataset.labels)
    test_mae = mean_absolute_error(preds, test_loader.dataset.labels)

    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Total training time: {total_train_time:.2f}s")


# No transformation
#train("/home/abdulfatai/Downloads/Internship/GNN-Simulation/Rxncgr/dataset/e2sn2", transform_type=None)

# CellularRing transformation
#print("NEW TEST\n")
#train("/home/abdulfatai/Downloads/Internship/GNN-Simulation/Rxncgr/dataset/e2sn2", transform_type="CellularRing")


# Subgraph transformation
#print("NEW TEST\n")
#train("/home/abdulfatai/Downloads/Internship/GNN-Simulation/Rxncgr/dataset/e2sn2", transform_type="Subgraph")

