import os
import csv
import json

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, GNNBenchmarkDataset
import torch.optim as optim
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from GrapTransformations.cell_encoding import CellularRingEncoding, CellularCliqueEncoding
from GrapTransformations.subgraph_bag_encoding import SubgraphBagEncoding, policy2transform, SubgraphBagEncodingNoSeparator
from GrapTransformations.add_zero_edge_attr import AddZeroEdgeAttr
from GrapTransformations.pad_node_attr import PadNodeAttr

from Models.gnn import GNN
# from Models.model import GNN
from Models.encoder import NodeEncoder, EdgeEncoder, ZincAtomEncoder, EgoEncoder, RxnAtomEncoder
from Models.ESAN.conv import GINConv, OriginalGINConv, GCNConv, ZINCGINConv, RxnGINConv 
from Models.ESAN.models import DSnetwork, DSSnetwork
from Models.ESAN.models import GNN as ESAN_GNN
from Models.mlp import MLP, MLPX

import Misc.config as config 


#Abdulfatai

from Rxncgr.rxn_cgr1 import ChemDataset
import numpy as np
import pandas as pd


data_path_base = "/home/abdulfatai/Downloads/Internship/GNN-Simulation/Rxncgr/dataset/full_rdb7/"
file_names = ['train.csv', 'val.csv', 'test.csv']

def get_transform(args):
    transforms = []

    if args.use_cliques:
        print("Using Cliques")
        transforms.append(CellularCliqueEncoding(args.max_struct_size, aggr_edge_atr=args.use_aggr_edge_atr, aggr_vertex_feat=args.use_aggr_vertex_feat,
            explicit_pattern_enc=args.use_expl_type_enc, edge_attr_in_vertices=args.use_edge_attr_in_vertices))
    elif args.use_rings:
        print("Using RIngs")
        transforms.append(CellularRingEncoding(args.max_struct_size, aggr_edge_atr=args.use_aggr_edge_atr, aggr_vertex_feat=args.use_aggr_vertex_feat,
            explicit_pattern_enc=args.use_expl_type_enc, edge_attr_in_vertices=args.use_edge_attr_in_vertices))
    elif not args.use_esan and args.policy != "":
        policy = policy2transform(args.policy, args.num_hops)
        # transform = SubgraphBagEncoding(policy, explicit_type_enc=args.use_expl_type_enc)
        print("Using new SBE")
        transforms.append(SubgraphBagEncodingNoSeparator(policy, 
            explicit_type_enc=args.use_expl_type_enc, dss_message_passing = args.use_dss_message_passing, 
            connect_accross_subg = args.use_additonal_gt_con))
        
    elif args.use_esan and args.policy != "":
        transforms.append(policy2transform(args.policy, args.num_hops))

    else:
        print("No Transformation")    
    return Compose(transforms)

def load_dataset(args, config):
    transform = get_transform(args)

    if transform is None:
        dir = os.path.join(config.DATA_PATH, args.dataset, "Original")
    else:
        print(repr(transform))
        trafo_str = repr(transform).replace("\n", "").replace(" ","")
        dir = os.path.join(config.DATA_PATH, args.dataset, trafo_str )

    if args.dataset.lower() == "zinc":
        datasets = [ZINC(root=dir, subset=True, split=split, pre_transform=transform) for split in ["train", "val", "test"]]

    elif args.dataset.lower() == "rxn_cgr":

        # data_path_base = "/home/abdulfatai/Internship/GNN-Simulation/Misc/BenchTransforms/rxn_test_data/e2sn2/"
        # file_names = ['train.csv', 'val.csv', 'test.csv']  # List of CSV files to process

        # List comprehension to process all datasets in a single line
        datasets = [
            ChemDataset(
                smiles=pd.read_csv(data_path_base + file_name).iloc[:, 1].values, #Present in the 2nd column
                labels=pd.read_csv(data_path_base + file_name).iloc[:, 2].values.astype(np.float32), #Present in the 3rd column
                transform = transform,
                extra = False
            ) for file_name in file_names
        ]


    else:
        raise NotImplementedError("Unknown dataset")
        
    if args.use_esan:
        print("Using ESAN")
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True, follow_batch=['subgraph_idx'])
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
    else:
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_model(args, num_classes, num_vertex_features, num_tasks):


    transform = get_transform(args)

    datasets = [
        ChemDataset(
            smiles=pd.read_csv(data_path_base + file_name).iloc[:, 1].values,
            labels=pd.read_csv(data_path_base + file_name).iloc[:, 2].values.astype(np.float32),
            transform = transform,
            extra = False
        ) for file_name in file_names
    ]


    # Placeholder for max ranges
    node_feature_ranges = None
    bond_feature_ranges = None

    for dataset in datasets:
        for data in dataset:
            # Assuming `data.x` contains node features and `data.edge_attr` contains bond features
            if node_feature_ranges is None:
                node_feature_ranges = data.x.min(dim=0).values, data.x.max(dim=0).values
            else:
                node_feature_ranges = (
                    torch.min(node_feature_ranges[0], data.x.min(dim=0).values),
                    torch.max(node_feature_ranges[1], data.x.max(dim=0).values),
                )
            
            if data.edge_attr is not None:  # Check if edge_attr exists
                if bond_feature_ranges is None:
                    bond_feature_ranges = data.edge_attr.min(dim=0).values, data.edge_attr.max(dim=0).values
                else:
                    bond_feature_ranges = (
                        torch.min(bond_feature_ranges[0], data.edge_attr.min(dim=0).values),
                        torch.max(bond_feature_ranges[1], data.edge_attr.max(dim=0).values),
                    )

    # Compute the feature dimensions
    node_feature_dims = [
        int(node_feature_ranges[1][i] - node_feature_ranges[0][i] + 1)
        for i in range(len(node_feature_ranges[0]))
    ]
    bond_feature_dims = [
        int(bond_feature_ranges[1][i] - bond_feature_ranges[0][i] + 1)
        for i in range(len(bond_feature_ranges[0]))
    ]


    if not args.use_esan:
        node_feature_dims = []
        bond_feature_dims = []
        
        model = args.model.lower()

        if args.use_expl_type_enc:
            if args.use_rings:
                node_feature_dims = [2,2,2]              
            if args.use_cliques:
                for _ in range(args.max_struct_size):
                    node_feature_dims.append(2)
            elif args.policy != "":
                node_feature_dims = [2,2]

        if args.dataset.lower() == "zinc":
            node_feature_dims += [21]
            if args.edge_attr_in_vertices:
                node_feature_dims += [4]
            
            node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
            edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=[4])


        elif args.dataset.lower() == "rxn_cgr": 


            transform = get_transform(args)

            # data_path_base = "/home/abdulfatai/Internship/GNN-Simulation/Misc/BenchTransforms/rxn_test_data/e2sn2/"
            # file_names = ['train.csv', 'val.csv', 'test.csv']  # List of CSV files to process

            # List comprehension to process all datasets in a single line
            datasets = [
                ChemDataset(
                    smiles=pd.read_csv(data_path_base + file_name).iloc[:, 1].values,
                    labels=pd.read_csv(data_path_base + file_name).iloc[:, 2].values.astype(np.float32),
                    transform = transform,
                    extra = False
                ) for file_name in file_names
            ]


            # Placeholder for max ranges
            node_feature_ranges = None
            bond_feature_ranges = None

            for dataset in datasets:
                for data in dataset:
                    # Assuming `data.x` contains node features and `data.edge_attr` contains bond features
                    if node_feature_ranges is None:
                        node_feature_ranges = data.x.min(dim=0).values, data.x.max(dim=0).values
                    else:
                        node_feature_ranges = (
                            torch.min(node_feature_ranges[0], data.x.min(dim=0).values),
                            torch.max(node_feature_ranges[1], data.x.max(dim=0).values),
                        )
                    
                    if data.edge_attr is not None:  # Check if edge_attr exists
                        if bond_feature_ranges is None:
                            bond_feature_ranges = data.edge_attr.min(dim=0).values, data.edge_attr.max(dim=0).values
                        else:
                            bond_feature_ranges = (
                                torch.min(bond_feature_ranges[0], data.edge_attr.min(dim=0).values),
                                torch.max(bond_feature_ranges[1], data.edge_attr.max(dim=0).values),
                            )

            # Compute the feature dimensions
            node_feature_dims = [
                int(node_feature_ranges[1][i] - node_feature_ranges[0][i] + 1)
                for i in range(len(node_feature_ranges[0]))
            ]
            bond_feature_dims = [
                int(bond_feature_ranges[1][i] - bond_feature_ranges[0][i] + 1)
                for i in range(len(bond_feature_ranges[0]))
            ]
            
            if args.edge_attr_in_vertices:
                node_feature_dims += bond_feature_dims
                

            node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
            edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=bond_feature_dims)


        else:
            node_encoder, edge_encoder = lambda x: x, lambda x: x
                
        if model in ["gin", "gcn", "gat"]:
            # Cell Encoding
            if args.use_cliques or args.use_rings:
                return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim, 
                        gnn_type =  model, virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = args.jk, 
                        graph_pooling = args.pooling, type_pooling = args.use_dim_pooling,
                        max_type = 3 if args.use_rings else args.max_struct_size, edge_encoder=edge_encoder, node_encoder=node_encoder, 
                        use_node_encoder = args.use_node_encoder, num_mlp_layers = args.num_mlp_layers)
            # Without Cell Encoding
            else:
                return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim, 
                        gnn_type = model, virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = args.jk, 
                        graph_pooling = args.pooling, type_pooling = args.use_dim_pooling,
                        max_type = 2 if args.policy != "" else 0, edge_encoder=edge_encoder, node_encoder=node_encoder, 
                        use_node_encoder = args.use_node_encoder, num_mlp_layers = args.num_mlp_layers)
        elif args.model.lower() == "mlp":
                if args.dataset.lower() == "rxn_cgr":
                    return MLPX(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim, 
                        num_classes=num_classes, num_tasks=num_tasks, dropout_rate=args.drop_out, graph_pooling=args.pooling, bond_feature_dims = bond_feature_dims, use_edge_features = True)
                else:
                    return MLP(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim, 
                        num_classes=num_classes, num_tasks=num_tasks, dropout_rate=args.drop_out, graph_pooling=args.pooling)
        else:
            print("Invalid model given.")

    # ESAN
    else:
        encoder = lambda x: x
        if 'ZINC' in args.dataset:
            encoder = ZincAtomEncoder(policy=args.policy, emb_dim=args.emb_dim)
        
        elif 'rxn_cgr' in args.dataset:
            encoder = RxnAtomEncoder(policy=args.policy, emb_dim=args.emb_dim)
        
        if 'ZINC' or 'rxn_cgr' in args.dataset:
            in_dim = args.emb_dim if args.policy != "ego_nets_plus" else args.emb_dim + 2
        
        
        else:
            in_dim = dataset.num_features

        # DSS
        if args.use_dss_message_passing:

            if args.model != 'rxngin':
                bond_feature_dims = None 
            if args.model == 'GIN':
                GNNConv = GINConv
            elif args.model == 'originalgin':
                GNNConv = OriginalGINConv
            elif args.model == 'graphconv':
                GNNConv = GraphConv
            elif args.model == 'gcn':
                GNNConv = GCNConv
            elif args.model == 'ZINCGIN':
                GNNConv = ZINCGINConv
            elif args.model == 'rxngin':
                GNNConv = RxnGINConv
                bond_feature_dims = bond_feature_dims
            else:
                raise ValueError('Undefined GNN type called {}'.format(args.model))

        
            model = DSSnetwork(num_layers=args.num_layers, in_dim=in_dim, emb_dim=args.emb_dim, num_tasks=num_tasks*num_classes,
                            feature_encoder=encoder, GNNConv=GNNConv, bond_feature_dims=bond_feature_dims)
        # DS
        else:

            if args.model == 'rxngin':
                GNNConv = RxnGINConv
                bond_feature_dims = bond_feature_dims

            else:
                bond_feature_dims = None
            subgraph_gnn = ESAN_GNN(gnn_type=args.model.lower(), num_tasks=num_tasks*num_classes, num_layer=args.num_layers, in_dim=in_dim,
                           emb_dim=args.emb_dim, drop_ratio=args.drop_out, graph_pooling='sum' if args.model != 'gin' else 'mean', 
                           feature_encoder=encoder, bond_feature_dims=bond_feature_dims)
                           
            model = DSnetwork(subgraph_gnn=subgraph_gnn, channels=[64, 64], num_tasks=num_tasks*num_classes,
                            invariant=args.dataset == 'ZINC' or args.dataset == 'rxn_cgr', bond_feature_dims=bond_feature_dims)
        return model


def get_optimizer_scheduler(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    elif args.lr_scheduler == "ReduceLROnPlateau":
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='min',
                                                                factor=args.lr_scheduler_decay_rate,
                                                                patience=args.lr_schedule_patience,
                                                                verbose=True)
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    return optimizer, scheduler

def get_loss(args):
    metric_method = None
    if args.dataset.lower() == "zinc":
        loss = torch.nn.L1Loss()
        metric = "mae"


    elif args.dataset.lower() == "rxn_cgr":
        loss = torch.nn.L1Loss()
        metric = "mae"

    else:
        raise NotImplementedError("No loss for this dataset")
    
    return {"loss": loss, "metric": metric, "metric_method": metric_method}

def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method