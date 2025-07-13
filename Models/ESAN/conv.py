"""
Code taken from ogb examples and adapted
"""

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import GINConv as PyGINConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class ZINCGINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(ZINCGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = torch.nn.Embedding(4, in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

#Abdulfatai

def check_tensor_types(x, edge_index, edge_attr):
        # Check type of x
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, but got {type(x)}")
        
        # Check type of edge_index
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(f"Expected edge_index to be a torch.Tensor, but got {type(edge_index)}")
        if edge_index.dtype != torch.long:
            raise TypeError(f"Expected edge_index to be of dtype 'torch.long', but got {edge_index.dtype}")

        # Check type of edge_attr
        if not isinstance(edge_attr, torch.Tensor):
            raise TypeError(f"Expected edge_attr to be a torch.Tensor, but got {type(edge_attr)}")
        

class RxnGINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim, bond_feature_dims):
        super(RxnGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = torch.nn.ModuleList([
            torch.nn.Embedding(dim, emb_dim) for dim in bond_feature_dims
        ])

    def forward(self, x, edge_index, edge_attr):
        bond_embedding = 0
        embedding_result = 0

        check_tensor_types(x, edge_index, edge_attr)

        # Ensure edge_attr is cast to long and is on the same device as x
        edge_attr = edge_attr.to(torch.long).to(device=x.device)

        # Encode edge attributes using bond encoder
        for i in range(edge_attr.shape[1]):

            # Shift the values so they start from 0
            min_value = edge_attr[:, i].min()  # Find the minimum value in the column
            if min_value < 0:
                shift_value = -min_value  # Make the minimum value zero by adding the absolute of the min value
                edge_attr_shifted = edge_attr[:, i] + shift_value  # Create a new tensor with shifted values
                #print(f"Shifted edge_attr[:, {i}] by {shift_value}, new values: {edge_attr_shifted}")
            else:
                edge_attr_shifted = edge_attr[:, i]  # No shift needed if values are already non-negative

            # Check if all values are within the valid range for embedding
            max_index = self.bond_encoder[i].num_embeddings - 1  # Get the max valid index
            min_index = 0  # Assuming indices start from 0

            # Check if there are any out-of-bounds values
            out_of_bounds = edge_attr_shifted < min_index
            out_of_bounds |= edge_attr_shifted > max_index

            if out_of_bounds.any():
                print(f"Warning: Some values in edge_attr[:, {i}] are out of bounds!")
                print(f"Out-of-bounds values: {edge_attr_shifted[out_of_bounds]}")

                # Optionally handle out-of-bounds values (e.g., set them to the max index)
                edge_attr_shifted[out_of_bounds] = max_index
                print(f"Replaced out-of-bounds values with the max index {max_index}")


            # Pass the edge attribute through the bond encoder
            try:
                embedding_result = self.bond_encoder[i](edge_attr_shifted)
                bond_embedding += embedding_result
            except Exception as e:
                print(f"Error during embedding for edge_attr[:, {i}]: {e}")
                break


        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=bond_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    

class OriginalGINConv(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(OriginalGINConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )
        self.layer = PyGINConv(nn=mlp, train_eps=False)

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(in_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + \
               F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, in_dim, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin',
                 num_random_features=0, feature_encoder=lambda x: x, bond_feature_dims=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            bond_feature_dims (list or None): Feature dimensions for edge attributes (optional).
        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.gnn_type = gnn_type
        self.bond_feature_dims = bond_feature_dims

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = feature_encoder
        self.num_random_features = num_random_features

        if num_random_features > 0:
            assert gnn_type == 'graphconv'

            self.initial_layers = torch.nn.ModuleList(
                [GraphConv(in_dim, emb_dim // 2), GraphConv(emb_dim // 2, emb_dim - num_random_features)]
            )
            # now the next layers will have dimension emb_dim
            in_dim = emb_dim

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            # Add support for bond_feature_dims
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'rxngin':
                self.convs.append(RxnGINConv(emb_dim if layer != 0 else in_dim, emb_dim, bond_feature_dims))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'originalgin':
                self.convs.append(OriginalGINConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'zincgin':
                self.convs.append(ZINCGINConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'graphconv':
                self.convs.append(GraphConv(emb_dim if layer != 0 else in_dim, emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        if self.num_random_features > 0:
            for layer in self.initial_layers:
                x = F.elu(layer(x, edge_index, edge_attr))

            # Implementation of RNI
            random_dims = torch.empty(x.shape[0], self.num_random_features).to(x.device)
            torch.nn.init.normal_(random_dims)
            x = torch.cat([x, random_dims], dim=1)

        ### computing input node embedding
        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layer):
            # Pass bond_feature_dims to RxnGINConv if applicable
            if self.bond_feature_dims is not None:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr if edge_attr is not None else None)

            h = self.batch_norms[layer](h)

            if self.gnn_type not in ['gin', 'gcn'] or layer != self.num_layer - 1:
                h = F.relu(h)  # remove last relu for ogb

            if self.drop_ratio > 0.:
                h = F.dropout(h, self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)

        return node_representation




# ### GNN to generate node embedding
# class GNN_node(torch.nn.Module):
#     """
#     Output:
#         node representations
#     """

#     def __init__(self, num_layer, in_dim, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin',
#                  num_random_features=0, feature_encoder=lambda x: x):
#         '''
#             emb_dim (int): node embedding dimensionality
#             num_layer (int): number of GNN message passing layers

#         '''

#         super(GNN_node, self).__init__()
#         self.num_layer = num_layer
#         self.drop_ratio = drop_ratio
#         self.JK = JK
#         ### add residual connection or not
#         self.residual = residual
#         self.gnn_type = gnn_type

#         if self.num_layer < 2:
#             raise ValueError("Number of GNN layers must be greater than 1.")

#         self.atom_encoder = feature_encoder
#         self.num_random_features = num_random_features

#         if num_random_features > 0:
#             assert gnn_type == 'graphconv'

#             self.initial_layers = torch.nn.ModuleList(
#                 [GraphConv(in_dim, emb_dim // 2), GraphConv(emb_dim // 2, emb_dim - num_random_features)]
#             )
#             # now the next layers will have dimension emb_dim
#             in_dim = emb_dim

#         ###List of GNNs
#         self.convs = torch.nn.ModuleList()
#         self.batch_norms = torch.nn.ModuleList()

#         for layer in range(num_layer):
#             if gnn_type == 'gin':
#                 self.convs.append(GINConv(emb_dim if layer != 0 else in_dim, emb_dim))
#             elif gnn_type == 'gcn':
#                 self.convs.append(GCNConv(emb_dim if layer != 0 else in_dim, emb_dim))
#             elif gnn_type == 'originalgin':
#                 self.convs.append(OriginalGINConv(emb_dim if layer != 0 else in_dim, emb_dim))
#             elif gnn_type == 'zincgin':
#                 self.convs.append(ZINCGINConv(emb_dim if layer != 0 else in_dim, emb_dim))
#             elif gnn_type == 'graphconv':
#                 self.convs.append(GraphConv(emb_dim if layer != 0 else in_dim, emb_dim))
#             else:
#                 raise ValueError('Undefined GNN type called {}'.format(gnn_type))

#             self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

#     def forward(self, batched_data):
#         x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

#         if self.num_random_features > 0:
#             for layer in self.initial_layers:
#                 x = F.elu(layer(x, edge_index, edge_attr))

#             # Implementation of RNI
#             random_dims = torch.empty(x.shape[0], self.num_random_features).to(x.device)
#             torch.nn.init.normal_(random_dims)
#             x = torch.cat([x, random_dims], dim=1)

#         ### computing input node embedding
#         h_list = [self.atom_encoder(x)]

#         for layer in range(self.num_layer):

#             h = self.convs[layer](h_list[layer], edge_index, edge_attr)

#             h = self.batch_norms[layer](h)

#             if self.gnn_type not in ['gin', 'gcn'] or layer != self.num_layer - 1:
#                 h = F.relu(h)  # remove last relu for ogb

#             if self.drop_ratio > 0.:
#                 h = F.dropout(h, self.drop_ratio, training=self.training)

#             if self.residual:
#                 h += h_list[layer]

#             h_list.append(h)

#         ### Different implementations of Jk-concat
#         if self.JK == "last":
#             node_representation = h_list[-1]
#         elif self.JK == "sum":
#             node_representation = 0
#             for layer in range(self.num_layer + 1):
#                 node_representation += h_list[layer]
#         elif self.JK == "concat":
#             node_representation = torch.cat(h_list, dim=1)

#         return node_representation
