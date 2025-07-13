"""
Completely ignores the graph structure and just applies an MLP to the pooled vertex features
"""

import torch
import torch_scatter
import torch.nn.functional as F
from torch.nn import Linear, ReLU, ModuleList
from torch_geometric.nn import global_add_pool, global_mean_pool

class MLP(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, num_tasks, dropout_rate, graph_pooling = "sum"):
        super(MLP, self).__init__()

        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        self.layers = ModuleList([Linear(num_features, hidden), ReLU()])
        for _ in range(num_layers-1):
            self.layers.append(Linear(hidden, hidden))
            self.layers.append(ReLU())

        self.final_lin = Linear(hidden, num_classes*num_tasks)
        
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise ValueError("unknown pooling")

    def forward(self, data):
        x = self.pool(data.x, data.batch).float()
        for layer in self.layers:
            x = layer(x)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.final_lin(x)

        if self.num_tasks == 1:
            x = x.view(-1, self.num_classes)
        else:
            x.view(-1, self.num_tasks, self.num_classes)
        return x

    def __repr__(self):
        return self.__class__.__name__
    

#Abdulfatai

class MLPX(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, num_tasks, dropout_rate, 
                 graph_pooling="sum", bond_feature_dims=None, use_edge_features=False):
        super(MLPX, self).__init__()

        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.use_edge_features = use_edge_features

        # Add an edge encoder if bond_feature_dims is provided
        if self.use_edge_features and bond_feature_dims is not None:
            self.edge_encoder = torch.nn.ModuleList([
                torch.nn.Embedding(dim, hidden) for dim in bond_feature_dims
            ])

        # Define MLP layers
        self.layers = ModuleList([Linear(num_features, hidden), ReLU()])
        for _ in range(num_layers - 1):
            self.layers.append(Linear(hidden, hidden))
            self.layers.append(ReLU())

        self.final_lin = Linear(hidden, num_classes * num_tasks)

        # Define pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise ValueError("Unknown pooling type.")

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Process edge features (if applicable)
        if self.use_edge_features and edge_attr is not None:
            edge_embedding = 0

            # Normalize edge_attr to ensure all indices are non-negative
            edge_attr_min = edge_attr.min(dim=0)[0]  # Minimum value for each feature column
            edge_attr_shifted = edge_attr - edge_attr_min.unsqueeze(0)  # Shift to make all values non-negative

            # Convert edge_attr_shifted to long (required for embedding)
            edge_attr_shifted = edge_attr_shifted.long()

            for i in range(edge_attr_shifted.shape[1]):  # Iterate over edge features
                edge_embedding += self.edge_encoder[i](edge_attr_shifted[:, i])

            # Ensure edge_embedding has the same feature dimension as x
            if edge_embedding.size(1) != x.size(1):
                edge_embedding = torch.nn.Linear(edge_embedding.size(1), x.size(1)).to(edge_embedding.device)(edge_embedding)

            # Aggregate edge embeddings into node features using torch_scatter
            x = x + torch_scatter.scatter(edge_embedding, edge_index[0], dim=0, dim_size=x.size(0), reduce="mean")

        # Pool node features to obtain graph-level features
        x = self.pool(x, batch).float()

        # Pass through MLP layers
        for layer in self.layers:
            x = layer(x)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.final_lin(x)

        # Reshape output for multi-task learning
        if self.num_tasks == 1:
            x = x.view(-1, self.num_classes)
        else:
            x = x.view(-1, self.num_tasks, self.num_classes)

        return x


    def __repr__(self):
        return self.__class__.__name__
