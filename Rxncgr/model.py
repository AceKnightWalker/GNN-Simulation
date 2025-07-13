import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, ReLU, Parameter, Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, global_add_pool, GCNConv, GATConv, GINConv, GATv2Conv, TransformerConv, GINEConv


class GIN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_units, depth):
        super(GIN, self).__init__()

        # Hyperparameters
        self.depth = depth
        self.hidden_size = hidden_units
        self.dropout = 0.02  # Dropout rate

        # Layers
        # Initial linear transformation for node features
        self.init = torch.nn.Linear(num_node_features, self.hidden_size)

        # GINConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(self.depth):
            # MLP for GINConv
            mlp = Sequential(
                Linear(self.hidden_size, self.hidden_size),  # Linear layer
                ReLU(),  # Activation function
                Linear(self.hidden_size, self.hidden_size),  # Linear layer
            )
            self.convs.append(GINConv(mlp))  # GINConv layer

        # Global pooling and final feed-forward network
        self.pool = global_add_pool
        self.ffn = torch.nn.Linear(self.hidden_size, 1)  # Output layer

    def forward(self, data):
        # Unpack data
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial transformation of node features
        h_0 = F.relu(self.init(x))
        h = h_0

        # GIN layers with residual connections
        for conv in self.convs:
            h = conv(h, edge_index)  # Apply GINConv
            h += h_0  # Residual connection
            h = F.dropout(F.relu(h), p=self.dropout, training=self.training)  # Apply dropout and ReLU

        # Global pooling and final prediction
        h_pooled = self.pool(h, batch)  # Pool nodes into graph-level representation
        out = self.ffn(h_pooled).squeeze(-1)  # Final prediction (squeeze to remove last dimension)
        return out
    

# 0.9 > than GIN
class GINE(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_units, depth):
        super(GINE, self).__init__()

        self.depth = depth
        self.hidden_size = hidden_units
        self.dropout = 0.02

        # Initial transformation of node features
        self.init = Linear(num_node_features, self.hidden_size)

        # GINEConv layers with edge_dim specified
        self.convs = torch.nn.ModuleList()
        for _ in range(self.depth):
            mlp = Sequential(
                Linear(self.hidden_size, self.hidden_size),
                ReLU(),
                Linear(self.hidden_size, self.hidden_size)
            )
            conv = GINEConv(nn=mlp, edge_dim=num_edge_features)
            self.convs.append(conv)

        # Global pooling and final prediction layer
        self.pool = global_add_pool
        self.ffn = Linear(self.hidden_size, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial node transformation
        h = F.relu(self.init(x))
        h_0 = h  # Save for residuals

        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = h + h_0  # Residual connection
            h = F.dropout(F.relu(h), p=self.dropout, training=self.training)

        # Pooling and final prediction
        h_pooled = self.pool(h, batch)
        out = self.ffn(h_pooled).squeeze(-1)
        return out