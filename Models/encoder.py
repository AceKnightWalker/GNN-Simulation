import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 


class NodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims=None):
        super(NodeEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        if feature_dims is None:
            feature_dims = get_atom_feature_dims()

        # Create embeddings for each atom feature dimension
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        atom_embedding = 0

        assert x.dim() == 2, "x should have two dimensions: [num_nodes, num_features]"

        # Normalize x to ensure all indices are non-negative
        # Shift values by subtracting the minimum value for each feature
        x_min = x.min(dim=0)[0]  # Minimum value for each feature column
        x_shifted = x - x_min.unsqueeze(0)

        x_shifted = x_shifted.long()

        for i in range(x_shifted.shape[1]):  
            if i < len(self.atom_embedding_list): 
                max_index = x_shifted[:, i].max().item()
                # print(f"Feature {i}: Min index = {x_shifted[:, i].min().item()}, Max index = {max_index}")

                # Check against the embedding dimension
                if max_index >= self.atom_embedding_list[i].num_embeddings:
                    raise ValueError(
                        f"Feature {i} contains indices ({max_index}) outside the embedding range "
                        f"[0, {self.atom_embedding_list[i].num_embeddings - 1}]"
                    )

                # Embed the feature
                atom_embedding += self.atom_embedding_list[i](x_shifted[:, i])
            else:
                print(f"Warning (Node Encoding): Index {i} out of range for atom embedding list")
                break

        return atom_embedding
        

class EdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims=None):
        super(EdgeEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()


        if feature_dims is None:
            feature_dims = get_bond_feature_dims()

        # Create embeddings for each bond feature dimension
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0

        assert edge_attr.dim() == 2, "edge_attr should have two dimensions: [num_edges, num_features]"

        # Normalize edge_attr to ensure all indices are non-negative
        # Shift values by subtracting the minimum value for each feature
        edge_attr_min = edge_attr.min(dim=0)[0]  # Minimum value for each feature column
        edge_attr_shifted = edge_attr - edge_attr_min.unsqueeze(0)

        for i in range(edge_attr_shifted.shape[1]):  
            if i < len(self.bond_embedding_list): 
                max_index = edge_attr_shifted[:, i].max().item()
                #print(f"Feature {i}: Min index = {edge_attr_shifted[:, i].min().item()}, Max index = {max_index}")

                # Check against the embedding dimension
                if max_index >= self.bond_embedding_list[i].num_embeddings:
                    raise ValueError(
                        f"Feature {i} contains indices ({max_index}) outside the embedding range "
                        f"[0, {self.bond_embedding_list[i].num_embeddings - 1}]"
                    )

                # Embed the feature
                bond_embedding += self.bond_embedding_list[i](edge_attr_shifted[:, i])
            else:
                print(f"Warning (Bond Encoding): Index {i} out of range for bond embedding list")
                break

        return bond_embedding

    

class EgoEncoder(torch.nn.Module):
    # From ESAN
    def __init__(self, encoder):
        super(EgoEncoder, self).__init__()
        self.num_added = 2
        self.enc = encoder

    def forward(self, x):
        return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:])))


class ZincAtomEncoder(torch.nn.Module):
    # From ESAN
    def __init__(self, policy, emb_dim):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(21, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_nets_plus':
            return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:].squeeze())))
        else:
            return self.enc(x.squeeze())
        

#Abdulfatai
"""
Only useful if we select the ego_nets_policy. Worse case scenario, we deactivate.
"""

class RxnAtomEncoder(torch.nn.Module):
    def __init__(self, policy, emb_dim):
        super(RxnAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2

        # Linear layer to map one-hot-encoded features to embeddings
        self.enc = torch.nn.Linear(78, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_nets_plus':
            # Pass the first `num_added` features as-is, encode the remaining one-hot features
            return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:])))
        else:
            # Encode the full one-hot feature vector
            return self.enc(x)
