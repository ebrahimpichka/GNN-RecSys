import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        """
        Initialize LightGCN model.
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Dimension of embeddings
            num_layers: Number of propagation layers
        """
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
    
    def forward(self, edge_index):
        """
        Forward pass through the LightGCN model.
        
        Args:
            edge_index: Edge indices of the graph
            
        Returns:
            final_user_embedding: User embeddings after propagation
            final_item_embedding: Item embeddings after propagation
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        embs = [all_emb]
        
        for lyr in range(self.num_layers):
            all_emb = self.propagate(edge_index, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)
    
        user_embs, item_embs = torch.split(embs, [self.num_users, self.num_items])
        
        return user_embs, item_embs
    
    def compute_normalized_adj(self, edge_index, num_nodes):
        """
        Compute normalized adjacency matrix for message passing.
        
        Args:
            edge_index: Edge indices
            num_nodes: Total number of nodes
            
        Returns:
            edge_index: Normalized edge indices
            norm: Normalization factors for each edge
        """
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float).to(edge_index.device)
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return edge_index, norm
    
    def propagate(self, edge_index, features):
        """
        Propagate features over the graph.
        
        Args:
            edge_index_norm: Normalized edge indices
            features: Node features
            
        Returns:
            Updated features after propagation
        """
        edge_index_norm, norm = self.compute_normalized_adj(edge_index, features.size(0))
        
        row, col = edge_index
        features_j = features[col]
        features_j = features_j * norm.view(-1, 1)
        
        features_agg = torch.zeros_like(features)
        features_agg.index_add_(0, row, features_j)
        
        return features_agg
    
    def predict(self, user_indices, item_indices, user_emb=None, item_emb=None):
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_indices: User indices
            item_indices: Item indices
            user_emb: Pre-computed user embeddings (optional)
            item_emb: Pre-computed item embeddings (optional)
            
        Returns:
            predictions: Predicted ratings
        """
        if user_emb is None or item_emb is None:
            user_emb = self.user_embedding.weight
            item_emb = self.item_embedding.weight
        
        user_vectors = user_emb[user_indices]
        item_vectors = item_emb[item_indices]
        
        predictions = torch.sum(user_vectors * item_vectors, dim=1)
        
        return predictions

class LightGCNv2(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3,
                 user_feat_dim=None, item_feat_dim=None, feature_projection_dim=64):
        """
        Initialize LightGCNv2 model.
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Dimension of embeddings
            num_layers: Number of propagation layers
            user_feat_dim: Original dimension of user features (if any)
            item_feat_dim: Original dimension of item features (if any)
            feature_projection_dim: Dimension to project features to before concatenation
        """
        super(LightGCNv2, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.feature_projection_dim = feature_projection_dim
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        if self.user_feat_dim is not None:
            self.user_feature_transform = nn.Linear(self.user_feat_dim, self.feature_projection_dim)
            self.user_concat_transform = nn.Linear(self.embedding_dim + self.feature_projection_dim, self.embedding_dim)
            nn.init.xavier_uniform_(self.user_feature_transform.weight)
            nn.init.xavier_uniform_(self.user_concat_transform.weight)

        if self.item_feat_dim is not None:
            self.item_feature_transform = nn.Linear(self.item_feat_dim, self.feature_projection_dim)
            self.item_concat_transform = nn.Linear(self.embedding_dim + self.feature_projection_dim, self.embedding_dim)
            nn.init.xavier_uniform_(self.item_feature_transform.weight)
            nn.init.xavier_uniform_(self.item_concat_transform.weight)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, edge_index, user_features=None, item_features=None):
        """
        Forward pass through the LightGCNWithFeatures model.
        
        Args:
            edge_index: Edge indices of the graph
            user_features: Tensor of user features (optional)
            item_features: Tensor of item features (optional)
            
        Returns:
            final_user_embedding: User embeddings after propagation
            final_item_embedding: Item embeddings after propagation
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        if self.user_feat_dim is not None and user_features is not None:
            if user_features.shape[0] != self.num_users or user_features.shape[1] != self.user_feat_dim:
                raise ValueError(f"User features shape mismatch. Expected ({self.num_users}, {self.user_feat_dim}), got {user_features.shape}")
            projected_user_feat = F.relu(self.user_feature_transform(user_features))
            user_emb = torch.cat([user_emb, projected_user_feat], dim=1)
            user_emb = F.relu(self.user_concat_transform(user_emb))

        if self.item_feat_dim is not None and item_features is not None:
            if item_features.shape[0] != self.num_items or item_features.shape[1] != self.item_feat_dim:
                raise ValueError(f"Item features shape mismatch. Expected ({self.num_items}, {self.item_feat_dim}), got {item_features.shape}")
            projected_item_feat = F.relu(self.item_feature_transform(item_features))
            item_emb = torch.cat([item_emb, projected_item_feat], dim=1)
            item_emb = F.relu(self.item_concat_transform(item_emb))
        
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        embs = [all_emb]
        
        for _ in range(self.num_layers):
            all_emb = self.propagate(edge_index, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
    
        final_user_emb, final_item_emb = torch.split(final_emb, [self.num_users, self.num_items])
        
        return final_user_emb, final_item_emb
    
    def compute_normalized_adj(self, edge_index, num_nodes):
        """
        Compute normalized adjacency matrix for message passing.
        """
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float).to(edge_index.device)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return edge_index, norm
    
    def propagate(self, edge_index, features):
        """
        Propagate features over the graph.
        """
        edge_index_norm, norm = self.compute_normalized_adj(edge_index, features.size(0))
        row, col = edge_index_norm
        
        aggregated_messages = features[col] * norm.view(-1, 1)
        
        out_features = torch.zeros_like(features)
        out_features.index_add_(0, row, aggregated_messages)
        
        return out_features

    def predict(self, user_indices, item_indices, user_emb_final, item_emb_final):
        """
        Predict ratings for user-item pairs using final embeddings.
        
        Args:
            user_indices: User indices
            item_indices: Item indices
            user_emb_final: Final user embeddings from the forward pass
            item_emb_final: Final item embeddings from the forward pass
            
        Returns:
            predictions: Predicted ratings (dot product)
        """
        user_vectors = user_emb_final[user_indices]
        item_vectors = item_emb_final[item_indices]
        predictions = torch.sum(user_vectors * item_vectors, dim=1)
        return predictions
