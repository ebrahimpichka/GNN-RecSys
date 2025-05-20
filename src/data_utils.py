import os
import urllib.request
import zipfile
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import json
import gc

def process_yelp_data_in_chunks(file_path, chunk_size=10000, output_dir="processed_chunks"):
    """Process large JSON files in chunks to avoid memory errors."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk_num = 0
    current_chunk = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            try:
                data = json.loads(line)
                current_chunk.append(data)
            except json.JSONDecodeError:
                continue

            if len(current_chunk) >= chunk_size:
                df_chunk = pd.DataFrame(current_chunk)

                # Save chunk to disk (either as CSV or as pickle)
                chunk_file = os.path.join(
                    output_dir, f'chunk_{str(chunk_num).zfill(3)}.parquet')
                df_chunk.to_parquet(chunk_file, index=False)

                # Clear memory
                del df_chunk
                current_chunk = []
                gc.collect()
                chunk_num += 1

        if current_chunk:
            df_chunk = pd.DataFrame(current_chunk)
            chunk_file = os.path.join(output_dir, f'chunk_{str(chunk_num).zfill(3)}.parquet')
            df_chunk.to_parquet(chunk_file, index=False)
            del df_chunk
            current_chunk = []
            gc.collect()
            chunk_num += 1
    print(f"Processed {chunk_num} chunks and saved to {output_dir}")


def create_graph_from_chunks(chunks_dir, user_mapping_file=None, business_mapping_file=None):
    """Create a PyG graph from processed chunks."""
    if user_mapping_file and os.path.exists(user_mapping_file):
        user_to_idx = pd.read_pickle(user_mapping_file)
    else:
        user_to_idx = {}

    if business_mapping_file and os.path.exists(business_mapping_file):
        business_to_idx = pd.read_pickle(business_mapping_file)
    else:
        business_to_idx = {}

    edge_index = [[], []]  # [user_indices, business_indices]
    # edge_attr = []  # ratings

    for chunk_file in tqdm(sorted(os.listdir(chunks_dir))):
        if not chunk_file.endswith('.parquet'):
            continue

        chunk_path = os.path.join(chunks_dir, chunk_file)
        df_chunk = pd.read_parquet(chunk_path)

        for _, row in df_chunk.iterrows():
            user_id = row['user_id']
            business_id = row['business_id']
            rating = row['stars']

            if user_id not in user_to_idx:
                user_to_idx[user_id] = len(user_to_idx)
            if business_id not in business_to_idx:
                business_to_idx[business_id] = len(business_to_idx)

            # Add edge
            if rating >= 4:
                edge_index[0].append(user_to_idx[user_id])
                edge_index[1].append(business_to_idx[business_id])
            # edge_attr.append(rating)

        del df_chunk
        gc.collect()

    num_users = len(user_to_idx)
    num_businesses = len(business_to_idx)
    # Create graph
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    direction = torch.ones(len(edge_index_tensor[0]), dtype=torch.long)
    edge_direction = torch.concat([direction, 2*direction],) # will be used for splitting
    edge_index_tensor[1] = edge_index_tensor[1] + num_users  # Adjust business indices
    # reverse direction (undirected graph)
    edge_index_tensor = torch.cat([edge_index_tensor, torch.stack([edge_index_tensor[1], edge_index_tensor[0]])], dim=1)
    # edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

    # Save mappings for future use
    if user_mapping_file:
        pd.to_pickle(user_to_idx, user_mapping_file)
    if business_mapping_file:
        pd.to_pickle(business_to_idx, business_mapping_file)

    return Data(
        edge_index=edge_index_tensor,
        edge_direction=edge_direction,
        num_nodes=num_users + num_businesses,
        num_users=num_users,
        num_businesses=num_businesses,
        num_items=num_businesses,
        # edge_attr=edge_attr_tensor
        )

def download_dataset(name):
    """Download dataset if not already available."""
    if name.lower() == 'movielens-100k':
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zip_path = "ml-100k.zip"
        extract_path = "ml-100k"
        
        if not os.path.exists(extract_path):
            print(f"Downloading {name} dataset...")
            urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            
            if os.path.exists(zip_path):
                os.remove(zip_path)
        
        return extract_path
    
    elif name.lower() == 'movielens-1m':
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = "ml-1m.zip"
        extract_path = "ml-1m"
        
        if not os.path.exists(extract_path):
            print(f"Downloading {name} dataset...")
            urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            
            if os.path.exists(zip_path):
                os.remove(zip_path)
        
        return extract_path
    elif name.lower() == 'yelp':
        # Assuming yelp dataset is in ./yelp/Yelp-JSON/Yelp JSON
        extract_path = os.path.join("yelp", "Yelp-JSON", "Yelp JSON") 
        if not os.path.exists(extract_path):
            # If you have a specific path where Yelp data is stored, you can set it here.
            # For example: extract_path = "C:/path/to/your/Yelp-JSON"
            raise FileNotFoundError(f"Yelp dataset not found at {extract_path}. "
                                    f"Please ensure the Yelp JSON files are in the correct directory.")
        print(f"Using Yelp dataset from {extract_path}")
        return extract_path
    else:
        raise ValueError(f"Dataset {name} not supported")

def load_movielens_100k(path):
    """Load MovieLens-100K dataset and return user-item interactions."""
    data_file = os.path.join(path, 'u.data')
    
    df = pd.read_csv(data_file, sep='\t', header=None, 
                    names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    df_positive = df[df['rating'] >= 4].copy()
    
    unique_users = df_positive['user_id'].unique()
    unique_items = df_positive['item_id'].unique()
    
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    df_positive['user_id'] = df_positive['user_id'].map(user_id_map)
    df_positive['item_id'] = df_positive['item_id'].map(item_id_map)

    user_nodes = df_positive['user_id'].values
    item_nodes = df_positive['item_id'].values
    
    return user_nodes, item_nodes, len(unique_users), len(unique_items)

def load_movielens_1m(path):
    """Load MovieLens-1M dataset and return user-item interactions."""
    data_file = os.path.join(path, 'ratings.dat')
    
    df = pd.read_csv(data_file, sep='::', header=None, engine='python',
                    names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    df_positive = df[df['rating'] >= 4].copy()
    
    unique_users = df_positive['user_id'].unique()
    unique_items = df_positive['item_id'].unique()
    
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    df_positive['user_id'] = df_positive['user_id'].map(user_id_map)
    df_positive['item_id'] = df_positive['item_id'].map(item_id_map)
    
    user_nodes = df_positive['user_id'].values
    item_nodes = df_positive['item_id'].values
    
    return user_nodes, item_nodes, len(unique_users), len(unique_items)

def create_graph_data(user_nodes, item_nodes, num_users, num_items):
    """Create PyTorch Geometric graph data from user-item interactions."""
   
    user_indices = torch.tensor(user_nodes, dtype=torch.long)
    item_indices = torch.tensor(item_nodes, dtype=torch.long)
    
    edge_index = torch.stack([
        torch.cat([user_indices, item_indices + num_users]),
        torch.cat([item_indices + num_users, user_indices])
    ], dim=0)
    
    data = Data(edge_index=edge_index, num_nodes=num_users + num_items)
    
    data.num_users = num_users
    data.num_items = num_items
    
    return data


def split_yelp_edges(data, test_ratio=0.2, val_ratio=0.05, save_path=None):
    """Split edges into train, validation, and test sets."""
    # Get user-item edges (only one direction)
    edge_direction = data.edge_direction
    src2dst_mask = (edge_direction == 1).flatten()
    src2dst = edge_direction[src2dst_mask]

    train_mask = torch.zeros_like(src2dst, dtype=torch.bool)
    val_mask = torch.zeros_like(src2dst, dtype=torch.bool)
    test_mask = torch.zeros_like(src2dst, dtype=torch.bool)


    indices = np.arange(len(src2dst))
    train_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=42)
    if val_ratio > 0:
        val_size = val_ratio / (1 - test_ratio)  # Adjusted ratio from remaining train data
        train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=43)

    train_mask[train_indices] = True
    if val_ratio > 0:
        val_mask[val_indices] = True
    test_mask[test_indices] = True

    train_mask = torch.cat([train_mask, train_mask]) #.T.flatten()
    val_mask = torch.cat([val_mask, val_mask])
    test_mask = torch.cat([test_mask, test_mask])

    train_edges = data.edge_index[:, train_mask]
    val_edges = data.edge_index[:, val_mask]
    test_edges = data.edge_index[:, test_mask]

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    num_users = data.num_users
    data.train_edges = [(u, i - num_users) for u, i in train_edges.t().tolist() if i >= num_users]
    data.val_edges = [(u, i - num_users) for u, i in val_edges.t().tolist() if i >= num_users]
    data.test_edges = [(u, i - num_users) for u, i in test_edges.t().tolist() if i >= num_users]

    if save_path:
        torch.save(data, save_path) # for effiecnt loading later
        print(f"Saved processed graph data to {save_path}")

    return data

def split_edges(data, test_ratio=0.2, val_ratio=0.05, save_path=None):
    """Split edges into train, validation, and test sets."""
    user_idx, item_idx = data.edge_index[0], data.edge_index[1]
    user_item_mask = user_idx < data.num_users
    
    user_idx = user_idx[user_item_mask]
    item_idx = item_idx[user_item_mask] - data.num_users
    
    edges = list(zip(user_idx.numpy(), item_idx.numpy()))
    
    train_edges, test_edges = train_test_split(edges, test_size=test_ratio, random_state=42)
    
    if val_ratio > 0:
        val_size = val_ratio / (1 - test_ratio)
        train_edges, val_edges = train_test_split(train_edges, test_size=val_size, random_state=42)
    else:
        val_edges = []
    
    edge_tuples = list(zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()))
    train_mask = torch.zeros(len(edge_tuples), dtype=torch.bool)
    val_mask = torch.zeros(len(edge_tuples), dtype=torch.bool)
    test_mask = torch.zeros(len(edge_tuples), dtype=torch.bool)
    
    train_edges_orig = [(u, i + data.num_users) for u, i in train_edges]
    val_edges_orig = [(u, i + data.num_users) for u, i in val_edges]
    test_edges_orig = [(u, i + data.num_users) for u, i in test_edges]
    
    train_edges_all = train_edges_orig + [(i, u) for u, i in train_edges_orig]
    val_edges_all = val_edges_orig + [(i, u) for u, i in val_edges_orig]
    test_edges_all = test_edges_orig + [(i, u) for u, i in test_edges_orig]
    
    for idx, edge in enumerate(edge_tuples):
        if edge in train_edges_all:
            train_mask[idx] = True
        elif edge in val_edges_all:
            val_mask[idx] = True
        elif edge in test_edges_all:
            test_mask[idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    data.train_edges = train_edges
    data.val_edges = val_edges 
    data.test_edges = test_edges
    
    if save_path: 
        torch.save(data, save_path)
        print(f"Saved processed graph data to {save_path}")
    
    return data

class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, edge_index, mask, num_users, num_items, negative_sampling_ratio=1):
        """
        Dataset for edge data with negative sampling.
        
        Args:
            edge_index: PyG edge index tensor
            mask: Boolean mask for selecting edges
            num_users: Number of users
            num_items: Number of items
            negative_sampling_ratio: Ratio of negative samples to positive samples
        """
        self.positive_edges = edge_index.t()[mask].cpu()
        self.num_users = num_users
        self.num_items = num_items
        self.negative_sampling_ratio = negative_sampling_ratio
        
        self.positive_edges_set = set()
        for edge in self.positive_edges:
            user_idx = edge[0].item()
            item_idx = edge[1].item() - num_users if edge[1] >= num_users else edge[1].item()
            self.positive_edges_set.add((user_idx, item_idx))
        
    def __len__(self):
        return len(self.positive_edges)
    
    def __getitem__(self, idx):
        pos_edge = self.positive_edges[idx]
        user_idx = pos_edge[0].item()
        if pos_edge[1] >= self.num_users:
            item_idx = pos_edge[1].item() - self.num_users
        else:
            item_idx = pos_edge[0].item() - self.num_users
            user_idx = pos_edge[1].item()
        
        negatives = []
        for _ in range(self.negative_sampling_ratio):
            while True:
                neg_item = np.random.randint(0, self.num_items - 1)
                neg_edge_tuple = (pos_edge[0].item(), neg_item)
                
                if (user_idx, neg_item) not in self.positive_edges_set:
                    negatives.append(torch.tensor([user_idx, neg_item]))
                    break
        
        return {
            'positive': torch.tensor([user_idx, item_idx]),
            'negatives': torch.stack(negatives)
        }
