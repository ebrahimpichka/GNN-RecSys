import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import argparse
import os # Added import
from .data_utils import EdgeDataset, download_dataset, load_movielens_100k, load_movielens_1m, create_graph_data, split_edges, process_yelp_data_in_chunks, create_graph_from_chunks, split_yelp_edges # Relative import
from .evaluate import compute_recall_at_k, get_val_metrics # Relative import
from .model import LightGCN # Relative import

def bpr_loss(pos_scores, neg_scores):
    """
    Bayesian Personalized Ranking (BPR) loss function.
    
    Args:
        pos_scores: Scores for positive edges
        neg_scores: Scores for negative edges
        
    Returns:
        loss: BPR loss value
    """
    diff = pos_scores.unsqueeze(1) - neg_scores
    loss = -torch.log(torch.sigmoid(diff))
    return loss.mean()

def train_lightgcn(model, data, dataset_name, epochs=200, batch_size=1024, learning_rate=0.001, 
                   weight_decay=1e-4, device='cuda', negative_sampling_ratio=1, recall_k_list=[10, 20],
                   save_path=None, save_best=True):
    """
    Train LightGCN model.
    
    Args:
        model: LightGCN model
        data: PyG Data object with edge_index and masks
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to run training on
        negative_sampling_ratio: Ratio of negative samples to positive samples
        
    Returns:
        training_history: Dictionary containing training metrics
    """
    model.to(device)
    data.to(device)
    
    num_users = model.num_users
    num_items = model.num_items
    
    train_dataset = EdgeDataset(data.edge_index, data.train_mask, num_users, num_items, negative_sampling_ratio)
    val_dataset = EdgeDataset(data.edge_index, data.val_mask, num_users, num_items, negative_sampling_ratio)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_auc_pr': [],
        'layer_weights': [],
        'recall_at_k': []
    }
    
    best_val_auc = 0
    patience = 20
    patience_counter = 0
    
    edge_index = data.edge_index.to(device)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training')
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            user_emb, item_emb = model(edge_index)
            
            pos_user_idx = batch['positive'][:, 0]
            pos_item_idx = batch['positive'][:, 1]
            pos_scores = model.predict(pos_user_idx, pos_item_idx, user_emb, item_emb)
            
            neg_user_idx = batch['negatives'][:, :, 0]
            neg_item_idx = batch['negatives'][:, :, 1]
            
            neg_user_idx_flat = neg_user_idx.reshape(-1)
            neg_item_idx_flat = neg_item_idx.reshape(-1)
            neg_scores_flat = model.predict(neg_user_idx_flat, neg_item_idx_flat, user_emb, item_emb)
            neg_scores = neg_scores_flat.reshape(neg_user_idx.shape)
            
        
            loss = bpr_loss(pos_scores, neg_scores)
            

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'batch loss': loss.item()})
        
        avg_train_loss = total_train_loss / train_batches
        
        model.eval()
        total_val_loss = 0
        val_batches = 0
        val_scores = []
        val_labels = []
        
        with torch.no_grad():
            user_emb, item_emb = model(edge_index)
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                pos_user_idx = batch['positive'][:, 0]
                pos_item_idx = batch['positive'][:, 1]
                pos_scores = model.predict(pos_user_idx, pos_item_idx, user_emb, item_emb)
                
                neg_user_idx = batch['negatives'][:, :, 0]
                neg_item_idx = batch['negatives'][:, :, 1]
                
                neg_user_idx_flat = neg_user_idx.reshape(-1)
                neg_item_idx_flat = neg_item_idx.reshape(-1)
                neg_scores_flat = model.predict(neg_user_idx_flat, neg_item_idx_flat, user_emb, item_emb)
                neg_scores = neg_scores_flat.reshape(neg_user_idx.shape)
                
                val_loss = bpr_loss(pos_scores, neg_scores)
                total_val_loss += val_loss.item()
                val_batches += 1
                
                val_scores.extend(pos_scores.cpu().numpy())
                val_scores.extend(neg_scores.cpu().numpy().flatten())
                
                val_labels.extend(np.ones_like(pos_scores.cpu().numpy()))
                val_labels.extend(np.zeros_like(neg_scores.cpu().numpy().flatten()))
        
        avg_val_loss = total_val_loss / val_batches
        
        # Compute validation metrics using the new get_val_metrics function
        val_auc, val_auc_pr = get_val_metrics(model, data, edge_index, batch, num_users, num_items, device) # Updated call
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        history['val_auc_pr'].append(val_auc_pr)

        with torch.no_grad():
            # user_emb, item_emb = model(edge_index)
            val_edges = data.val_edges
            train_edges = data.train_edges  
            
            recall_metrics = compute_recall_at_k(
                user_emb=user_emb.cpu(),
                item_emb=item_emb.cpu(),
                val_edges=val_edges,
                train_edges=train_edges,
                num_users=num_users,
                num_items=num_items,
                k_list=recall_k_list
            )
        
        history['recall_at_k'].append(recall_metrics)      
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val AUC-ROC: {val_auc:.4f}")
        print(f"Val AUC-PR: {val_auc_pr:.4f}")
        for metric_name, value in recall_metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print("-" * 50)

        if not save_path:
            save_path = f"best_lightgcn_model_{dataset_name}.pth"
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            if save_best:
                print(f"Saving best model to {save_path}")
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    if save_best:
        print(f"Loading best model from {save_path}")
        model.load_state_dict(torch.load(save_path))
    
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LightGCN model')
    parser.add_argument('--dataset_name', type=str, default='movielens-100k', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--negative_sampling_ratio', type=int, default=1, help='Negative sampling ratio')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the best model')

    args = parser.parse_args()
    dataset_name = args.dataset_name
    device = args.device if torch.cuda.is_available() else 'cpu'

    dataset_path = download_dataset(dataset_name)
    graph_obj_path = dataset_name+'_processed_graph.pt'
    if os.path.exists(graph_obj_path):
        data = torch.load(graph_obj_path, weights_only=False)
        data.num_items = data.num_businesses
        data = data.to(device)
        print(f"Loaded processed graph data from {graph_obj_path}")
    else:
        if dataset_name.lower() == 'movielens-100k':
            user_nodes, item_nodes, num_users, num_items = load_movielens_100k(dataset_path)
        elif dataset_name.lower() == 'movielens-1m':
            user_nodes, item_nodes, num_users, num_items = load_movielens_1m(dataset_path)
        elif dataset_name.lower() == 'yelp':
            review_data_file = os.path.join(dataset_path, 'yelp_academic_dataset_review.json')
            process_yelp_data_in_chunks(review_data_file, chunk_size=50000)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")
            

        print("Creating graph data and splitting edges into train/val/test sets...")
        if dataset_name.lower() == 'yelp':
            data = create_graph_from_chunks(
                'processed_chunks',
                user_mapping_file='user_to_idx.pkl',
                business_mapping_file='business_to_idx.pkl'
                )
            data = split_yelp_edges(data, test_ratio=0.2, val_ratio=0.05, save_path=graph_obj_path)
        else:   
            data = create_graph_data(user_nodes, item_nodes, num_users, num_items)
            data = split_edges(data, test_ratio=0.2, val_ratio=0.05)
        print("Graph data created")        
        print("Edge splitting completed\n----\n")
        
    
    num_users = data.num_users
    num_items = data.num_items

    print(f"Loaded {dataset_name} dataset with {num_users} users and {num_items} items")
    print(f"Number of user nodes: {num_users}")
    print(f"Number of item nodes: {num_items}")
    print(f"Number of edges: {data.edge_index.size(1)//2}")

    data = data.to(args.device)
    model = LightGCN(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers
    ).to(args.device)

    train_lightgcn(
        model,
        data,
        args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        negative_sampling_ratio=args.negative_sampling_ratio,
        save_path=args.save_path
    )
