import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def compute_recall_at_k(user_emb, item_emb, val_edges, train_edges, num_users, num_items, k_list=(10, 20)):
    """Compute recall at k for validation set."""
    recall_dict = {}
    # all_scores = torch.matmul(user_emb, item_emb.T)  # [num_users, num_items]
    
    val_edges_dict = {}
    for u, i in val_edges:
        if u not in val_edges_dict:
            val_edges_dict[u] = set()
        val_edges_dict[u].add(i)
    
    train_edges_dict = {}  
    for u, i in train_edges:
        if u not in train_edges_dict:
            train_edges_dict[u] = set()
        train_edges_dict[u].add(i)

    recall_at_k = {k: 0.0 for k in k_list}
    total_users = 0
    
    for user_id in range(num_users):
        user_val_items = val_edges_dict.get(user_id, set())
        if not user_val_items:
            continue
            
        total_users += 1

        # user_scores = all_scores[user_id].clone() # Use clone to avoid modifying original scores
        user_scores = torch.matmul(user_emb[user_id], item_emb.T)
        if user_id in train_edges_dict:
            # for train_item in train_edges_dict[user_id]:
            #     if 0 <= train_item < num_items: # Ensure train_item is a valid index
            #         user_scores[train_item] = float('-inf')
            user_scores[list(train_edges_dict[user_id])] = float('-inf')  # Mask out items in training set
                
        for k in k_list:
            if len(user_scores) < k:
                # Handle case where k is larger than the number of items
                # print(f"Warning: k={k} is larger than the number of available items for user {user_id}. Adjusting k.")
                actual_k = len(user_scores)
            else:
                actual_k = k
            
            if actual_k == 0:
                continue # Skip if no items to rank

            topk_items = torch.topk(user_scores, actual_k).indices.cpu().numpy()
            hit_count = len(set(topk_items) & user_val_items)
            recall_at_k[k] += hit_count / len(user_val_items)
            
    for k in k_list:
        recall_dict[f'Recall@{k}'] = recall_at_k[k] / total_users if total_users > 0 else 0.0
    return recall_dict

def get_val_metrics(model, data, edge_index, batch, num_users, num_items, device):
    """Compute validation metrics like AUC-ROC and AUC-PR."""
    user_emb, item_emb = model(edge_index)
    
    pos_user_idx = batch['positive'][:, 0]
    pos_item_idx = batch['positive'][:, 1]
    pos_scores = model.predict(pos_user_idx, pos_item_idx, user_emb, item_emb)
    
    neg_user_idx = batch['negatives'][:, :, 0]
    neg_item_idx = batch['negatives'][:, :, 1]
    
    neg_user_idx_flat = neg_user_idx.reshape(-1)
    neg_item_idx_flat = neg_item_idx.reshape(-1)
    neg_scores_flat = model.predict(neg_user_idx_flat, neg_item_idx_flat, user_emb, item_emb)
    
    val_scores_all = torch.cat([pos_scores, neg_scores_flat]).detach().cpu().numpy()
    val_labels_all = np.concatenate([
        np.ones_like(pos_scores.detach().cpu().numpy()), 
        np.zeros_like(neg_scores_flat.detach().cpu().numpy())
    ])
    
    val_auc = roc_auc_score(val_labels_all, val_scores_all)
    precision, recall, _ = precision_recall_curve(val_labels_all, val_scores_all)
    val_auc_pr = auc(recall, precision)
    
    return val_auc, val_auc_pr
