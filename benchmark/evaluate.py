import torch
from torch_geometric.utils import structured_negative_sampling
import numpy as np

# helper method to get positive items for train/test sets
def get_positive_items(edge_index):
    """
    Return positive items for all users in form of list
    Parameters:
      edge_index (torch.Tensor): The edge index representing the user-item interactions.
    Returns:
      pos_items (torch.Tensor): A list containing the positive items for all users.
    """
    pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in pos_items:
            pos_items[user] = []
        pos_items[user].append(item)
    return pos_items


def evaluation(model, edge_index, sparse_edge_index, mask_index, k, lambda_val, bpr_loss):
    """
    Evaluates model loss and metrics including recall, precision on the 
    Parameters:
    model: LightGCN model to evaluate.
    edge_index (torch.Tensor): Edges for the split to evaluate.
    sparse_edge_index (torch.SparseTensor): Sparse adjacency matrix.
    mask_index(torch.Tensor): Edges to remove from evaluation, in the form of a list.
    k (int): Top k items to consider for evaluation.

    Returns: loss, recall, precision
        - loss: The loss value of the model on the given split.
        - recall: The recall value of the model on the given split.
        - precision: The precision value of the model on the given split.
    """
    # get embeddings and calculate the loss
    users_emb, users_emb_0, items_emb, items_emb_0 = model.forward(sparse_edge_index)
    edges = structured_negative_sampling(edge_index, contains_neg_self_loops=False)
    
    user_indices, pos_indices, neg_indices = edges[0], edges[1], edges[2]
    users_emb, users_emb_0 = users_emb[user_indices], users_emb_0[user_indices]
    pos_emb, pos_emb_0 = items_emb[pos_indices], items_emb_0[pos_indices]
    neg_emb, neg_emb_0 = items_emb[neg_indices], items_emb_0[neg_indices]

    loss = bpr_loss(users_emb, users_emb_0, pos_emb, pos_emb_0,
                    neg_emb, neg_emb_0, lambda_val).item()

    users_emb_w = model.users_emb.weight
    items_emb_w = model.items_emb.weight

    # set ratings matrix between every user and item, mask out existing ones
    rating = torch.matmul(users_emb_w, items_emb_w.T)

    for index in mask_index:
        user_pos_items = get_positive_items(index)
        masked_users = []
        masked_items = []
        for user, items in user_pos_items.items():
            masked_users.extend([user] * len(items))
            masked_items.extend(items)

        rating[masked_users, masked_items] = float("-inf")

    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users and actual ratings for evaluation
    users = edge_index[0].unique()
    test_user_pos_items = get_positive_items(edge_index)

    actual_r = [test_user_pos_items[user.item()] for user in users]
    pred_r = []

    for user in users:
        items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in items, top_K_items[user]))
        pred_r.append(label)
    
    pred_r = torch.Tensor(np.array(pred_r).astype('float'))
    

    correct_count = torch.sum(pred_r, dim=-1)
    # number of items liked by each user in the test set
    liked_count = torch.Tensor([len(actual_r[i]) for i in range(len(actual_r))])
    
    recall = torch.mean(correct_count / liked_count)
    precision = torch.mean(correct_count) / k


    return loss, recall, precision