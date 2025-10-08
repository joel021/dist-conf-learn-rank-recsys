"""
package: recsysconfident.ml.fit_eval.rank_helper.py
"""
import torch
from recsysconfident.ml.models.torchmodel import TorchModel


def sample_unseen_item(seen: tuple, num_items: int, max_tries: int=20) -> int:

    tries = 0
    unseen_item_id = torch.randint(0, num_items, (1,)).item()
    while unseen_item_id in seen[0]: #tuple with: itemId, X|None, rating
        unseen_item_id = torch.randint(0, num_items, (1,)).item()
        tries += 1
        if tries >= max_tries:
            unseen_item_id = num_items #Set nonexistent item when not finding a negative sample.
            break
    return unseen_item_id

def learn_to_rank_step(model: TorchModel, model_input):

    users_ids, high_rank_items = model_input

    low_rank_items_idx = get_low_rank_items(users_ids, model.items_per_user, model.n_items)
    pos_scores = model(model_input)
    neg_scores = model((users_ids, low_rank_items_idx.to(users_ids.device)))

    return pos_scores[:,0], neg_scores[:,0]

def get_low_rank_items(user_ids: torch.Tensor, items_per_user: dict, num_items: int) -> torch.Tensor:

    neg_items_idxs = []
    for u_id in user_ids:
        seen = items_per_user[int(u_id)]
        unseen_item_id = sample_unseen_item(seen, num_items)
        neg_items_idxs.append(unseen_item_id)

    return torch.tensor(neg_items_idxs)

def bpr_loss(model, model_input):
    p_s, n_s = learn_to_rank_step(model, model_input)

    diff = p_s - n_s
    return -torch.log(torch.sigmoid(diff) + 1e-8).mean()
