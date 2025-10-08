"""
package: recsysconfident.ml.fit_eval.learn_rank_helper.py
"""
import torch
from recsysconfident.ml.models.torchmodel import TorchModel


def sample_unseen_item(seen: set, num_items: int, max_tries: int=20) -> int:

    tries = 0
    unseen_item_id = torch.randint(0, num_items, (1,)).item()
    while unseen_item_id in seen:
        unseen_item_id = torch.randint(0, num_items, (1,)).item()
        tries += 1
        if tries >= max_tries:
            unseen_item_id = num_items #Set nonexistent item when not finding a negative sample.
            break
    return unseen_item_id

def learn_to_rank_step(model: TorchModel, users_ids: torch.Tensor, high_rank_items: torch.Tensor):
    low_rank_items = get_low_rank_items(users_ids, model.items_per_user, model.n_items)
    pos_scores = model(users_ids, high_rank_items)
    neg_scores = model(users_ids, low_rank_items.to(users_ids.device))

    return pos_scores[:,0], neg_scores[:,0]

def get_low_rank_items(user_ids: torch.Tensor, items_per_user: dict, num_items: int) -> torch.Tensor:

    neg_items = []
    for u_id in user_ids:
        seen = items_per_user[int(u_id)]
        unseen_item_id = sample_unseen_item(seen, num_items)
        neg_items.append(unseen_item_id)

    return torch.tensor(neg_items)

def bpr_loss(model, user_ids, item_ids):
    p_s, n_s = learn_to_rank_step(model, user_ids, item_ids)

    diff = p_s - n_s
    return -torch.log(torch.sigmoid(diff) + 1e-8).mean()

