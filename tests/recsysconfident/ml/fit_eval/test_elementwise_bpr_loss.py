import unittest
import torch

from recsysconfident.ml.eval.learn_elementwise_loss import sample_unseen_item, get_low_rank_items


class TestElementWiseBPR(unittest.TestCase):

    def test_sample_unseen_item_success(self):
        seen_items = {0, 2, 4}
        num_total_items = 10
        unseen = sample_unseen_item(seen_items, num_total_items)
        self.assertNotIn(int(unseen), seen_items, "No seen users were sampled as unseen.")
        self.assertLess(int(unseen), num_total_items, "The unseen item is valid.")

    def test_sample_unseen_item_nonexistent_item(self):
        seen_items = {0, 1, 2}
        num_total_items = 3
        unseen = sample_unseen_item(seen_items, num_total_items)
        self.assertEqual(int(unseen), num_total_items, "The unseen item is not valid when not finding a negative item.")

    def test_sample_unseen_item_find_the_unique_neg(self):
        seen_items = {0, 1, 3}
        num_total_items = 4
        unseen = sample_unseen_item(seen_items, num_total_items)
        self.assertEqual(int(unseen), 2, "It finds the unique negative within the allowed tries.")

    def test_get_low_rank_items_success(self):
        user_ids = torch.tensor([0, 1, 0])
        items_per_user = {
            0: [1, 3, 5],
            1: [0, 2]
        }
        num_items = 6
        negative_samples = get_low_rank_items(user_ids, items_per_user, num_items)
        self.assertEqual(negative_samples.shape, user_ids.shape, "Find a negative or nonexistent item for each user.")
        self.assertEqual(negative_samples.dtype, torch.int64, "Keep the expected datatype.")
