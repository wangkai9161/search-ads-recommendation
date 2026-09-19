import unittest

import torch

from train_sequence_models import SeqDataset


class SequenceDatasetTest(unittest.TestCase):
    def test_padding_follows_valid_tokens(self):
        dataset = SeqDataset({1: [1, 2, 3, 4]}, num_items=5, max_len=5, mode="train")
        sequence, positive, _ = dataset[0]

        self.assertEqual(sequence.tolist(), [1, 2, 0, 0, 0])
        self.assertEqual(positive.item(), 3)
        self.assertEqual(torch.count_nonzero(sequence).item(), 2)


if __name__ == "__main__":
    unittest.main()
