import unittest

import torch

from train_sequence_models import SeqDataset


class SequenceDatasetTest(unittest.TestCase):
    def test_padding_follows_valid_tokens(self):
        dataset = SeqDataset({1: [1, 2, 3, 4]}, num_items=5, max_len=5, mode="train")
        sequence, positive, _ = dataset[0]

        self.assertEqual(sequence.tolist(), [1, 0, 0, 0, 0])
        self.assertEqual(positive.item(), 2)
        self.assertEqual(torch.count_nonzero(sequence).item(), 1)

    def test_validation_and_test_targets_are_separate(self):
        sequences = {1: [1, 2, 3, 4, 5]}
        validation = SeqDataset(sequences, num_items=6, max_len=5, mode="val")
        test = SeqDataset(sequences, num_items=6, max_len=5, mode="test")

        val_sequence, val_target, _ = validation[0]
        test_sequence, test_target, _ = test[0]

        self.assertEqual(val_sequence.tolist(), [1, 2, 3, 0, 0])
        self.assertEqual(val_target.item(), 4)
        self.assertEqual(test_sequence.tolist(), [1, 2, 3, 4, 0])
        self.assertEqual(test_target.item(), 5)


if __name__ == "__main__":
    unittest.main()
