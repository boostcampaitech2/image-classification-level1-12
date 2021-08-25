import torch

from model.metric import mask_total_accuracy


x = torch.tensor(
        [
            [0.2, 0.6, 0.2],
            [0.9, 0.05, 0.05]
        ]
    ), \
    torch.tensor(
        [
            [0.7, 0.5],
            [0.3, 0.4]
        ]
    ), \
    torch.tensor(
        [
            [0.2, 0.6, 0.7],
            [0.9, 0.05, 0.05]
        ]
    )

# target = torch.LongTensor([1, 0]), torch.FloatTensor([1., 0.]), torch.LongTensor([1, 1])
target = torch.LongTensor([1, 0]), torch.LongTensor([0, 1]), torch.LongTensor([2, 0])


if __name__ == "__main__":
    print(mask_total_accuracy(x, target))

