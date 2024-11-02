import torch

vals = torch.tensor([1, 3])

print(torch.nn.functional.one_hot(vals, num_classes=3))