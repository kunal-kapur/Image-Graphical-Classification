from gnn import GNN
import torch
import os
from dataloader import AnimalsDatasetParquet
import matplotlib.pyplot as plt
import numpy as np


# Create an instance of your model
# model = GNN(k=3) 

# # Load the state dictionary from the saved file
# model.load_state_dict(torch.load(os.path.join("results", "animals_dist25_k2_epochs18_schedule6",
#                                              "model_weights.pth"))) 

dist = 5
data = AnimalsDatasetParquet(f"animals_{dist}.parquet")

# # coords, inputs, targets, paths = data[0]

# for i in range(0, 1000):
#     coords, inputs, targets, paths = data[i]
#     # if paths == "/Users/kunalkapur/Workspace/cs593-proj/Animals/cats/0_0053.jpg":
#     #     print("HERE")
#     model.forward(torch.unsqueeze(input=inputs, dim=0), intermediate_graphs=True, coordinates=coords, image_path=paths)


num_dict = {}
for i in data:
    coords, inputs, targets, paths = i
    num = inputs.shape[0]
    num_dict[num] = num_dict.get(num, 0) + 1

mylist = [key for key, val in num_dict.items() for _ in range(val)]
plt.hist(mylist, bins=8)
plt.xlabel("Number of key points") 
plt.title(f'Harris Distance {dist}')
plt.savefig(f'hist{dist}.png') 
